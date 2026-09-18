#!/usr/bin/env python3
"""实验: MiniMind-3 (68.8M, CPU 单线程 numpy) 当逐位置条件分布源。

目的: 验证「模型在推理框架里跑, 本项目拿 logit softmax 建七叉树」的可行性:
    - 单线程 CPU 每字符前向耗时(带宽是人工吹奏, 百 ms 级即够);
    - 模型字符级条件分布 + order-0 回退 混合建树的实测码长 vs 固定字频表;
    - token 级交叉熵作模型下限(算术编码可达), 分离"建树/聚合粒度损失"。

聚合说明: 词表 6400 是 ByteLevel BPE, 高频字 71% 为单 token(字频覆盖
92%); 多 token 字的概率按字节前缀归属碎片 token 后按 P0 分摊(近似,
略悲观)。权重: ModelScope gongjy/minimind-3-pytorch (pretrain/full_sft_768.pth)。
"""
import csv
import math
import os
import pickle
import sys
import time
import zipfile
from collections import OrderedDict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

K = 7
Q = 1 << 20                      # 概率定点化分母
S_CAP = int(0.99 * Q)            # 模型质量占比上限, 其余留给 order-0 回退


# --- 1. 无 torch 读 .pth (zip + pickle 协议) -----------------------------------

_DTYPE = {"FloatStorage": np.float32, "HalfStorage": np.float16,
          "BFloat16Storage": np.uint16, "LongStorage": np.int64,
          "IntStorage": np.int32, "ByteStorage": np.uint8}


class _Stub:
    def __init__(self, *a, **k):
        pass


def _contig(size):
    s, acc = [], 1
    for d in reversed(size):
        s.append(acc)
        acc *= d
    return list(reversed(s))


def load_pth(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[:-len("data.pkl")]
    storages = {}

    def rebuild(key, offset, size, stride):
        dt, _ = storages[key]
        raw = zf.read(prefix + "data/" + key)
        base = np.frombuffer(raw, dtype=dt)
        if dt == np.uint16:                       # bf16 -> fp32
            base = (base.astype(np.uint32) << 16).view(np.float32)
        else:
            base = base.astype(np.float32)
        size, stride = list(size), list(stride)
        if not size:
            return float(base[offset])
        if stride == _contig(size):
            return base[offset:offset + int(np.prod(size))].reshape(size)
        return np.lib.stride_tricks.as_strided(
            base[offset:], size, np.array(stride) * base.itemsize).copy()

    def _rebuild_tensor_v2(key, offset, size, stride, *a, **k):
        return rebuild(key, offset, size, stride)

    def _rebuild_parameter(data, *a, **k):
        return data

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch") and name in _DTYPE:
                return type(name, (), {})
            if module == "torch._utils":
                return {"_rebuild_tensor_v2": _rebuild_tensor_v2,
                        "_rebuild_parameter": _rebuild_parameter}[name]
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            return _Stub

        def persistent_load(self, pid):
            _, dtype_cls, key, _, numel = pid
            storages[key] = (_DTYPE[dtype_cls.__name__], numel)
            return key

    return U(zf.open(pkl), encoding="latin1").load()


# --- 2. numpy 前向 (对照 model/model_minimind.py 移植) -------------------------

CFG = dict(H=768, L=8, NH=8, NKV=4, HD=96, EPS=1e-6, THETA=1e6, VOCAB=6400,
           INTER=2432)


class MiniMindNumpy:
    def __init__(self, path):
        sd = load_pth(path)
        H, L, NH, NKV, HD = CFG["H"], CFG["L"], CFG["NH"], CFG["NKV"], CFG["HD"]
        self.E = np.asarray(sd["model.embed_tokens.weight"], np.float32)
        assert self.E.shape == (CFG["VOCAB"], H)
        g = lambda n, s: self._get(sd, n, s)
        self.layers = []
        for i in range(L):
            p = f"model.layers.{i}."
            self.layers.append(dict(
                ln1=g(p + "input_layernorm.weight", (H,)),
                ln2=g(p + "post_attention_layernorm.weight", (H,)),
                wq=g(p + "self_attn.q_proj.weight", (NH * HD, H)),
                wk=g(p + "self_attn.k_proj.weight", (NKV * HD, H)),
                wv=g(p + "self_attn.v_proj.weight", (NKV * HD, H)),
                wo=g(p + "self_attn.o_proj.weight", (H, NH * HD)),
                qn=g(p + "self_attn.q_norm.weight", (HD,)),
                kn=g(p + "self_attn.k_norm.weight", (HD,)),
                w_gate=g(p + "mlp.gate_proj.weight", (CFG["INTER"], H)),
                w_up=g(p + "mlp.up_proj.weight", (CFG["INTER"], H)),
                w_down=g(p + "mlp.down_proj.weight", (H, CFG["INTER"])),
            ))
        self.norm_f = self._get(sd, "model.norm.weight", (H,))
        inv = 1.0 / (CFG["THETA"] ** (np.arange(0, HD, 2)[: HD // 2] / HD))
        t = np.arange(32768, dtype=np.float32)
        f = np.outer(t, inv)
        self.cos = np.concatenate([np.cos(f), np.cos(f)], -1)
        self.sin = np.concatenate([np.sin(f), np.sin(f)], -1)

    @staticmethod
    def _get(sd, name, shape):
        w = np.asarray(sd[name], dtype=np.float32)
        assert w.shape == tuple(shape), f"{name}: {w.shape} != {tuple(shape)}"
        return w

    @staticmethod
    def _rms(x, w, eps):
        return w * (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))

    def forward(self, ids, cache=None, last_only=True):
        """ids: [T]; cache: 每层 (k,v) [past,NKV,HD]。返回 (logits, 新 cache)。

        last_only=True 只回最后位置; False 回全部位置(teacher forcing 用)。"""
        H, NH, NKV, HD = CFG["H"], CFG["NH"], CFG["NKV"], CFG["HD"]
        T = len(ids)
        past = 0 if cache is None else cache[0][0].shape[0]
        pos = np.arange(past, past + T)
        x = self.E[np.asarray(ids, dtype=np.int64)]
        nrep = NH // NKV
        new_cache = []
        for li, lw in enumerate(self.layers):
            k_cache = None if cache is None else cache[li][0]
            v_cache = None if cache is None else cache[li][1]
            h = self._rms(x, lw["ln1"], CFG["EPS"])
            q = (h @ lw["wq"].T).reshape(T, NH, HD)
            k = (h @ lw["wk"].T).reshape(T, NKV, HD)
            v = (h @ lw["wv"].T).reshape(T, NKV, HD)
            q = self._rms(q, lw["qn"], CFG["EPS"])
            k = self._rms(k, lw["kn"], CFG["EPS"])
            c, s = self.cos[pos], self.sin[pos]
            q = q * c[:, None, :] + np.concatenate(
                (-q[..., HD // 2:], q[..., : HD // 2]), -1) * s[:, None, :]
            k = k * c[:, None, :] + np.concatenate(
                (-k[..., HD // 2:], k[..., : HD // 2]), -1) * s[:, None, :]
            k = np.concatenate([k_cache, k]) if k_cache is not None else k
            v = np.concatenate([v_cache, v]) if v_cache is not None else v
            new_cache.append((k, v))
            kk = np.repeat(k, nrep, axis=1)
            vv = np.repeat(v, nrep, axis=1)
            att = np.einsum("thd,shd->hts", q, kk) / math.sqrt(HD)
            tri = np.triu(np.ones((T, past + T)), 1 + past) > 0
            att[:, tri] = -1e9
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = np.einsum("hts,shd->thd", att, vv).reshape(T, NH * HD)
            x = x + o @ lw["wo"].T
            h = self._rms(x, lw["ln2"], CFG["EPS"])
            g_ = h @ lw["w_gate"].T
            x = x + (g_ / (1 + np.exp(-g_)) * (h @ lw["w_up"].T)) @ lw["w_down"].T
        if last_only:
            x = x[-1:]
        logits = self._rms(x, self.norm_f, CFG["EPS"]) @ self.E.T
        return logits, new_cache


# --- 3. 字节级词表 -> 字符归属 --------------------------------------------------

def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_bytes(tok, n_vocab):
    """token id -> 原始字节串(ByteLevel 逆映射); 无法逆映射 -> None。"""
    u2b = {v: k for k, v in _bytes_to_unicode().items()}
    out = {}
    for i in range(n_vocab):
        s = tok.id_to_token(i)
        try:
            out[i] = bytes(u2b[ch] for ch in s)
        except KeyError:
            out[i] = None
    return out


def char_paths(tok, alphabet):
    """字母表字符 -> token 路径(发送/回喂约定一致)。"""
    return {c: tok.encode(c, add_special_tokens=False).ids
            for c, _ in alphabet if c != "\ue000"}


# --- 4. order-0 字母表 与七叉哈夫曼 ---------------------------------------------

def load_p0():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from build_huffman_table import DEFAULT_CSV, SYNTHETIC
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = list(csv.reader(open(os.path.join(root, DEFAULT_CSV), encoding="utf-8")))
    alphabet = [(r[1], int(r[2])) for r in rows[1:] if len(r[1]) == 1]
    alphabet += [(c, int(w)) for c, w in SYNTHETIC]
    total = sum(w for _, w in alphabet)
    alphabet.append(("\ue000", total // 256))                 # ESC
    alphabet.sort(key=lambda t: (-t[1], t[0]))
    return alphabet


def huff7_codes(weights):
    """正整数权重 -> 规范七叉码 [(码长, 数字元组)] 按字母表序。"""
    import heapq
    n = len(weights)
    dummies = (-(n - 1)) % (K - 1)
    ww = list(weights) + [0] * dummies
    nodes = [None] * (n + dummies)
    heap = [(w, i) for i, w in enumerate(ww)]
    heapq.heapify(heap)
    while len(heap) > 1:
        children = [heapq.heappop(heap)[1] for _ in range(K)]
        sw = sum(ww[c] for c in children)
        ww.append(sw)
        nodes.append(children)
        heapq.heappush(heap, (sw, len(nodes) - 1))
    lengths = [0] * n
    stack = [(len(nodes) - 1, 0)]
    while stack:
        nd, d = stack.pop()
        if nodes[nd] is None:
            if nd < n:
                lengths[nd] = d
        else:
            for c in nodes[nd]:
                stack.append((c, d + 1))
    out = [None] * n
    prev_end = 0
    for Lv in range(1, max(lengths) + 1):
        idxs = [i for i in range(n) if lengths[i] == Lv]
        start = prev_end * K
        for j, i in enumerate(idxs):
            code = start + j
            out[i] = (Lv, tuple((code // K ** (Lv - 1 - k)) % K
                                for k in range(Lv)))
        prev_end = start + len(idxs)
    return out


# --- 5. 逐位置编码器 ------------------------------------------------------------

class PositionalCoder:
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.alphabet = load_p0()
        self.P0W = [w for _, w in self.alphabet]
        self.P0TOT = sum(self.P0W)
        self.ch2idx = {c: i for i, (c, _) in enumerate(self.alphabet)}
        self.paths = char_paths(tok, self.alphabet)          # 字 -> token 路径
        self.tbytes = token_bytes(tok, CFG["VOCAB"])
        # 单 token 字: id -> 字
        self.id2ch = {}
        # 碎片 token -> [(字符, 字节前缀长)] (多 token 字的首段)
        self.frag = {}
        for c, path in self.paths.items():
            if len(path) == 1 and self.tbytes.get(path[0]) is not None:
                self.id2ch[path[0]] = c
            elif len(path) >= 2:
                self.frag.setdefault(path[0], []).append(c)
        self._codes = {}
        self._path_cache = {}

    def _char_dist(self, prob):
        """token softmax -> 字符分布(单 token 直取; 碎片质量按 P0 分摊候选字)。"""
        char_p = {}
        frag_mass = 0.0
        for i in range(CFG["VOCAB"]):
            p = float(prob[i])
            if p < 1e-9:
                continue
            c = self.id2ch.get(i)
            if c is not None:
                char_p[c] = char_p.get(c, 0.0) + p
            elif i in self.frag:
                frag_mass += p
        if frag_mass > 0:
            for i, cands in self.frag.items():
                p = float(prob[i])
                if p < 1e-9:
                    continue
                cand = [c for c in cands if c in self.ch2idx]
                tot = sum(self.P0W[self.ch2idx[c]] for c in cand)
                for c in cand:
                    char_p[c] = char_p.get(c, 0.0) + \
                        p * self.P0W[self.ch2idx[c]] / tot
        return char_p

    def _mix(self, char_p):
        """字符分布 -> 全字母表整数权重(未覆盖字按 P0 归一回退)。"""
        seen = {self.ch2idx[c]: int(p * Q) for c, p in char_p.items()
                if c in self.ch2idx and c != "\ue000"}
        S = sum(seen.values())
        if S > S_CAP:
            f = S_CAP / S
            S = S_CAP
            seen = {i: int(pi * f) for i, pi in seen.items()}
        w = list(self.P0W)
        for i, pi in seen.items():
            w[i] = pi * self.P0TOT // Q
        seen_p0 = sum(self.P0W[i] for i in seen)
        rem, denom = Q - S, self.P0TOT - seen_p0
        for i in range(len(w)):
            if w[i] == 0:
                w[i] = max(1, rem * self.P0W[i] // denom)
        return w

    def _codes_for(self, char_p):
        key = tuple(sorted(char_p.items()))
        cs = self._codes.get(key)
        if cs is None:
            cs = self._codes[key] = huff7_codes(self._mix(char_p))
        return cs

    def encode(self, text, bos=1):
        """逐字: 前向 -> 字符分布 -> 建树 -> 编码 -> 回喂该字 token。
        返回 (digits, 累计交叉熵 bit, 模型 ms, 建树 ms)。"""
        cache = None
        digits = []
        ce = t_model = t_tree = 0.0
        t0 = time.perf_counter()
        logits, cache = self.model.forward([bos])
        t_model += time.perf_counter() - t0
        for ch in text:
            t0 = time.perf_counter()
            full = np.exp(logits.ravel() - logits.max())
            prob = full / full.sum()
            char_p = self._char_dist(prob)
            t1 = time.perf_counter()
            codes = self._codes_for(char_p)
            t_tree += time.perf_counter() - t1
            idx = self.ch2idx.get(ch)
            if idx is None:                      # 表外字: ESC + UTF-8 字节
                digits += codes[self.ch2idx["\ue000"]][1]
                for b in ch.encode("utf-8"):
                    digits += (b // (K * K), (b // K) % K, b % K)
                ce += 15.0                        # 粗略占位: 表外字按 ESC+24bit 记
            else:
                digits += codes[idx][1]
                p_c = char_p.get(ch, 0.0)
                ce -= math.log2(max(p_c, 1e-12))
            t0 = time.perf_counter()
            logits, cache = self.model.forward(self.path_of(ch), cache)
            t_model += time.perf_counter() - t0
        return digits, ce, t_model, t_tree

    def path_of(self, ch):
        """任意字符 -> token 路径(字节级回退); 缓存。两端约定一致。"""
        p = self._path_cache.get(ch)
        if p is None:
            p = self.tok.encode(ch, add_special_tokens=False).ids or [0]
            self._path_cache[ch] = p
        return p

    def token_ce(self, text, bos=1):
        """token 级 teacher-forcing 交叉熵(bit/字)。"""
        ids = [bos] + self.tok.encode(text, add_special_tokens=False).ids
        logits, _ = self.model.forward(ids, last_only=False)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        bit = 0.0
        for t in range(1, len(ids)):
            bit -= math.log2(max(float(p[t - 1, ids[t]]), 1e-12))
        return bit / max(len(text), 1)


# --- 6. 主流程 -----------------------------------------------------------------

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.environ.get(
        "MINIMIND_TOKENIZER", "/tmp/minimind/model/tokenizer.json"))
    from huffman_codec import _SAMPLES, text_to_digits as o0_digits

    samples = _SAMPLES + ["有人堵桥"]
    for tag, path in (("pretrain", "pretrain_768.pth"),
                      ("full_sft", "full_sft_768.pth")):
        model = MiniMindNumpy(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
        coder = PositionalCoder(model, tok)
        # 生成 sanity
        logits, cache = model.forward([1])
        gen = []
        for _ in range(20):
            nxt = int(np.argmax(logits))
            gen.append(nxt)
            if nxt == 2:
                break
            logits, cache = model.forward([nxt], cache)
        print(f"[{tag}] 生成: {tok.decode(gen)!r}")
        print(f"{'样例':<16} {'字':>3} {'tokbit/字':>9} {'o0':>5} {'o1':>5} "
              f"{'音符/字':>7} {'模型ms/字':>8} {'树ms/字':>7}")
        tot1 = totn = 0
        for text in samples:
            digits, ce, t_model, t_tree = coder.encode(text)
            ce_tok = coder.token_ce(text)
            n = max(len(text), 1)
            tot1 += len(digits)
            totn += n
            print(f"{text[:14]:<16} {len(text):>3} {ce_tok:>9.2f} "
                  f"{len(o0_digits(text)):>5} {len(digits):>5} "
                  f"{len(digits) / n:>7.2f} {t_model * 1000 / n:>8.0f} "
                  f"{t_tree * 1000 / n:>7.0f}")
        print(f"[{tag}] 合计: {tot1}/{totn} = {tot1 / totn:.2f} 数字/字"
              f"  (order-0 对照: 3.42)\n")


if __name__ == "__main__":
    main()
