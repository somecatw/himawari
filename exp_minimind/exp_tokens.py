#!/usr/bin/env python3
"""实验: token 级逐位置建树 —— 信源符号 = token ID, 模型分布直接建七叉树。

方案: text --tok.encode--> token ID 序列; 每位置 前向 -> softmax -> 定点化
(下限 1e-6 平滑) -> 6400 符号七叉规范哈夫曼 -> 当前 token 的码字 -> 数字
-> 旋转层(1 数字 = 1 音符)。解码端对称: 用已解出 token 前缀重建同一棵树,
数字流走树得 token ID, 字节拼接还原文本(无重分词歧义)。

对照: order-0 字符表(huffman_codec) / 字符级逐位置树(exp.py) / token 级
CE 下限(算术编码可达) / ts_zip(README 实测 2.45 音符/字 @187字)。
"""
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import exp as E

Q = 1 << 20
EPS = 1e-6                     # 分布下限平滑: 任何 token 的码长 <= ~7 数字


class TokenCoder:
    def __init__(self, model, tok):
        self.m = model
        self.tok = tok
        self.tbytes = E.token_bytes(tok, E.CFG["VOCAB"])   # id -> 原始字节

    def _weights(self, logits):
        z = logits.ravel() - logits.max()
        p = np.exp(z)
        p /= p.sum()
        w = np.maximum(p, EPS) * Q
        return np.maximum(1, w.astype(np.int64)).tolist()

    @staticmethod
    def _tree(w):
        codes = E.huff7_codes(w)
        lookup = {digs: i for i, (_, digs) in enumerate(codes)}
        return codes, lookup, max(L for L, _ in codes)

    def encode(self, text, bos=1):
        """-> (digits, token 序列, 模型 ms/token, 建树 ms/token)"""
        ids = self.tok.encode(text, add_special_tokens=False).ids
        cache = None
        logits, cache = self.m.forward([bos])
        digits, t_model, t_tree = [], 0.0, 0.0
        for t in ids:
            t0 = time.perf_counter()
            codes, _, _ = self._tree(self._weights(logits))
            t1 = time.perf_counter()
            digits += codes[t][1]
            logits, cache = self.m.forward([t], cache)
            t_model += time.perf_counter() - t1
            t_tree += t1 - t0
        return digits, ids, t_model / max(len(ids), 1), t_tree / max(len(ids), 1)

    def decode(self, digits, bos=1):
        """数字流 -> (文本, token 数, 是否边界收尾)。失步时丢 1 数字再同步。"""
        logits, cache = self.m.forward([bos])
        ids = []
        i, bad = 0, 0
        while i < len(digits):
            codes, lookup, max_len = self._tree(self._weights(logits))
            seq = []
            hit = None
            while i < len(digits):
                seq.append(digits[i])
                i += 1
                hit = lookup.get(tuple(seq))
                if hit is not None or len(seq) >= max_len:
                    break
            if hit is None:                     # 窗口耗尽仍是半码 -> 丢队首数字
                bad += 1
                i -= len(seq) - 1
                if i >= len(digits):
                    break
                continue
            if hit is not None:
                ids.append(hit)
                logits, cache = self.m.forward([hit], cache)
        text = b"".join(self.tbytes[j] or b"" for j in ids).decode(
            "utf-8", errors="replace")
        return text, ids, (i >= len(digits) and bad == 0)


K7 = 7


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.environ.get(
        "MINIMIND_TOKENIZER", "/tmp/minimind/model/tokenizer.json"))
    from huffman_codec import _SAMPLES, text_to_digits as o0_digits

    model = E.MiniMindNumpy(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pretrain_768.pth"))
    tc = TokenCoder(model, tok)
    cc = E.PositionalCoder(model, tok)          # 字符级对照

    samples = _SAMPLES + ["有人堵桥",
                          "这段消息用来测试哈夫曼编码在较长中文文本下的平均码长，"
                          "包括标点、数字 123 与 English。",
                          "周六下午三点在老地方进行团建，请各位准时到达，迟到的"
                          "话自己补零食。活动内容包括桌游、烧烤和一场小型的编程"
                          "比赛，欢迎大家带朋友来玩。如果天气不好就改到下周，具体"
                          "安排会在群里提前一天通知大家。"]

    print(f"{'样例':<15} {'字':>3} {'tok':>4} {'o0':>5} {'字符树':>5} {'token树':>5} "
          f"{'CE折算':>6} {'tk数字/字':>7} {'模型ms/t':>7} {'树ms/t':>6} 往返")
    sum_tk = sum_n = 0
    for text in samples:
        d0 = o0_digits(text)
        d1, ids1, _, _ = cc.encode(text)
        t0 = time.perf_counter()
        d2, ids, tpm, tpt = tc.encode(text)
        wall = time.perf_counter() - t0
        back, ids2, ok = tc.decode(d2)
        okrt = (back == text and ids2 == ids)
        ce = cc.token_ce(text) / math.log2(7) * len(text)   # 折算成数字
        n = max(len(text), 1)
        sum_tk += len(d2)
        sum_n += n
        print(f"{text[:13]:<15} {len(text):>3} {len(ids):>4} {len(d0):>5} "
              f"{len(d1):>5} {len(d2):>5} {ce:>6.0f} {len(d2)/n:>7.2f} "
              f"{tpm*1000:>7.0f} {tpt*1000:>6.0f} {'✓' if okrt else '✗'}")
    print(f"合计: token树 {sum_tk}/{sum_n} = {sum_tk/sum_n:.2f} 数字/字"
          f"  (o0 3.42, 字符树 3.24, CE 下限 ~1.7)\n")

    # --- 码长分布(94 字样例) ---
    long_text = samples[-1]
    ids = tok.encode(long_text, add_special_tokens=False).ids
    cache = None
    logits, cache = model.forward([1])
    hist = {}
    for t in ids:
        w = tc._weights(logits)
        codes, _, _ = tc._tree(w)
        L = codes[t][0]
        hist[L] = hist.get(L, 0) + 1
        logits, cache = model.forward([t], cache)
    print("token 码长分布(94字):",
          {k: hist[k] for k in sorted(hist)})

    # --- 损伤探针: 单数字删除 -> 解码传播 ---
    d2, ids, _, _ = tc.encode(long_text)
    n0 = len(d2)
    for frac in (0.1, 0.3, 0.6):
        k = int(n0 * frac)
        damaged = d2[:k] + d2[k + 1:]
        back, ids2, _ = tc.decode(damaged)
        lcp = 0
        for a, b in zip(back, long_text):
            if a != b:
                break
            lcp += 1
        print(f"删第{k}位数字: 解出 {len(back)} 字, 正确前缀 {lcp} 字"
              f" (原文 {len(long_text)} 字) -> {'前缀存活' if lcp else '全毁'}")


if __name__ == "__main__":
    main()
