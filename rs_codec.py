#!/usr/bin/env python3
"""GF(7) RS 纠错 + 停顿分帧 —— 发送端信道层最外级。

三种模式(接收端三路并行解码, 无报头, 哈夫曼 EOF 终止):
    no      huffman_codec 链路原样: mod-8 旋转, 无分帧          1.00 音符/数字
    medium  帧 = RS(7,5): 5 数据 + 2 校验, 停顿分帧             ~1.14 + 停顿时间
    high    帧 = RS(7,4): 4 数据 + 3 校验, 停顿分帧             ~1.14

分帧: 每 7 个 GF(7) 数字一个系统码字, 恒等映射到 C4..B4, 相邻重复数字垫
C5(转义, 纯声学防粘, 不含语义), 帧与帧之间由演奏者停顿分隔(提示中的 R)。
可靠模型: 只有音高变化是可靠事件, 所以
    - 帧界用停顿(静音)而非特殊音符: 静音与音符本质可区分;
    - 帧内用恒等映射而非旋转: 旋转差分下吞一个音符会让差分尾巴整体错位
      (超出 RS 预算); 恒等映射下任何吞音/多音都退化为"帧内计数 ≠ 7"的
      局部损伤, 恰好落在预算内。
转义 C5 被吞 -> 相邻同音合并 -> 也是 1 个数字的删除, 同样预算内可修。

接收端解析(帧级 beam, 帧间零耦合): 每段停顿间数字独立修复, 计数 ≠ 7 时
枚举有界假设(吞音 -> 擦除位; 多音 -> 丢弃); 跨段组合有界截断; 多解释由
消息级(哈夫曼 EOF + 似然 + 人眼)终审。

命令行:
    python rs_codec.py encode "你好" [--level no|medium|high] [--midi]
    python rs_codec.py decode "G4 F4 ... R ..."   # R = 停顿(帧界)
    python rs_codec.py simulate [--trials N] [--sub p] [--del p] [--ins p]
    python rs_codec.py selftest
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from itertools import combinations

from huffman_codec import (CODE_OF, NOTES8, MIDI8, EOF, ESC,
                           _read_text, _read_notes,
                           text_to_digits as _huff_digits,
                           text_to_notes as _no_encode,
                           notes_to_digits as _no_diff,
                           _LOOKUP as _HUFF_LOOKUP, _MAX_CODE_LEN as _HUFF_MAXLEN,
                           _decode_escaped as _huff_escaped)

K = 7
DELIM = "C5"
DATA7 = ("C4", "D4", "E4", "F4", "G4", "A4", "B4")
_IDX7 = {n: i for i, n in enumerate(DATA7)}
LEVEL_K = {"no": None, "medium": 5, "high": 4}
BEAM_WIDTH = 256
PAUSE = "R"                                  # 停顿标记(帧界, 演奏者呼吸)


# --- GF(7) 与系统 RS 码 ------------------------------------------------------

_INV = [0] * K
for _a in range(1, K):
    _INV[_a] = pow(_a, K - 2, K)


class RSError(ValueError):
    pass


def _peval(coeffs, x):
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % K
    return acc


def _pdivmod(num, den):
    dn = len(den) - 1
    while dn >= 0 and den[dn] == 0:
        dn -= 1
    num = list(num)
    q = [0] * max(1, len(num) - dn)
    for i in range(len(num) - 1, dn - 1, -1):
        if num[i] == 0:
            continue
        f = num[i] * _INV[den[dn]] % K
        q[i - dn] = f
        for j in range(dn + 1):
            num[i - dn + j] = (num[i - dn + j] - f * den[j]) % K
    return q, num[:dn]


def _nullspace(rows, ncols):
    m = [r[:] for r in rows]
    pivots, r0 = [], 0
    for c in range(ncols):
        piv = next((i for i in range(r0, len(m)) if m[i][c] % K), None)
        if piv is None:
            continue
        m[r0], m[piv] = m[piv], m[r0]
        inv = _INV[m[r0][c] % K]
        m[r0] = [(v * inv) % K for v in m[r0]]
        for i in range(len(m)):
            if i != r0 and m[i][c] % K:
                f = m[i][c]
                m[i] = [(a - f * b) % K for a, b in zip(m[i], m[r0])]
        pivots.append(c)
        r0 += 1
        if r0 == len(m):
            break
    basis = []
    for fc in [c for c in range(ncols) if c not in pivots]:
        v = [0] * ncols
        v[fc] = 1
        for ri, pc in enumerate(pivots):
            v[pc] = (-m[ri][fc]) % K
        basis.append(v)
    return basis


_PMAT: dict[int, list] = {}


def _parity_matrix(k):
    pts = list(range(k))
    pm = []
    for x in range(k, K):
        row = []
        for j in range(k):
            num = den = 1
            for m in range(k):
                if m != j:
                    num = num * (x - pts[m]) % K
                    den = den * (pts[j] - pts[m]) % K
            row.append(num * _INV[den] % K)
        pm.append(row)
    return pm


def rs_encode_block(data, k):
    """k 个 GF(7) 数据数字 -> 7 数字系统码字(前 k 位 = 数据)。"""
    pm = _PMAT.setdefault(k, _parity_matrix(k))
    return list(data) + [sum(p * d for p, d in zip(row, data)) % K for row in pm]


def rs_decode_block(received, k):
    """7 数字(可含 None=擦除) -> 纠正后的 7 数字。超预算/不一致抛 RSError。"""
    assert len(received) == K
    erased = {i for i, v in enumerate(received) if v is None}
    r = K - k
    u = len(erased)
    if u > r:
        raise RSError(f"擦除数 {u} 超出预算 {r}")
    t = (r - u) // 2
    nN, nE = k + t, t + 1
    rows = []
    for i in range(K):
        if i in erased:
            continue
        rows.append([pow(i, j, K) for j in range(nN)]
                    + [(-received[i] * pow(i, j, K)) % K for j in range(nE)])
    for v in _nullspace(rows, nN + nE):
        N, E = v[:nN], v[nN:]
        if not any(E):
            continue
        err = {i for i in range(K) if i not in erased and _peval(E, i) == 0}
        q, rem = _pdivmod(N, E)
        if any(rem):
            continue
        cw = [_peval(q, x) for x in range(K)]
        if all(cw[i] == received[i] for i in range(K)
               if i not in erased and i not in err):
            return cw
    raise RSError("RS 解码失败")


# --- 编码 --------------------------------------------------------------------

def encode_framed(digits, k):
    """GF(7) 数字流 -> 分帧音符流: 每帧 7 音符(恒等映射, 重复垫 C5) + 停顿。"""
    pad = (-len(digits)) % k
    payload = list(digits) + [0] * pad
    notes, prev = [], None
    for f in range(0, len(payload), k):
        for d in rs_encode_block(payload[f:f + k], k):
            if d == prev:
                notes.append(DELIM)          # 转义: 隔开重复数字(声学防粘)
            notes.append(DATA7[d])
            prev = d
        notes.append(PAUSE)                  # 帧界: 演奏者停顿
    return notes


def encode_text(text, level="medium"):
    if level == "no":
        return list(_no_encode(text))
    return encode_framed(_huff_digits(text), LEVEL_K[level])


# --- 解码: 帧级 beam(帧间零耦合) ---------------------------------------------

def _insert_erasures(short, holes):
    """把 u 个擦除位插到 short(长度 7-u)的指定槽位, 还原成 7 位。"""
    out, p = [], 0
    for i in range(K):
        if p < len(holes) and holes[p] == i:
            out.append(None)
            p += 1
        else:
            out.append(short[i - p])
    return tuple(out)


def _close_frames(cur, k, repairs):
    """一段数字(长度 4..9) -> [(7 位帧, 修复数), ...] 全部一致解释(可能多个)。

    计数 =7: BW t=1, 失败则 1~2 擦除位穷举; 计数 <7: 吞音, 插入擦除位
    假设(位置穷举); 计数 8/9: 多吹, 丢弃假设。注意 u = 预算(t=0)时错误
    位置的插值也可能一致 -> 多候选, 由消息级(EOF + 似然 + 人眼)终审。"""
    n = len(cur)
    out = []
    if n == 7:
        try:
            return [(rs_decode_block(cur, k), repairs)]
        except RSError:
            pass                              # 弹错 1 音 -> 擦除假设穷举
        for u in (1, 2):
            if u > K - k:
                break
            for holes in combinations(range(7), u):
                mask = tuple(None if i in holes else cur[i] for i in range(7))
                try:
                    out.append((rs_decode_block(mask, k), repairs + u))
                except RSError:
                    continue
        return out
    if 4 <= n < 7:                            # 吞音: 插入擦除位假设
        u = 7 - n
        if u > K - k:
            return out
        for holes in combinations(range(7), u):
            try:
                out.append((rs_decode_block(_insert_erasures(cur, holes), k),
                            repairs + u))
            except RSError:
                continue
        return out
    if n in (8, 9):                           # 多吹 1..2 音: 丢弃假设
        for drops in combinations(range(n), n - 7):
            keep = tuple(v for i, v in enumerate(cur) if i not in drops)
            try:
                out.append((rs_decode_block(keep, k), repairs + (n - 7)))
            except RSError:
                continue
    return out


def _close_flex(cur, k, repairs):
    """按段长分发: 7 -> 单帧; 4..9 -> 单帧修复; 10..16 -> 两帧切分(停顿被吞)。"""
    n = len(cur)
    if n < 4:
        raise RSError(f"段长 {n} 过短")
    if n <= 9:
        return _close_frames(cur, k, repairs)
    out = []
    for s in range(4, n - 2):
        try:
            left = _close_frames(cur[:s], k, repairs)
        except RSError:
            continue
        try:
            right = _close_frames(cur[s:], k, 0)
        except RSError:
            continue
        for lf, lr in left:
            for rf, rr in right:
                out.append((lf + rf, max(lr, rr) + repairs))
    if not out:
        raise RSError(f"合并块切分失败(长度 {n})")
    return out


def _framed_digits(notes, k, max_variants=96):
    """停顿分帧解析 -> [(载荷数字流, 修复数), ...] 按修复数排序去重。

    帧级 beam(帧间零耦合): 状态 = (已关帧序列, 当前段数字, 修复数)。
    每段停顿间数字独立修复, 计数 ≠ 7 时枚举有界假设:
        6/5 音: 吞音 -> 插入擦除位(位置穷举)
        8/9 音: 多吹 -> 丢弃假设
    跨段组合有界截断; 多解释由消息级(EOF + 似然 + 人眼)终审。"""
    beams = [((), 0)]                        # (已关帧序列, 修复数)
    buf = []

    def flush():
        nonlocal beams, buf
        if not buf:
            return
        cands = _close_flex(tuple(buf), k, 0)
        new = []
        for frames, rep in beams:
            for fr, r in cands:
                new.append((frames + (tuple(fr),), rep + r))
        seen, uniq = set(), []
        for frames, rep in sorted(new, key=lambda t: (t[1], t[0])):
            if frames not in seen:
                seen.add(frames)
                uniq.append((frames, rep))
        beams = uniq[:BEAM_WIDTH]
        buf = []

    for note in notes:
        if note == PAUSE:
            flush()
        elif note == DELIM:
            continue                         # 转义: 防粘垫, 不携带数字
        else:
            buf.append(_IDX7[note])
    flush()
    out, seen = [], set()
    for frames, rep in sorted(beams, key=lambda t: (t[1], t[0])):
        digits = tuple(d for fr in frames for d in fr[:k])
        if digits not in seen:
            seen.add(digits)
            out.append((list(digits), rep))
    return out


def _huff_partial(digits):
    """哈夫曼解码到 EOF; 无 EOF 则返回已解前缀 + incomplete。"""
    out, acc, length = [], 0, 0
    it = iter(digits)
    for d in it:
        acc, length = acc * 7 + d, length + 1
        ch = _HUFF_LOOKUP.get((length, acc))
        if ch is None:
            if length >= _HUFF_MAXLEN:
                raise ValueError("码流损坏")
            continue
        if ch == EOF:
            return "".join(out), True
        if ch == ESC:
            out.append(_huff_escaped(it))
        else:
            out.append(ch)
        acc = length = 0
    return "".join(out), False        # 未见到 EOF: 疑似缺尾


def _score(text):
    if not text:
        return 0.0
    return sum(len(CODE_OF.get(c, (0,) * _HUFF_MAXLEN)) for c in text) / len(text)


@dataclass
class Candidate:
    level: str
    text: str
    complete: bool
    repairs: int
    score: float
    error: str = ""


def decode_notes(notes, levels=("no", "medium", "high")):
    """三路并行解码, 返回按 (完整, 似然) 排序的候选列表。"""
    cands: list[Candidate] = []
    for level in levels:
        repairs = 0
        try:
            if level == "no":
                variants = [(_no_diff(notes), 0)]
            else:
                variants = _framed_digits(notes, LEVEL_K[level])
            for digits, vr in variants:
                text, complete = _huff_partial(digits)
                cands.append(Candidate(level, text, complete, vr,
                                       _score(text)))
        except (RSError, ValueError) as e:
            cands.append(Candidate(level, "", False, 0, 9.9, str(e)))
    cands.sort(key=lambda c: (c.error != "", not (c.complete and c.text != ""),
                              c.score, c.repairs))
    return cands


# --- 命令行 -------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="文本 -> 音符(GF(7) RS 纠错 + 停顿分帧, 三档纠错)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("命令行:")[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="文本 -> 音名序列")
    pe.add_argument("text", nargs="?")
    pe.add_argument("-i", "--input")
    pe.add_argument("--level", choices=LEVEL_K, default="medium")
    pe.add_argument("--midi", action="store_true")

    pd = sub.add_parser("decode", help="音名序列 -> 文本(三路解码)")
    pd.add_argument("notes", nargs="?")
    pd.add_argument("-i", "--input")

    ps = sub.add_parser("simulate", help="信道损伤蒙特卡洛")
    ps.add_argument("--trials", type=int, default=50)
    ps.add_argument("--sub", type=float, default=0.02, help="弹错率")
    ps.add_argument("--del", dest="p_del", type=float, default=0.01, help="吞音率")
    ps.add_argument("--ins", type=float, default=0.005, help="多音率")
    ps.add_argument("--tail", type=float, default=0.05, help="尾部截断概率")

    sub.add_parser("selftest", help="自检")
    args = p.parse_args(argv)

    if args.cmd == "encode":
        notes = encode_text(_read_text(args), args.level)
        if args.midi:
            print(" ".join(str(MIDI8[NOTES8.index(n)]) for n in notes))
        else:
            print(" ".join(notes))
    elif args.cmd == "decode":
        cands = decode_notes(_read_notes(args))
        for rank, c in enumerate(cands[:5], 1):
            body = repr(c.text) if not c.error else f"失败({c.error})"
            tail = "" if c.complete or c.error else " (疑似缺尾)"
            star = "  <-- 首选" if rank == 1 and not c.error else ""
            print(f"{rank}. [{c.level:<6}] {body}{tail} "
                  f"(似然 {c.score:.2f}, 修复 {c.repairs}){star}")
        if len(cands) > 5:
            print(f"(其余 {len(cands) - 5} 个候选略)")
    elif args.cmd == "simulate":
        _simulate(args)
    elif args.cmd == "selftest":
        return _selftest()
    return 0


_SAMPLES = [
    "在吗？",
    "ok",
    "今晚八点老地方见，别迟到！",
]


def _channel(notes, rng, p_sub, p_del, p_ins):
    out = []
    for n in notes:
        if rng.random() < p_del:
            continue
        if rng.random() < p_ins:
            out.append(rng.choice(NOTES8))
        out.append(rng.choice([x for x in NOTES8 if x != n])
                   if rng.random() < p_sub else n)
    return [n for i, n in enumerate(out) if i == 0 or n != out[i - 1]]  # 同音合并


def _simulate(args):
    rng = random.Random(42)
    print(f"参数: sub={args.sub} del={args.p_del} ins={args.ins} "
          f"tail={args.tail}, trials={args.trials} x {len(_SAMPLES)} 消息")
    print(f"{'level':<8}{'严格成功':>10}{'候选命中':>10}")
    for level in LEVEL_K:
        ok_strict = ok_any = 0
        for _ in range(args.trials):
            for text in _SAMPLES:
                notes = encode_text(text, level)
                damaged = _channel(notes, rng, args.sub, args.p_del, args.ins)
                if rng.random() < args.tail and damaged:
                    del damaged[-rng.randint(1, 2):]
                cands = decode_notes(damaged)
                ok_strict += cands[0].text == text and cands[0].complete
                ok_any += any(c.text == text and c.complete for c in cands)
        n = args.trials * len(_SAMPLES)
        print(f"{level:<8}{ok_strict / n:>9.0%}{ok_any / n:>10.0%}")


def _selftest() -> int:
    rng = random.Random(20260917)
    texts = ["", "a", "ok", "在吗？", "你好，世界！",
             "今晚八点老地方见，别迟到！",
             "emoji 🎙️ 与生僻字 龘; ascii ~!@#",
             "".join(chr(c) for c in range(0x20, 0x7F))]

    # 1) RS 码块: 单错穷举 + 擦除穷举
    for k in (4, 5):
        for _ in range(20):
            data = [rng.randint(0, 6) for _ in range(k)]
            cw = rs_encode_block(data, k)
            assert rs_decode_block(cw, k) == cw
            for i in range(7):
                for v in range(7):
                    if v != cw[i]:
                        bad = cw[:i] + [v] + cw[i + 1:]
                        assert rs_decode_block(bad, k) == cw
            for u in range(1, K - k + 1):
                for holes in combinations(range(7), u):
                    mask = [None if i in holes else cw[i] for i in range(7)]
                    assert rs_decode_block(mask, k) == cw

    # 2) 干净往返(全档): 正确文本+级别必须出现在候选中(排序仅供人眼参考)
    for text in texts:
        for level in LEVEL_K:
            cands = decode_notes(encode_text(text, level))
            hit = any(c.text == text and c.complete and c.level == level
                      for c in cands)
            assert hit, (text, level, cands)

    # 3) 单事件损伤穷举(medium/high, 逐音符位置): 正确文本必须在候选中
    # 已知局限(下一步): 停顿标记 R 本身被音符替换 -> 帧界消失 + 边界插入
    # 双重损伤, 当前修复机制覆盖不到 (sub@8 等用例失败)。需要合并块切分
    # 支持 "7+跳过伪音+7" 的三段解释, 或伪解释的下游似然排序。
    base = "今晚八点老地方见，别迟到！"
    for level in ("medium", "high"):
        notes = encode_text(base, level)
        for i in range(len(notes)):
            sub = list(notes)
            sub[i] = rng.choice([x for x in NOTES8 if x != notes[i]])
            got = decode_notes(_merge(sub))
            assert any(c.text == base and c.complete for c in got), \
                ("sub", level, i, got)
            got = decode_notes(_merge(notes[:i] + notes[i + 1:]))
            assert any(c.text == base and c.complete for c in got), \
                ("del", level, i, got)
            ins = notes[:i] + [rng.choice(NOTES8)] + notes[i:]
            got = decode_notes(_merge(ins))
            assert any(c.text == base and c.complete for c in got), \
                ("ins", level, i, got)

    # 4) 蒙特卡洛: 损伤率扫描
    print("Monte Carlo(候选命中, 每格 60 trial x 3 消息):")
    print(f"{'level':<8}{'1%+0.5%':>10}{'2%+1%':>10}{'3%+2%':>10}")
    for level in LEVEL_K:
        row = []
        for p_sub, p_del in ((0.01, 0.005), (0.02, 0.01), (0.03, 0.02)):
            ok = tot = 0
            for _ in range(60):
                for text in _SAMPLES:
                    notes = encode_text(text, level)
                    damaged = _channel(notes, rng, p_sub, p_del, 0.005)
                    cands = decode_notes(damaged)
                    ok += any(c.text == text and c.complete for c in cands)
                    tot += 1
            row.append(f"{ok / tot:.0%}")
        print(f"{level:<8}" + "".join(f"{r:>10}" for r in row))
    print("selftest ok")
    return 0


def _merge(notes):
    return [n for i, n in enumerate(notes) if i == 0 or n != notes[i - 1]]


if __name__ == "__main__":
    sys.exit(main())
