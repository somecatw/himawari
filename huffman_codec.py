#!/usr/bin/env python3
"""文本 -> 音符序列(七叉哈夫曼 + 旋转信道层)。

相对 notes_codec.py(baseline: 每字节 3 个基-7 数字 + C5 转义)的两处升级:

1. 信源层: 按字频(25 亿字语料 + 少量合成权重)建七叉哈夫曼, 码字就是
   0..6 数字序列, 高频字符 2~4 个数字(见 huffman_table.py, 收发共用);
   表外字符用 ESCAPE 码字 + UTF-8 字节逐字节(3 数字/字节)转义;
   流式协议默认不发 EOF(省符号): 消息结束由演奏停顿定界, 需要码流
   自定界的场合显式 text_to_digits(text, eof=True)。

2. 信道层: 旋转映射取代 C5 转义。把 8 个音(C4..C5)看成一个环,
   数字 d 编为「上一个音符 + 1 + d (mod 8)」:
       state 初始为 C4;  d=0 -> 上行一度, d=6 -> 下行一度(模 8)。
   相邻音符天然不同(8 个音全部成为数据载体), 不再浪费 1/7 容量插转义
   —— 平均码长恰为转义方案的 7/8。解码端作差即可: d = (note - prev - 1) mod 8。

!! 接收端注意: frontend/symbols.py 目前仍是 C5=ESC 的方案, 需同步改为
   上面的作差解码, 两端才能互通(码表部分本来就共用)。

命令行:
    (需能 import frontend, 即在仓库根目录下运行)
    python huffman_codec.py encode "你好, world"   # 文本 -> 音名序列
    python huffman_codec.py encode -i msg.txt
    python huffman_codec.py encode "hi" --midi     # 输出 MIDI 编号
    python huffman_codec.py decode "D4 F4 ..."     # 音名序列 -> 文本
    python huffman_codec.py stats                  # 与 baseline 的码长对比
    python huffman_codec.py selftest               # 往返与信道约束自检
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from fractions import Fraction

from frontend.symbols import midi_name
from frontend.__main__ import parse_note

from huffman_table import CODEBOOK
from notes_codec import _read_text, _read_notes, text_to_notes as _baseline_notes

K = 7
ESC, EOF = "\ue000", "\ue001"

# --- 旋转信道层: 8 音环 ------------------------------------------------------

NOTES8 = tuple(midi_name(m) for m in (60, 62, 64, 65, 67, 69, 71, 72))  # C4..C5
MIDI8 = (60, 62, 64, 65, 67, 69, 71, 72)
START = 0                     # 发送端 state 初值: C4(收发约定)
_NOTE_IDX = {name: i for i, name in enumerate(NOTES8)}


def digits_to_notes(digits, start: int = START) -> list[str]:
    """数字流 -> 音符流(旋转映射, 保证相邻不同音, 8 音全用)。"""
    notes, state = [], start
    for d in digits:
        state = (state + 1 + d) % len(NOTES8)
        notes.append(NOTES8[state])
    return notes


def notes_to_digits(notes, start: int = START) -> list[int]:
    """音符流 -> 数字流(作差)。表外音符报错。"""
    idx = []
    for n in notes:
        i = _NOTE_IDX.get(str(n).strip().upper())
        if i is None:
            raise ValueError(f"不认识的音名: {n!r} (信道只允许 {' '.join(NOTES8)})")
        idx.append(i)
    prev, out = start, []
    for i in idx:
        out.append((i - prev - 1) % len(NOTES8))
        prev = i
    return out


# --- 哈夫曼码表: 规范码字重建 ------------------------------------------------

def _build_codes() -> tuple[dict[str, tuple[int, ...]], dict[tuple[int, int], str]]:
    book = dict(CODEBOOK)
    code_of: dict[str, tuple[int, ...]] = {}
    lookup: dict[tuple[int, int], str] = {}
    prev_end = 0
    for L in range(1, max(book) + 1):
        start = prev_end * K
        syms = book.get(L, "")
        for j, ch in enumerate(syms):
            code = start + j
            digits = tuple((code // K ** (L - 1 - k)) % K for k in range(L))
            code_of[ch] = digits
            lookup[(L, code)] = ch
        prev_end = start + len(syms)
    return code_of, lookup


CODE_OF, _LOOKUP = _build_codes()
_MAX_CODE_LEN = max(L for L, _ in CODEBOOK)

# 完整性: 不含哑叶的码表应满足 Kraft 和 <= 1(前缀无歧义)
assert sum(Fraction(1, K ** len(c)) for c in CODE_OF.values()) <= 1


# --- 编码 / 解码 --------------------------------------------------------------

def _byte_digits(b: int) -> tuple[int, ...]:
    return (b // (K * K), (b // K) % K, b % K)


def text_to_digits(text: str, eof: bool = False) -> list[int]:
    digits: list[int] = []
    for ch in text:
        code = CODE_OF.get(ch)
        if code is not None:
            digits += code
        else:                       # 表外字符: ESCAPE + UTF-8 字节逐字节
            digits += CODE_OF[ESC]
            for b in ch.encode("utf-8"):
                digits += _byte_digits(b)
    if eof:
        digits += CODE_OF[EOF]
    return digits


def digits_to_text(digits, *, strict: bool = True) -> tuple[str, bool]:
    """哈夫曼批式解码: 数字流 -> (文本, 是否完整收尾)。

    text_to_digits 的逆运算。完整收尾 = 见到 EOF 码字, 或码字边界收尾
    (流式协议无 EOF 的正常结束); 末尾半码 -> (前缀, False)。
    strict=True(默认): 末尾半码/损坏/EOF 后多余码元一律抛 ValueError
    —— 供有完整性上下文的调用方(RS 帧裁剪)使用。strict=False(宽容):
    半码截断返回已解前缀(complete=False), 码流损坏跳过 1 个数字再同步
    —— 供多候选排序场景使用(候选可能本就是错的, 不值得抛异常)。
    流式解码用 Receiver 的 _HuffStream(增量 + 永不抛)。"""
    out: list[str] = []
    acc = length = 0
    it = iter(digits)
    while True:
        try:
            d = next(it)
        except StopIteration:
            if strict and length:
                raise ValueError("码流截断: 末尾半码") from None
            return "".join(out), not length
        acc, length = acc * K + d, length + 1
        ch = _LOOKUP.get((length, acc))
        if ch is None:
            if length >= _MAX_CODE_LEN:
                if strict:
                    raise ValueError(f"码流损坏: 前缀 {acc} 无法匹配任何码字")
                return "".join(out), False     # 宽容: 损坏点之后的尾丢给上层
            continue
        if ch == EOF:
            try:
                next(it)
                if strict:
                    raise ValueError("EOF 之后仍有多余码元")
            except StopIteration:
                pass
            return "".join(out), True
        if ch == ESC:
            out.append(_decode_escaped(it))
        else:
            out.append(ch)
        acc = length = 0


def _decode_escaped(it) -> str:
    first = _read_byte(it)
    n = 1 if first < 0x80 else 2 if first < 0xE0 else 3 if first < 0xF0 else 4
    data = bytes([first]) + bytes(_read_byte(it) for _ in range(n - 1))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"转义字节不是合法 UTF-8: {data.hex()}") from e


def _read_byte(it) -> int:
    b = 0
    for _ in range(3):
        b = b * K + next(it, -1)
        if b < 0:
            raise ValueError("码流截断: 转义字节不完整")
    if b > 255:
        raise ValueError(f"转义字节越界: {b}")
    return b


def text_to_notes(text: str) -> list[str]:
    """文本 -> 音名序列(哈夫曼 + 旋转信道, 无 C5 转义)。"""
    return digits_to_notes(text_to_digits(text))


def text_to_midi(text: str) -> list[int]:
    return [MIDI8[NOTES8.index(n)] for n in text_to_notes(text)]


def notes_to_text(notes) -> str:
    """音名序列 -> 文本(流式协议无 EOF, 边界收尾即正常结束)。"""
    return digits_to_text(notes_to_digits(notes))[0]


# --- 命令行 -------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="文本 -> 音名/MIDI 序列(七叉哈夫曼 + 旋转信道)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("命令行:")[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="文本 -> 音名序列")
    pe.add_argument("text", nargs="?", help="要编码的文本; 省略时读 -i 文件或 stdin")
    pe.add_argument("-i", "--input", help="从文件读取文本(UTF-8)")
    pe.add_argument("--midi", action="store_true", help="输出 MIDI 编号而非音名")

    pd = sub.add_parser("decode", help="音名序列 -> 文本")
    pd.add_argument("notes", nargs="?", help="空格/逗号分隔的音名; 省略时读 -i 或 stdin")
    pd.add_argument("-i", "--input", help="从文件读取音名序列")

    sub.add_parser("stats", help="与 baseline(每字节 3 数字 + C5 转义)的码长对比")
    sub.add_parser("selftest", help="往返与信道约束自检")
    args = p.parse_args(argv)

    if args.cmd == "encode":
        if args.midi:
            print(" ".join(str(m) for m in text_to_midi(_read_text(args))))
        else:
            print(" ".join(text_to_notes(_read_text(args))))
    elif args.cmd == "decode":
        try:
            print(notes_to_text(_read_notes(args)))
        except (ValueError, UnicodeDecodeError) as e:
            print(f"decode 失败: {e}", file=sys.stderr)
            return 1
    elif args.cmd == "stats":
        _stats()
    elif args.cmd == "selftest":
        return _selftest()
    return 0


_SAMPLES = [
    "在吗？",
    "你好，世界！",
    "ok",
    "Hello, world!",
    "今晚八点老地方见，别迟到！",
    "这段消息用来测试哈夫曼编码在较长中文文本下的平均码长，包括标点、数字 123 与 English。",
    "emoji 🎙️ 与生僻字 龘 混合。",
]


def _stats() -> None:
    print(f"{'样例':<18} {'字符':>4} {'字节':>4} | {'baseline数字':>9} {'baseline音符':>9} | "
          f"{'哈夫曼数字':>8} {'哈夫曼音符':>8} | {'音符/字符':>8}")
    for text in _SAMPLES:
        base = _baseline_notes(text)
        huff = text_to_notes(text)
        n = max(len(text), 1)
        print(f"{text[:16]:<18} {len(text):>4} {len(text.encode('utf-8')):>4} | "
              f"{3 * len(text.encode('utf-8')):>11} {len(base):>12} | "
              f"{len(text_to_digits(text)):>10} {len(huff):>12} | "
              f"{len(huff) / n:>8.2f}")


def _selftest() -> int:
    cases = [
        "", "a", "ok", "aa", "aaaa",
        "Hello, world!", "你好，世界！", "在吗？",
        "emoji 🎙️🎵 and ça va? 100%",
        "生僻字: 龘 龗 䲜; 全角ＡＢＣ１２３",
        "".join(chr(c) for c in range(0x20, 0x7F)),
    ]
    rng = random.Random(20260917)
    for _ in range(200):
        cases.append("".join(rng.choice("abcdefghijklmn abcXYZ.,!?你我他这哪"
                                        "，。！？") for _ in range(rng.randint(0, 60))))
    for _ in range(200):
        cases.append("".join(chr(rng.randint(0, 0x4FFF))
                             for _ in range(rng.randint(0, 40))))

    for i, text in enumerate(cases):
        digits = text_to_digits(text)
        notes = digits_to_notes(digits)
        for a, b in zip(notes, notes[1:]):     # 信道约束: 无相邻同音
            assert a != b, f"case {i}: 相邻同音 {a}"
        assert notes_to_text(notes) == text, f"case {i} 往返不一致: {text!r}"

    # 旋转信道层单独往返: 任意数字流必须无损
    for _ in range(500):
        ds = [rng.randint(0, K - 1) for _ in range(rng.randint(0, 64))]
        assert notes_to_digits(digits_to_notes(ds)) == ds
    # 非法输入(在 notes_to_text 层面必须报错)
    for bad in (["C4"], ["C4", "C5"], ["C4", "G3"], ["C4", "C5", "X"]):
        try:
            notes_to_text(bad)
            raise SystemExit(f"非法音符序列未被拒绝: {bad}")
        except ValueError:
            pass
    try:
        digits_to_text([0])                    # 末尾半码(边界收尾是合法的)
        raise SystemExit("半码截断码流未被拒绝")
    except ValueError:
        pass

    print(f"selftest ok: {len(cases)} 个文本用例 + 500 个随机数字流, "
          f"全部通过往返与信道约束校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
