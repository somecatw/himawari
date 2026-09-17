#!/usr/bin/env python3
"""文本 <-> 音名序列 编码工具(七元符号信道的发送端前级)。

分层(与 frontend/symbols.py 的约定对齐):
    - 信道层 —— 数据音高表 / ESC 转义 / 游戏键位 —— 由 CodecSpec 定义,
      收发共用(单一事实来源), 本模块不重复维护;
    - 本模块只做 信息 <-> GF(7) 符号 的分帧: UTF-8 字节流, 每字节固定展开
      为 3 个基-7 数字(大端, 7^3 = 343 >= 256)。
      3 个音符传 8 bit, 信道利用率 8/(3*log2 7) ~= 95%; 字节边界固定,
      单个符号识别错最多损坏 1 个字节, 不扩散; 数据符号总数必为 3 的倍数,
      可用作校验。

完整编码链:
    text -> bytes -> symbols(0..6) -> CodecSpec.encode -> MIDI(相邻同音间插 ESC)
    -> 音名 / 该按的键

解码链(本模块的 notes_to_text, 供往返校验):
    音名 -> MIDI -> CodecSpec.symbol_of -> 剥 ESC(校验位置合法) -> symbols
    -> bytes -> text

命令行:
    (需能 import frontend, 即在仓库根目录下运行)
    python notes_codec.py encode "你好, world"      # 文本 -> 音名序列(含 C5)
    python notes_codec.py encode -i msg.txt
    cat msg.txt | python notes_codec.py encode
    python notes_codec.py encode "hi" --midi        # 输出 MIDI 编号
    python notes_codec.py encode "hi" --keys        # 输出该按的游戏键位
    python notes_codec.py decode "C4 D4 E4 ..."     # 音名序列 -> 文本
    python notes_codec.py selftest                  # 往返与信道约束自检
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from typing import Iterable, List, Sequence

from frontend.symbols import ESCAPE, ERASURE, CodecSpec, midi_name
from frontend.__main__ import parse_note

DEFAULT_SPEC = CodecSpec()
BASE = DEFAULT_SPEC.q          # 7
DIGITS_PER_BYTE = 3            # 7^3 = 343 >= 256


# --- 信息 -> 符号(分帧层) ---------------------------------------------------

def text_to_symbols(text: str) -> List[int]:
    """文本(UTF-8) -> GF(7) 符号序列(0..6, 不含 ESC)。"""
    symbols: List[int] = []
    for byte in text.encode("utf-8"):
        symbols += [byte // (BASE * BASE), (byte // BASE) % BASE, byte % BASE]
    return symbols


def symbols_to_text(symbols: Sequence[int]) -> str:
    """GF(7) 符号序列 -> 文本(UTF-8)。长度必须是 3 的倍数。"""
    if len(symbols) % DIGITS_PER_BYTE:
        raise ValueError(
            f"数据符号共 {len(symbols)} 个, 不是 {DIGITS_PER_BYTE} 的倍数(传输出错?)")
    out = bytearray()
    for i in range(0, len(symbols), DIGITS_PER_BYTE):
        d0, d1, d2 = symbols[i:i + DIGITS_PER_BYTE]
        if not (0 <= d0 < BASE and 0 <= d1 < BASE and 0 <= d2 < BASE):
            raise ValueError(f"符号越界: {symbols[i:i + DIGITS_PER_BYTE]}")
        out.append(d0 * BASE * BASE + d1 * BASE + d2)
    return bytes(out).decode("utf-8")


# --- 符号 <-> 信道层(音名 / MIDI / 键位) ------------------------------------

def symbols_to_notes(symbols: Sequence[int], spec: CodecSpec = DEFAULT_SPEC) -> List[str]:
    """符号 -> 音名序列(相邻同音之间插 ESC, 由 CodecSpec 保证)。"""
    return [midi_name(m) for m in spec.encode(symbols)]


def text_to_notes(text: str, spec: CodecSpec = DEFAULT_SPEC) -> List[str]:
    """文本 -> 音名序列(含转义 C5)。"""
    return symbols_to_notes(text_to_symbols(text), spec)


def text_to_midi(text: str, spec: CodecSpec = DEFAULT_SPEC) -> List[int]:
    """文本 -> MIDI 编号序列(含转义)。"""
    return spec.encode(text_to_symbols(text))


def text_to_keys(text: str, spec: CodecSpec = DEFAULT_SPEC) -> List[str]:
    """文本 -> 该按的游戏键位序列(显示给演奏者用)。"""
    return [spec.key_of(spec.symbol_of(m)) for m in text_to_midi(text, spec)]


def notes_to_symbols(notes: Iterable[str], spec: CodecSpec = DEFAULT_SPEC) -> List[int]:
    """音名序列 -> GF(7) 符号。剥掉 ESC 并做位置合法性校验。"""
    midi = [parse_note(n) for n in notes]
    symbols: List[int] = []
    for i, m in enumerate(midi):
        sym = spec.symbol_of(m)
        if sym == ERASURE:
            raise ValueError(
                f"第 {i + 1} 个音符 {midi_name(m)!r} 不在符号表 "
                f"({', '.join(spec.name_of(s) for s in range(spec.q))} / ESC) 内")
        if sym == ESCAPE:
            # 合法编码中, ESC 两侧必须是同一个数据音符
            if not (0 < i < len(midi) - 1 and midi[i - 1] == midi[i + 1] != m):
                raise ValueError(f"第 {i + 1} 个音符: 转义符 {midi_name(m)} 位置非法")
            continue
        symbols.append(sym)
    return symbols


def notes_to_text(notes: Iterable[str], spec: CodecSpec = DEFAULT_SPEC) -> str:
    """音名序列 -> 文本(UTF-8)。"""
    return symbols_to_text(notes_to_symbols(notes, spec))


def check_channel_constraints(midi_seq: Sequence[int], spec: CodecSpec = DEFAULT_SPEC) -> None:
    """校验 MIDI 序列满足信道约束: 全部在符号表内, 无相邻同音。失败抛 AssertionError。"""
    table = set(spec.pitches) | {spec.escape}
    for i, m in enumerate(midi_seq):
        assert m in table, f"第 {i + 1} 个音 {midi_name(m)} 超出符号表"
    for i, (a, b) in enumerate(zip(midi_seq, midi_seq[1:])):
        assert a != b, f"第 {i + 1}/{i + 2} 个音出现相邻同音: {midi_name(a)} {midi_name(b)}"


# --- 命令行 -----------------------------------------------------------------

def _read_text(args) -> str:
    if args.text is not None:
        return args.text
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _read_notes(args) -> List[str]:
    raw = args.notes
    if raw is None:
        raw = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    return [t for t in re.split(r"[\s,]+", raw) if t]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="文本 -> 音名/MIDI/键位 序列(七元符号信道, ESC 转义)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("命令行:")[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="文本 -> 音名序列")
    pe.add_argument("text", nargs="?", help="要编码的文本; 省略时读 -i 文件或 stdin")
    pe.add_argument("-i", "--input", help="从文件读取文本(UTF-8)")
    fmt = pe.add_mutually_exclusive_group()
    fmt.add_argument("--midi", action="store_true", help="输出 MIDI 编号而非音名")
    fmt.add_argument("--keys", action="store_true", help="输出该按的游戏键位")

    pd = sub.add_parser("decode", help="音名序列 -> 文本")
    pd.add_argument("notes", nargs="?", help="空格/逗号分隔的音名; 省略时读 -i 或 stdin")
    pd.add_argument("-i", "--input", help="从文件读取音名序列")

    sub.add_parser("selftest", help="往返编码与信道约束自检")
    args = p.parse_args(argv)

    if args.cmd == "encode":
        if args.midi:
            print(" ".join(str(m) for m in text_to_midi(_read_text(args))))
        elif args.keys:
            print(" ".join(text_to_keys(_read_text(args))))
        else:
            print(" ".join(text_to_notes(_read_text(args))))
    elif args.cmd == "decode":
        try:
            print(notes_to_text(_read_notes(args)))
        except (ValueError, UnicodeDecodeError) as e:
            print(f"decode 失败: {e}", file=sys.stderr)
            return 1
    elif args.cmd == "selftest":
        return _selftest()
    return 0


def _selftest() -> int:
    cases = [
        "",
        "a", "aa", "aaa", "aaaa", "abab",
        "Hello, world!",
        "你好，世界！口琴信道测试。",
        "emoji 🎙️🎵 and ça va? 100%",
        "".join(chr(c) for c in range(0x20, 0x7F)),
    ]
    rng = random.Random(20260917)
    for _ in range(200):
        cases.append("".join(rng.choice("ABCDEFGHIJKLM abcXYZ .,!?")
                             for _ in range(rng.randint(0, 60))))
    for _ in range(200):  # 随机 Unicode, 覆盖多字节字符
        cases.append("".join(chr(rng.randint(0x0, 0x4FFF))
                             for _ in range(rng.randint(0, 40))))

    n_notes_total = 0
    for i, text in enumerate(cases):
        notes = text_to_notes(text)
        check_channel_constraints([parse_note(n) for n in notes])
        assert notes_to_text(notes) == text, f"case {i} 往返不一致: {text!r}"
        # 符号层往返 + 键位序列与音名序列一一对应
        syms = text_to_symbols(text)
        assert notes_to_symbols(symbols_to_notes(syms)) == syms
        assert len(text_to_keys(text)) == len(notes)
        n_notes_total += len(notes)

    # 随机符号流(全字母表覆盖)也必须能无损走完 音名 -> 符号 这条路
    for _ in range(200):
        syms = [rng.randint(0, BASE - 1) for _ in range(rng.randint(0, 63))]
        assert notes_to_symbols(symbols_to_notes(syms)) == syms

    print(f"selftest ok: {len(cases)} 个文本用例 + 200 个随机符号流, "
          f"共 {n_notes_total} 个音符, 全部通过往返与信道约束校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
