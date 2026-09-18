#!/usr/bin/env python3
"""口琴信道接收端解码 —— 第一阶段: 无纠错贪心解码。

接收管线: 音频 -> 音高检测(frontend/engine.py 的 PitchFrontend 输出
NoteEvent 流) -> [本模块] -> 文本。

可靠模型(用户公理): 接收端没有可靠时钟, 只有「音高变化」是可靠事件:
    - 音符可能被随机吞掉(删除);
    - YIN 可能把长音拆成同音短音(前端按同音 run 合并, 到达本模块的
      相邻同音只可能是残余损伤 -> 跳过并计数);
    - 一个音可能被检成两个不同音高(替换 + 插入);
    - 可能混入杂音(插入)。

无纠错模式的设计取舍:
    - 贪心解码, 永不崩溃: 损伤只表现为局部文本乱码 —— 7 叉哈夫曼前缀码
      失步后最多 _MAX_CODE_LEN 个数字必然再同步(见 _HuffStream), 乱码
      不向后扩散;
    - 旋转信道层作差还原数字: d = (idx - prev - 1) mod 8, 起始 prev = C4
      (序号 0), 与 huffman_codec 的发送端编码互逆。删除/替换各只损坏
      1~2 个数字, 环外杂音可在作差前直接滤除;
    - 流式协议默认不发 EOF(省符号): complete 标记 = 码字边界收尾
      (半码残留 -> incomplete, 已解出的前缀照常返回, 不报错), 消息分界
      由调用方按演奏停顿判定; EOF 符号仍在码表, 见到即恒定完整(遗留兼
      容), 其后音符计为余音。

与其它模块的关系: 只依赖收发共用的 huffman_codec(码表 + 旋转层约定),
不依赖 rs_codec(并行开发中)也不 import frontend —— NoteEvent 等含
midi/note 属性的对象按鸭子类型直接可喂(见 Receiver.feed)。

命令行:
    python3 receiver.py decode "G4 F4 ..."    # 音名/MIDI 序列 -> 文本(贪心)
    python3 receiver.py decode -i notes.txt
    python3 receiver.py simulate              # 损伤扫描: 量化退化表格
    python3 receiver.py simulate --trials 200 --del 0.03 --ins 0.02 --sub 0.03
    python3 receiver.py selftest              # 往返 + 边界 + 损伤扫描自检
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

from huffman_codec import (CODE_OF, ESC, EOF, K, MIDI8, NOTES8, START,
                           _LOOKUP, _MAX_CODE_LEN, _NOTE_IDX, _byte_digits,
                           _read_notes, digits_to_notes, text_to_digits,
                           text_to_midi, text_to_notes)

_MIDI_IDX = {m: i for i, m in enumerate(MIDI8)}     # MIDI 号 -> 8 音环序号


# --- 输入归一化 ---------------------------------------------------------------

def note_index_of(note) -> Optional[int]:
    """任意形式的音符输入 -> 8 音环序号 0..7; 无法识别返回 None。

    接受: 音名字符串('G4', 大小写不敏感)、MIDI 号(int/float, 含 numpy
    数值)、纯数字字符串('62', 命令行分词的产物)、含 midi 或 note 属性的
    对象(如 frontend 的 NoteEvent / FramePitch)。表外音高(杂音)返回
    None, 由调用方跳过并计数。
    """
    if isinstance(note, str):
        idx = _NOTE_IDX.get(note.strip().upper())
        if idx is not None:
            return idx
        try:                          # 纯数字字符串按 MIDI 号解析
            return _MIDI_IDX.get(int(note.strip()))
        except ValueError:
            return None
    if hasattr(note, "midi"):                 # NoteEvent / FramePitch 等
        if note.midi is None:
            return None
        note = note.midi                      # midi 属性优先, 继续按数值解析
    elif hasattr(note, "note"):
        note = note.note
        if isinstance(note, str):
            return _NOTE_IDX.get(note.strip().upper())
    try:
        return _MIDI_IDX.get(int(round(note)))
    except (TypeError, ValueError, OverflowError):
        return None


# --- 解码结果 -----------------------------------------------------------------

@dataclass
class DecodeResult:
    """一次解码的完整结果(文本 + 完成状态 + 信道损伤计数)。"""

    text: str = ""          # 解出的文本(半码截断时为已解出的前缀)
    complete: bool = False  # 消息完整结束: 码字边界收尾, 或见过遗留 EOF
    n_notes: int = 0        # 接受的音符数(= 还原的数字数)
    skipped: int = 0        # 跳过的杂音(不认识的音名/表外 MIDI)
    repeats: int = 0        # 跳过的相邻同音(旋转编码下必为损伤)
    trailing: int = 0       # 遗留 EOF 之后的余音
    bad_digits: int = 0     # 哈夫曼失步时丢弃的数字
    error: str = ""         # 非致命解码异常(如转义字节非法), 空串 = 无

    @property
    def ok(self) -> bool:
        """干净成功: 边界收尾且无解码异常。"""
        return self.complete and not self.error

    def summary(self) -> str:
        """一行状态摘要(诊断用)。"""
        state = "complete(边界收尾)" if self.complete else "incomplete(半码截断)"
        extra = f", error={self.error!r}" if self.error else ""
        return (f"{state}, 音符 {self.n_notes}, 杂音跳过 {self.skipped}, "
                f"重复音跳过 {self.repeats}, 余音 {self.trailing}, "
                f"失步丢弃数字 {self.bad_digits}{extra}")


# --- 流式哈夫曼前缀解码 -------------------------------------------------------

_NEED = object()    # _match 的哨兵: 窗口耗尽仍无定论, 等待更多数字


class _HuffStream:
    """流式 7 叉哈夫曼前缀解码(贪心 + 自动再同步, 永不抛异常)。

    - 逐数字喂入; 从当前同步点匹配最短码字, 命中即输出字符
      (EOF(遗留) -> 恒定完整; ESC -> 进入转义态);
    - 转义态: 3 数字/字节, 首字节定长(1~4 字节), 整段按 UTF-8 解码;
      字节越界/序列非法时输出 U+FFFD 并记录 error, 不中断后续解码;
    - 从当前同步点起连续 _MAX_CODE_LEN 个数字匹配不上任何码字(损伤所致):
      丢弃队首 1 个数字再试 —— 前缀码失步后快速再同步, 乱码被限制在局部。
      (rs_codec 的无纠错路径遇此直接报错, 本模块按用户公理永不崩溃。)
    """

    def __init__(self) -> None:
        self.chars: list[str] = []
        self.complete = False            # 边界收尾(或遗留 EOF, 恒定完整)
        self.error = ""
        self._eof = False                # 见过遗留 EOF 码字(锁存)
        self._fed = False                # 已喂过数字(空流不算边界收尾)
        self.bad_digits = 0              # 再同步时丢弃的数字
        self._pend: deque[int] = deque() # 未消费数字窗口(<= 码长上限)
        self._esc: Optional[list[int]] = None  # None=普通态; 否则已收转义字节
        self._esc_total = 0              # 转义总字节数(首字节到手后确定)
        self._b = 0                      # 当前字节的三数字累积
        self._nd = 0                     # 当前字节已收数字数

    def feed(self, d: int) -> None:
        """喂入一个 GF(7) 数字(0..6)。"""
        self._fed = True
        self._pend.append(d)
        self._run()
        if not self._eof:                # 流式完整判据: 无半码残留/转义中断
            self.complete = (not self._pend and self._esc is None)

    def _run(self) -> None:
        while self._pend and not self.complete:
            if self._esc is not None:    # 转义态: 数字按字节组装
                self._esc_digit(self._pend.popleft())
                continue
            m = self._match()
            if m is _NEED:               # 可能是码字前缀, 等后续数字
                break
            if m is None:                # 队首不可能是任何码字起点: 丢弃再同步
                self.bad_digits += 1
                self._pend.popleft()
                continue
            ch, used = m
            for _ in range(used):
                self._pend.popleft()
            if ch == EOF:
                self._eof = True
                self.complete = True
            elif ch == ESC:
                self._esc, self._b, self._nd = [], 0, 0
            else:
                self.chars.append(ch)

    def _match(self):
        """从队首匹配一个码字, 返回 (字符, 消耗数字数)。

        窗口内凑满码长上限仍无匹配 -> None(队首数字必不是码字起点);
        窗口先耗尽 -> _NEED(前缀仍可延伸, 等待更多数字)。
        """
        acc = length = 0
        for i, d in enumerate(self._pend):
            acc = acc * K + d
            length += 1
            ch = _LOOKUP.get((length, acc))
            if ch is not None:
                return ch, i + 1
            if length >= _MAX_CODE_LEN:
                return None
        return _NEED

    def _esc_digit(self, d: int) -> None:
        self._b = self._b * K + d
        self._nd += 1
        if self._nd < 3:
            return
        byte, self._b, self._nd = self._b, 0, 0
        if byte > 255:
            self._fail_escape(f"转义字节越界: {byte}")
            return
        self._esc.append(byte)
        if self._esc_total == 0:         # 首字节定长(与发送端 UTF-8 约定一致)
            self._esc_total = (1 if byte < 0x80 else
                               2 if byte < 0xE0 else
                               3 if byte < 0xF0 else 4)
        if len(self._esc) >= self._esc_total:
            data = bytes(self._esc)
            try:
                self.chars.append(data.decode("utf-8"))
            except UnicodeDecodeError:
                self.chars.append("\ufffd")
                self.error = f"转义字节不是合法 UTF-8: {data.hex()}"
            self._esc, self._esc_total = None, 0

    def _fail_escape(self, msg: str) -> None:
        self.chars.append("\ufffd")
        self.error = msg
        self._esc, self._esc_total = None, 0


# --- 接收器 -------------------------------------------------------------------

class Receiver:
    """逐音符喂入的接收端解码器(无纠错模式, 贪心, 永不崩溃)。

    用法:
        r = Receiver()
        for ev in note_events:      # NoteEvent / 音名 / MIDI 号都可以
            r.feed(ev)
        print(r.result.text, r.result.complete)

    也可对接 PitchFrontend 的回调: on_event=lambda e: r.feed(e)
    (FramePitch 会被当作数值输入处理, 但常规接法是只喂 NoteEvent 流)。
    """

    def __init__(self) -> None:
        self.prev = START                # 收发约定: 起始 C4(序号 0)
        self.notes: list[str] = []       # 接受的音名(诊断用)
        self.digits: list[int] = []      # 还原的数字流(可与 RS 层对接)
        self.skipped = 0                 # 不认识的输入(杂音/表外音高)
        self.repeats = 0                 # 相邻同音(旋转编码下必为损伤)
        self.trailing = 0                # 遗留 EOF 之后的余音
        self._stream = _HuffStream()

    def feed(self, note) -> "Receiver":
        """喂入一个音符, 返回 self(可链式调用)。

        接受: 音名字符串('G4', 大小写不敏感)、MIDI 号(int/float)、含 midi
        或 note 属性的对象(如 frontend 的 NoteEvent, 其 kind=='off' 事件
        自动忽略)。无法识别的输入跳过并计数, 不抛异常。
        """
        if getattr(note, "kind", None) == "off":
            return self                  # 音符结束事件不携带语义
        idx = note_index_of(note)
        if idx is None:
            self.skipped += 1            # 环外杂音: 作差前直接滤除
            return self
        if self._stream._eof:
            self.trailing += 1           # 遗留 EOF 之后的余音: 计数不进解码
            return self
        if idx == self.prev:
            # 发送端旋转编码保证相邻不同音; 相邻同音只可能是检测损伤
            # (多检/拆分残余)。丢弃它不影响差分链(prev 不变), 恰好恢复
            # 原数字流 —— 同音插入是唯一能被完美修复的损伤。
            self.repeats += 1
            return self
        d = (idx - self.prev - 1) % len(NOTES8)   # 旋转层作差, 0..6
        self.digits.append(d)
        self._stream.feed(d)
        self.prev = idx
        self.notes.append(note if isinstance(note, str)
                          else str(getattr(note, "note", note)))
        return self

    def feed_all(self, notes) -> "Receiver":
        """依次喂入一串音符。"""
        for n in notes:
            self.feed(n)
        return self

    @property
    def result(self) -> DecodeResult:
        """当前解码结果(流中随时可读, 文本为已解出的前缀)。"""
        s = self._stream
        return DecodeResult(text="".join(s.chars), complete=s.complete,
                            n_notes=len(self.notes), skipped=self.skipped,
                            repeats=self.repeats, trailing=self.trailing,
                            bad_digits=s.bad_digits, error=s.error)


# --- 模块级便捷函数 -----------------------------------------------------------

def _plausibility(text: str) -> float:
    """字频似然: 平均哈夫曼码长(越短 = 高频字占比越高 = 越像正常文本)。"""
    if not text:
        return float("inf")
    return sum(len(CODE_OF.get(c, (0,) * _MAX_CODE_LEN))
               for c in text) / len(text)


def decode_assisted(notes, max_candidates: int = 6) -> list[DecodeResult]:
    """单错误假设枚举 + 字频似然排序(短报文推荐模式)。

    贪心解码干净(边界收尾且无杂音/失步痕迹)时直接返回单候选快速路径;
    否则枚举单错误假设:
        吞音:  删除第 i 个音符              (n 个假设)
        多音:  第 i 位前插入音 v            (n × 8)
        错音:  第 i 位替换为 v              (n × 7)
    每个假设独立贪心解码(哈夫曼失步自动再同步, 损伤局部化), 按
    (完整收尾, 平均码长升序) 排序, 前 max_candidates 个供人眼终审。"""
    base = decode_notes(notes)
    if base.ok and not base.skipped and not base.bad_digits:
        base.repairs = 0
        return [base]

    seen_notes, variants = set(), []
    seq = list(notes)
    for i in range(len(seq)):
        for v in NOTES8:
            if v != seq[i]:
                variants.append(seq[:i] + [v] + seq[i + 1:])   # 错音
        variants.append(seq[:i] + seq[i + 1:])                  # 吞音
        for v in NOTES8:
            variants.append(seq[:i] + [v] + seq[i:])            # 多音
    for v in NOTES8:
        variants.append(seq + [v])                               # 尾部多音

    cands: list[DecodeResult] = []
    seen_text = set()
    for seq_v in variants[:2048]:        # 组合上限(2 错场景由纠错档覆盖)
        res = decode_notes(seq_v)
        if res.error or res.text in seen_text:
            continue
        seen_text.add(res.text)
        res.repairs = 1                  # 单错误假设
        cands.append(res)
    cands.sort(key=lambda c: (not c.complete, _plausibility(c.text)))
    return cands[:max_candidates] if len(cands) > max_candidates else cands


def decode_notes(notes) -> DecodeResult:
    """音名/MIDI 序列(或任意音符对象流) -> DecodeResult(一次整段解码)。"""
    return Receiver().feed_all(notes).result


def decode_events(events) -> DecodeResult:
    """frontend 的 NoteEvent 流(含 on/off 事件) -> DecodeResult。"""
    return Receiver().feed_all(events).result


def notes_to_text(notes) -> str:
    """音名/MIDI 序列 -> 文本(贪心; 半码截断时返回已解出的前缀, 不报错)。"""
    return decode_notes(notes).text


# --- 损伤扫描(量化退化数据) ---------------------------------------------------

_SCAN_SAMPLES = [                         # 无空串: 空报文无边界收尾可言
    "在吗？",
    "ok",
    "你好，世界！",
    "今晚八点老地方见，别迟到！",
    "这段消息用来测试无纠错解码在删除/插入/替换组合损伤下的退化行为。",
]

_DAMAGES = (                      # (吞音率, 多音率, 弹错率)
    (0.00, 0.00, 0.00),           # 对照: 干净流
    (0.01, 0.01, 0.01),
    (0.01, 0.02, 0.03),
    (0.02, 0.02, 0.02),
    (0.03, 0.03, 0.03),
    (0.05, 0.03, 0.05),           # 压力: 远超实测检测误差量级
)


@dataclass
class ScanRow:
    """损伤扫描的一行统计(比例均为 0..1)。"""

    p_del: float
    p_ins: float
    p_sub: float
    strict: float    # 严格成功: 文本与原文一致且完整收尾(首选即原文)
    partial: float   # 部分成功: 非严格, 但解码在首个损伤点之前正确
                     #   (与原文有非空公共前缀; 单候选贪心解码下「原文是
                     #    候选之一」即退化为这条, 含只丢尾码的轻微情形)
    fail: float      # 失败: 首字符即错(损伤落在消息开头, 无先验可救)
    complete: float  # 完整收尾(码字边界)的比例
    prefix: float    # 平均公共前缀长度 / 原文长度


def _channel(notes, rng, p_del: float, p_ins: float, p_sub: float) -> list[str]:
    """信道损伤模型, 与可靠模型逐条对应:

    - p_del: 整音被吞(检测失败/漏检);
    - p_ins: 多检出杂音(环内任意音高);
    - p_sub: 音高检错(替换为另一个音);
    - 末尾按同音 run 合并: 模拟前端行为(长音拆成同音短音无害)。
    """
    out: list[str] = []
    for n in notes:
        if rng.random() < p_del:
            continue
        if rng.random() < p_ins:
            out.append(rng.choice(NOTES8))
        out.append(n if rng.random() >= p_sub
                   else rng.choice([x for x in NOTES8 if x != n]))
    return [n for i, n in enumerate(out) if i == 0 or n != out[i - 1]]


def _lcp(a: str, b: str) -> int:
    """最长公共前缀长度。"""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _grade(text: str, res: DecodeResult) -> str:
    """三级评定: 严格 / 部分 / 失败(定义见 ScanRow 注释)。"""
    if res.complete and res.text == text:
        return "严格"
    if res.text == text or _lcp(res.text, text) > 0:
        return "部分"
    return "失败"


def damage_scan(trials: int = 200, seed: int = 20260917,
                damages: Optional[Sequence[tuple[float, float, float]]] = None,
                samples: Optional[Sequence[str]] = None) -> list[ScanRow]:
    """损伤扫描: 每组损伤率下 编码->损伤->解码 trials x len(samples) 次。

    固定随机种子保证可复现。返回逐行统计, 排版见 format_scan()。
    """
    rng = random.Random(seed)
    texts = _SCAN_SAMPLES if samples is None else list(samples)
    rows: list[ScanRow] = []
    for p_del, p_ins, p_sub in (_DAMAGES if damages is None else damages):
        strict = partial = fail = comp = lcp_sum = denom = 0
        for _ in range(trials):
            for text in texts:
                notes = _channel(text_to_notes(text), rng,
                                 p_del, p_ins, p_sub)
                res = decode_notes(notes)
                g = _grade(text, res)
                strict += g == "严格"
                partial += g == "部分"
                fail += g == "失败"
                comp += res.complete
                lcp_sum += (max(len(text), 1) if res.text == text
                            else _lcp(res.text, text))
                denom += max(len(text), 1)
        n = trials * len(texts)
        rows.append(ScanRow(p_del, p_ins, p_sub, strict / n, partial / n,
                            fail / n, comp / n, lcp_sum / denom))
    return rows


def format_scan(rows: Sequence[ScanRow]) -> str:
    """扫描结果 -> 比例表(可直接打印)。"""
    head = (f"{'吞音':>6} {'多音':>6} {'弹错':>6} | "
            f"{'严格成功':>8} {'部分成功':>8} {'失败':>7} | "
            f"{'完整收尾':>7} {'平均前缀比':>8}")
    lines = ["无纠错贪心解码退化扫描 (严格=文本一致且完整收尾;"
             " 部分=首个损伤点前正确; 失败=首字符即错)"]
    lines.append(head)
    lines.append("-" * len(head))
    for r in rows:
        lines.append(f"{r.p_del:>7.0%} {r.p_ins:>7.0%} {r.p_sub:>7.0%} | "
                     f"{r.strict:>9.1%} {r.partial:>9.1%} {r.fail:>8.1%} | "
                     f"{r.complete:>8.1%} {r.prefix:>10.1%}")
    return "\n".join(lines)


# --- 自检 ---------------------------------------------------------------------

def _selftest() -> int:
    rng = random.Random(20260917)

    class _Ev:
        """模拟 frontend.engine.NoteEvent 的鸭子类型测试替身。"""

        def __init__(self, kind: str, midi: int, note: str) -> None:
            self.kind, self.midi, self.note = kind, midi, note

    # --- 1. 干净流往返(与发送端 text_to_notes, 即 "no" 模式): 三种喂法 ---
    esc_ch = next(chr(c) for c in range(0x4E00, 0x9FFF)
                  if chr(c) not in CODE_OF)          # 必然走 ESC 转义的表外字
    cases = [
        "", "a", "ok", "aa", "在吗？", "你好，世界！",
        "Hello, world!", "emoji 🎙️🎵 与生僻字 龘 龗 混合。",
        esc_ch, esc_ch * 3, f"表外字{esc_ch}走转义。",
        "".join(chr(c) for c in range(0x20, 0x7F)),
    ]
    for _ in range(200):
        cases.append("".join(rng.choice("abcdefghijklmn abcXYZ.,!?你我他这哪"
                                        "，。！？")
                             for _ in range(rng.randint(0, 60))))
    for _ in range(100):
        # 随机 Unicode(避开代理区: 代理字符无法 UTF-8 编码, 在协议域外)
        cases.append("".join(chr(rng.choice(
            [*range(0, 0xD800), *range(0xE000, 0x10FFF)]))
            for _ in range(rng.randint(0, 40))))
    for text in cases:
        notes = text_to_notes(text)
        for a, b in zip(notes, notes[1:]):           # 发送端约束: 无相邻同音
            assert a != b, f"发送端违反相邻不同音: {text!r}"
        res = decode_notes(notes)
        assert res.text == text, f"音名往返不一致: {text!r}"
        assert res.complete or not text, f"非空流应边界收尾: {text!r}"
        res = decode_notes(text_to_midi(text))
        assert res.text == text, f"MIDI 往返不一致: {text!r}"
        assert res.complete or not text
        res = decode_notes([str(m) for m in text_to_midi(text)])  # 数字字符串
        assert res.text == text, f"数字串往返不一致: {text!r}"
        assert res.complete or not text
        evs: list[_Ev] = []
        for n in notes:
            evs.append(_Ev("on", MIDI8[NOTES8.index(n)], n))
            evs.append(_Ev("off", MIDI8[NOTES8.index(n)], n))  # off 须被忽略
        res = decode_events(evs)
        assert res.text == text, f"事件往返不一致: {text!r}"
        assert res.complete or not text
    n_round = len(cases)

    # --- 2. 边界与鲁棒性 ---
    text = "你好，世界！"
    notes = text_to_notes(text)

    r = Receiver()                       # 空流: 未收任何数字 -> incomplete
    assert r.result.text == "" and not r.result.complete

    res = decode_notes(notes[:-2])       # 截断: 半码收尾, 前缀照常返回
    assert not res.complete and text.startswith(res.text)

    notes_eof = digits_to_notes(text_to_digits(text, eof=True))
    res = decode_notes(notes_eof + ["C4"])   # 遗留 EOF 后余音: 计数不影响文本
    assert res.complete and res.text == text and res.trailing == 1

    for junk in ("C#4", "G3", "A7", "99"):    # 环外杂音: 滤除, 解码不受影响
        res = decode_notes(notes[:3] + [junk] + notes[3:])
        assert res.text == text and res.complete and res.skipped == 1, junk

    res = decode_notes(notes[:3] + [notes[2]] + notes[3:])   # 相邻重复音(多检)
    assert res.text == text and res.complete and res.repeats == 1

    # 杂音 + 重复音 + 遗留 EOF 后余音混合
    res = decode_notes(["X9", notes_eof[0], notes_eof[0], "C#8"]
                       + notes_eof[1:] + ["D4"])
    assert res.text == text and res.complete
    assert res.skipped == 2 and res.repeats == 1 and res.trailing == 1

    # 转义字节损坏: 输出 U+FFFD 并记录 error, 不崩溃, 后续 EOF 照常生效
    body = text_to_digits("ok", eof=True)[:-len(CODE_OF[EOF])]  # 剥掉末尾 EOF
    bad = (body + list(CODE_OF[ESC])
           + list(_byte_digits(0x80)) + list(_byte_digits(0x80))
           + list(CODE_OF[EOF]))
    res = decode_notes(digits_to_notes(bad))
    assert res.complete and res.error and "\ufffd" in res.text

    # --- 3. 损伤扫描(小规模; 完整表格见 simulate 子命令) ---
    rows = damage_scan(trials=50)
    assert rows[0].strict == 1.0 and rows[0].complete == 1.0, \
        "干净流(对照行)应 100% 严格成功"
    assert all(r.strict + r.partial + r.fail == 1.0 for r in rows)

    print(f"selftest ok: {n_round} 个文本用例(音名/MIDI/事件三种喂法往返一致)"
          f" + 截断/余音/杂音/重复音/损坏转义鲁棒性用例 + 损伤扫描 "
          f"{len(rows)} 行 x 50 次全部通过。")
    return 0


# --- 命令行 -------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="口琴信道接收端: 音符序列 -> 文本(无纠错贪心解码)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("命令行:")[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("decode", help="音名/MIDI 序列 -> 文本(贪心, 不报错)")
    pd.add_argument("notes", nargs="?",
                    help="空格/逗号分隔的音名或 MIDI 号; 省略时读 -i 或 stdin")
    pd.add_argument("-i", "--input", help="从文件读取音符序列")

    ps = sub.add_parser("simulate", help="损伤扫描(量化退化表格)")
    ps.add_argument("--trials", type=int, default=200,
                    help="每组损伤率的试验次数(默认 200)")
    ps.add_argument("--del", dest="p_del", type=float, default=None,
                    help="吞音率; 给定则只跑单组(需与 --ins/--sub 同用)")
    ps.add_argument("--ins", dest="p_ins", type=float, default=None,
                    help="多音率")
    ps.add_argument("--sub", dest="p_sub", type=float, default=None,
                    help="弹错率")

    sub.add_parser("selftest", help="往返 + 边界 + 损伤扫描自检")
    args = p.parse_args(argv)

    if args.cmd == "decode":
        cands = decode_assisted(_read_notes(args))
        for rank, c in enumerate(cands, 1):
            star = "  <-- 首选" if rank == 1 else ""
            body = repr(c.text) if c.text else "''"
            print(f"{rank}. {body}{star} "
                  f"(complete={c.complete}, 音符 {c.n_notes})", file=sys.stderr)
        print("首选文本:", cands[0].text if cands else "")
    elif args.cmd == "simulate":
        if args.p_del is not None:
            damages = [(args.p_del,
                        0.0 if args.p_ins is None else args.p_ins,
                        0.0 if args.p_sub is None else args.p_sub)]
        else:
            damages = None
        print(format_scan(damage_scan(args.trials, damages=damages)))
    elif args.cmd == "selftest":
        return _selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
