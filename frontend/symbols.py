"""7+1 音高符号信道 —— 接收端(解码器第二级)。

系统形态: 编码端把信息编成符号序列并告诉用户该按哪些键; 用户在游戏里演奏
(zxcvbnm, 一行键 + 鼠标中键升半音); 信道是游戏音频加采集/检测; 本模块从
检测出的音高还原符号流, 交给后续 RS 解码。

三条约束决定了实现方式:

1. **没有可靠时钟** —— 人演奏的时长不可靠, 符号边界只能由**音高变化**界定。
   所以不能复用 engine._NoteSegmenter: 它按 min_note_s / rest_confirm 这类
   时间阈值切分, 依赖的正是这个不可靠的时钟。
2. **发送端保证无相邻同音**(escape 编码), 接收端才有确定的切分点:
   相邻同音之间必然插了 ESC, 因此每个音高变化就是一个新符号。
3. **字母表 7+1** —— 7 个数据符号 + 1 个 ESC。7 是质数, 便于后续在 GF(7)
   上做 Reed-Solomon。

不折叠: 前端既然能分辨音高, 每个键就对应确定的 MIDI 号, 直接查表即可。

用法:
    spec = CodecSpec()
    ch = SymbolChannel(spec)
    fe = PitchFrontend(sr=48000, on_event=lambda e: ch.on_frame(e)
                       if isinstance(e, FramePitch) else None)
    ...
    ch.flush()
    print(ch.data)          # GF(7) 符号序列, 直接喂 RS
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from . import FileAudioSource, FramePitch, PitchFrontend, WasapiLoopbackSource
from .__main__ import engine_frame_hop, parse_note

# 符号取值: 0..q-1 是数据, 另外两个是信道标记
ESCAPE = -1
ERASURE = -2

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_name(m: int) -> str:
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


@dataclass(frozen=True)
class CodecSpec:
    """收发**共用**的符号表 —— 单一事实来源。

    编码端(信息 -> 符号 -> 该按哪些键)与接收端(音高 -> 符号)都从这里取。
    要改编码方案就改这里, 两端同时生效。

    默认键位: z x c v b n m = C4 D4 E4 F4 G4 A4 B4, ',' = C5 作 ESC。
    若游戏实际发出的音高与 pitches 不符(整体偏高/偏低), 改这一个参数即可,
    解码逻辑不变 —— 查表本来就是按 MIDI 号做的。
    """

    pitches: Tuple[int, ...] = (60, 62, 64, 65, 67, 69, 71)
    escape: int = 72
    keys: Tuple[str, ...] = ("z", "x", "c", "v", "b", "n", "m", ",")

    def __post_init__(self) -> None:
        if len(self.pitches) != 7:
            raise ValueError(f"数据符号必须 7 个, 给了 {len(self.pitches)}")
        if len(set(self.pitches)) != 7:
            raise ValueError("数据音高有重复")
        if self.escape in self.pitches:
            raise ValueError("ESC 不能与数据音高重合(否则剥离时会有歧义)")
        if len(self.keys) != len(self.pitches) + 1:
            raise ValueError(f"按键数应为 {len(self.pitches) + 1}, 给了 {len(self.keys)}")

    @property
    def q(self) -> int:
        """数据字母表大小(质数 7, 对应 GF(7))。"""
        return len(self.pitches)

    def symbol_of(self, midi: int) -> int:
        """MIDI 号 -> 符号。表外一律 ERASURE(信道错误统计的来源)。"""
        if midi == self.escape:
            return ESCAPE
        try:
            return self.pitches.index(midi)
        except ValueError:
            return ERASURE

    def name_of(self, sym: int) -> str:
        if sym == ESCAPE:
            return "ESC"
        if sym == ERASURE:
            return "*"
        return midi_name(self.pitches[sym])

    def key_of(self, sym: int) -> str:
        """符号 -> 该按的键(编码端显示给用户用)。"""
        if sym == ESCAPE:
            return self.keys[-1]
        return self.keys[sym]

    # ---- 编码端(接收端测试也走这条路径, 保证收发规格一致) ----
    def encode(self, symbols: Sequence[int]) -> List[int]:
        """符号序列 -> 音高序列, 相邻同音之间插入 ESC。

        发送端由此保证"无相邻同音", 接收端才能靠音高变化切分。
        ESC 不属于数据字母表, 所以接收端删掉所有 ESC 即可无损还原。
        """
        out: List[int] = []
        prev: Optional[int] = None
        for s in symbols:
            if not 0 <= s < self.q:
                raise ValueError(f"符号越界: {s}")
            if s == prev:
                out.append(self.escape)
            out.append(self.pitches[s])
            prev = s
        return out


def band_for(notes: Sequence[int]) -> Tuple[int, int]:
    """信道用到的音高集合 -> 分析带(最低音下方一个八度, 最高音上方一个八度)。

    上限必须留够, 这不是保守取值而是必需品: 引擎的局部峰检测用 b[1:-1],
    **带的上边界那根 bin 永远不可能被判为峰**, 于是音符的强谐波一旦压在边界上,
    谐波占比门限的分子就只剩弱基频, 整个音被门限掉。实测 C5 的 2 次谐波恰在
    fmax 上时 0/124 帧有声, 放宽一个八度即 23/124。

    GUI 与 CLI 都必须走这里, 否则信道里的高音会被显示成"无声"。
    """
    return min(notes) - 12, max(notes) + 12


@dataclass
class SymbolEvent:
    """一个还原出来的符号。"""

    t: float          # 起始音频时间
    end: float
    sym: int          # 0..q-1 / ESCAPE / ERASURE
    midi: Optional[int]
    frames: int       # 该行程占用的帧数

    @property
    def dur(self) -> float:
        return self.end - self.t


class SymbolChannel:
    """逐帧音高 -> 符号流。

    状态机: 连续同符号帧并为一个符号(无时钟时切分点就是音高变化);
    短的行程判为检测抖动丢弃; ESC 剥离后得到数据流。
    """

    def __init__(self, spec: Optional[CodecSpec] = None,
                 min_run_frames: int = 4, gap_frames: int = 12,
                 emit_erasure: bool = True,
                 on_symbol: Optional[Callable[[SymbolEvent], None]] = None):
        self.spec = spec or CodecSpec()
        self.min_run_frames = min_run_frames
        self.gap_frames = gap_frames
        self.emit_erasure = emit_erasure
        self.on_symbol = on_symbol
        self.events: List[SymbolEvent] = []
        self.data: List[int] = []
        # 统计(RS 参数设计要用)
        self.dropped = 0          # 被去抖丢弃的短行程
        self.erasure_count = 0
        self.escape_count = 0
        self.merged = 0           # 被干扰切断又并回来的符号数
        self.run_hist: List[int] = []
        self._cur: Optional[int] = None
        self._start = 0.0
        self._end = 0.0
        self._n = 0
        self._gap = 0
        self._gap_start = 0.0
        self._last_t = 0.0
        self._cur_midi: Optional[int] = None

    # ---- 输入 ----
    def on_frame(self, fp: FramePitch) -> None:
        self._last_t = fp.t
        if not fp.voiced or fp.midi_smooth is None:
            if self._gap == 0:
                self._gap_start = fp.t
            self._gap += 1
            if self._gap <= self.gap_frames:
                # 短暂掉帧: 判为**上一个符号还没结束**, 不切分。
                # 发送端保证无相邻同音, 所以被干扰切断的同符号本来就该并回
                # 一个符号 —— 在源头不切断比事后合并更干净, 也避免了"两段都
                # 太短被去抖丢弃"导致整个符号丢失。
                return
            self._close_run()          # 掉帧够久, 符号才真的结束
            if self.emit_erasure and self._gap == self.gap_frames + 1:
                self._emit(ERASURE, None, self._gap_start, fp.t, self._gap)
                self.erasure_count += 1
            return
        self._gap = 0
        midi = int(round(fp.midi_smooth))
        sym = self.spec.symbol_of(midi)
        if self._cur is None:
            self._cur, self._start, self._n = sym, fp.t, 1
        elif sym == self._cur:
            self._n += 1
        else:
            self._close_run()
            self._cur, self._start, self._n = sym, fp.t, 1
        self._end = fp.t
        self._cur_midi = midi

    def _close_run(self) -> None:
        if self._cur is None:
            return
        if self._n < self.min_run_frames:
            self.dropped += 1              # 抖动, 不当符号
        else:
            self._emit(self._cur, self._cur_midi, self._start,
                       self._end, self._n)
        self._cur = None
        self._n = 0

    def _emit(self, sym: int, midi: Optional[int], t0: float, t1: float,
              frames: int) -> None:
        # 兜底: 万一还是出现相邻同符号(发送端违规, 或掉帧刚好跨过门限),
        # 按同一语义并入上一个符号, 而不是当成两个符号
        if (sym != ERASURE and self.events
                and self.events[-1].sym == sym):
            last = self.events[-1]
            last.end = t1
            last.frames += frames
            self.merged += 1
            return
        ev = SymbolEvent(t0, t1, sym, midi, frames)
        self.events.append(ev)
        if self.on_symbol is not None:
            self.on_symbol(ev)
        if sym == ESCAPE:
            self.escape_count += 1
            self.run_hist.append(frames)
        elif sym >= 0:
            self.run_hist.append(frames)   # 擦除不计入行程分布
            self.data.append(sym)          # ESC 剥离 → 数据流

    def flush(self) -> None:
        self._close_run()

    # ---- 输出 ----
    def raw_text(self, wrap: int = 0) -> str:
        """原始流: 含 ESC 与擦除, 便于核对。"""
        return " ".join(self.spec.name_of(e.sym) for e in self.events)

    def data_text(self, wrap: int = 40) -> str:
        """数据流: 空格分隔的 GF(7) 符号, 直接喂 RS。"""
        items = [str(s) for s in self.data]
        if wrap <= 0:
            return " ".join(items)
        return "\n".join(" ".join(items[i:i + wrap])
                         for i in range(0, len(items), wrap))

    def detail_text(self) -> str:
        out = []
        for e in self.events:
            flags = []
            if e.sym == ESCAPE:
                flags.append("ESC")
            elif e.sym == ERASURE:
                flags.append("ERASURE")
            elif e.frames < self.min_run_frames * 2:
                flags.append("short")
            m = "-" if e.midi is None else str(e.midi)
            out.append(f"t={e.t:7.3f}  {self.spec.name_of(e.sym):>4}  "
                       f"midi={m:>3}  sym={e.sym:>3}  run={e.frames:>3}  "
                       f"flags={','.join(flags) if flags else '-'}")
        return "\n".join(out)

    def stats_text(self) -> str:
        from collections import Counter
        lines = [f"字母表 q={self.spec.q} (GF({self.spec.q}))",
                 f"符号总数 {len(self.data)}",
                 f"干扰合并    {self.merged}"
                 "  (发送端保证无相邻同音, 故相邻同符号即被干扰切断, 已并回)",
                 ""]
        hist = Counter(self.data)
        lines.append("符号直方图:")
        for s in range(self.spec.q):
            c = hist.get(s, 0)
            bar = "#" * min(60, c)
            lines.append(f"  {s} {self.spec.name_of(s):>4} "
                         f"[{self.spec.key_of(s)}] {c:>5}  {bar}")
        lines.append("")
        lines.append(f"ESC 数        {self.escape_count}")
        lines.append(f"擦除数        {self.erasure_count}")
        lines.append(f"去抖丢弃行程  {self.dropped}")
        if self.run_hist:
            rh = sorted(self.run_hist)
            lines.append(f"行程帧数  最短 {rh[0]}  中位 {rh[len(rh) // 2]}  "
                         f"最长 {rh[-1]}  (去抖门限 {self.min_run_frames})")
        return "\n".join(lines)


def _parse_pitches(s: str) -> Tuple[int, ...]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(parse_note(part) if part[0].isalpha() else int(part))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="7+1 音高符号信道(接收端): 音频 -> GF(7) 符号流")
    ap.add_argument("infile", nargs="?", default=None)
    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--lo", default=None,
                    help="分析带下限音名; 缺省由符号表推出(最高音下方一个八度)")
    ap.add_argument("--hi", default=None,
                    help="分析带上限音名; 缺省由符号表推出(最高音上方一个八度)")
    ap.add_argument("--hp", type=float, default=100.0)
    ap.add_argument("--f0-method", choices=("yin", "peak"), default="yin")
    ap.add_argument("--pitches", default="60,62,64,65,67,69,71",
                    help="7 个数据符号的 MIDI 号(或音名), 逗号分隔")
    ap.add_argument("--escape", default="72", help="ESC 的 MIDI 号或音名")
    ap.add_argument("--min-run-frames", type=int, default=4,
                    help="行程短于此帧数判为抖动丢弃")
    ap.add_argument("--gap-frames", type=int, default=12,
                    help="静音超过此帧数输出一个擦除")
    ap.add_argument("--no-erasure", action="store_true")
    ap.add_argument("--csv", default=None, help="明细日志输出文件")
    ap.add_argument("--data-out", default=None, help="GF(7) 符号流输出文件")
    args = ap.parse_args()

    esc = parse_note(args.escape) if args.escape[0].isalpha() else int(args.escape)
    spec = CodecSpec(_parse_pitches(args.pitches), esc)

    if args.loopback:
        src = WasapiLoopbackSource(device=args.device)
        print(f"回环采集: {src.device_name!r} (idx={src.device_index})")
    else:
        if not args.infile:
            ap.error("需要音频文件, 或用 --loopback")
        src = FileAudioSource(args.infile)

    # 分析带由符号表推出, 而不是写死: 引擎的局部峰检测用 b[1:-1], **带的上边界
    # 那根 bin 永远不可能被判为峰**, 于是只要某个音符的强谐波正好压在边界上,
    # 谐波占比门限的分子就只剩弱基频, 该音会被整个门限掉(实测 ESC 音 C5:
    # 2 次谐波恰在 fmax 上时 0/124 帧有声, 放宽一点即 23/124)。
    # 故上限取"符号表最高音再高一个八度", 保证至少两次谐波落在带内。
    lo_d, hi_d = band_for(spec.pitches + (spec.escape,))
    lo_note = parse_note(args.lo) if args.lo else lo_d
    hi_note = parse_note(args.hi) if args.hi else hi_d

    frame, hop = engine_frame_hop(src.sample_rate, 4096, 512)
    ch = SymbolChannel(spec, args.min_run_frames, args.gap_frames,
                       not args.no_erasure)
    fe = PitchFrontend(sr=src.sample_rate, frame=frame, hop=hop,
                       range_midi=(lo_note, hi_note),
                       f0_method=args.f0_method, hp_hz=args.hp,
                       on_event=lambda e: ch.on_frame(e)
                       if isinstance(e, FramePitch) else None)

    print(f"采样率 {src.sample_rate} Hz, 窗/帧移 {frame}/{hop}, "
          f"分析带 {midi_name(lo_note)}~{midi_name(hi_note)}")
    print("符号表: " + "  ".join(f"{spec.key_of(i)}={spec.name_of(i)}"
                                 for i in range(spec.q))
          + f"  {spec.key_of(ESCAPE)}=ESC")
    try:
        used = 0.0
        for chunk in src:
            fe.process(chunk)
            used += len(chunk) / src.sample_rate
            if args.duration and used >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n[中断] 收尾中 ...")
    finally:
        fe.flush()
        ch.flush()
        src.close()

    print()
    print(ch.stats_text())
    print("\n原始流(含 ESC 与擦除):")
    print("  " + ch.raw_text())
    print("\n数据流 GF(7):")
    print(ch.data_text())

    if args.data_out:
        with open(args.data_out, "w", encoding="utf-8") as f:
            f.write(ch.data_text() + "\n")
        print(f"\n-> {args.data_out}")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(ch.detail_text() + "\n")
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
