"""命令行: 离线批处理与实时流式模拟。

    python -m frontend test.mp3                  # 离线, 写 pitch.csv / notes.csv
    python -m frontend test.mp3 --stream-sim     # 分块推送模拟实时, 打印事件流/延迟
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import List, Optional, Tuple

import numpy as np

from . import FileAudioSource, FramePitch, NoteEvent, PitchFrontend
from .engine import midi_to_note

_NOTE_INDEX = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def parse_note(s: str) -> int:
    """音名转 MIDI 编号: C3->48, F#4->66, B5->83。"""
    s = s.strip().upper()
    i = 0
    while i < len(s) and (s[i].isalpha() or s[i] == "#"):
        i += 1
    name, octave = s[:i], int(s[i:])
    if name not in _NOTE_INDEX:
        raise ValueError(f"无法解析音名: {s!r}")
    return _NOTE_INDEX[name] + 12 * (octave + 1)


def engine_frame_hop(sr: int, frame: int, hop: int) -> Tuple[int, int]:
    """把 48kHz 调好的窗长/帧移按采样率等比缩放, 保持时间尺度不变。

    引擎的 4096/512 对应 85.3ms/10.7ms; 若端点跑 44.1kHz 而直接沿用样本数,
    窗长会变成 92.9ms, median/smooth/min_note_ms 等按时间定的参数全部失准。
    """
    if sr == 48000:
        return frame, hop
    k = sr / 48000.0
    return (max(2, int(round(frame * k)) // 2 * 2),
            max(1, int(round(hop * k)) // 2 * 2))


def run(path: Optional[str], stream_sim: bool, chunk_ms: float, realtime: bool,
        silence_db, min_note_ms: float, csv_path: str, notes_path: str,
        range_midi, f0_method: str = "yin", hp_hz: float = 100.0,
        loopback: bool = False, device=None, duration: float = 0.0) -> None:
    events: List[Tuple[float, object]] = []   # (产出时的音频进度, 事件)
    now = {"t": 0.0}

    def collect(e):
        events.append((now["t"], e))

    if loopback:
        from .audio import WasapiLoopbackSource
        src = WasapiLoopbackSource(device=device, chunk_ms=chunk_ms)
        print(f"回环采集: {src.device_name!r} (idx={src.device_index}) "
              f"ch={src.channels}")
    else:
        src = FileAudioSource(path, chunk_ms=chunk_ms)

    frame, hop = engine_frame_hop(src.sample_rate, 4096, 512)
    fe = PitchFrontend(sr=src.sample_rate, frame=frame, hop=hop,
                       silence_db=silence_db,
                       min_note_ms=min_note_ms, range_midi=range_midi,
                       f0_method=f0_method, hp_hz=hp_hz, on_event=collect)
    print(f"采样率 {src.sample_rate} Hz, 块长 {chunk_ms:.1f} ms, "
          f"窗/帧移 {frame}/{hop}, "
          f"音域 MIDI {range_midi[0]}~{range_midi[1]}, "
          f"模式: {'回环采集' if loopback else ('流式模拟' if stream_sim else '离线批处理')}"
          + (f", 时长 {duration:.1f}s" if duration else ""))

    t0 = time.perf_counter()
    try:
        for chunk in src:
            fe.process(chunk)
            now["t"] += len(chunk) / src.sample_rate
            if realtime:
                dt = (t0 + now["t"]) - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
            if duration and now["t"] >= duration:
                break
    except KeyboardInterrupt:
        print("\n[中断] 收尾中 ...")
    finally:
        fe.flush()
        src.close()
    wall = time.perf_counter() - t0

    frames = [e for _, e in events if isinstance(e, FramePitch)]
    notes = [e for _, e in events if isinstance(e, NoteEvent)]

    # on/off 配对成音符区间
    note_rows, open_on = [], {}
    for e in notes:
        if e.kind == "on":
            open_on[e.midi] = e
        else:
            s = open_on.pop(e.midi, None)
            if s is not None:
                note_rows.append((s.t, e.t, e.midi, e.note))
    for e in open_on.values():   # 流尾未闭合的 on
        note_rows.append((e.t, frames[-1].t if frames else e.t, e.midi, e.note))
    note_rows.sort()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "freq_hz", "midi", "note", "voiced"])
        for fr in frames:
            if fr.voiced:
                w.writerow([f"{fr.t:.4f}", f"{fr.freq:.2f}",
                            f"{fr.midi_smooth:.2f}", midi_to_note(fr.midi_smooth), 1])
            else:
                w.writerow([f"{fr.t:.4f}", "", "", "", 0])
    with open(notes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "dur_s", "midi", "note"])
        for s, e_, m, name in note_rows:
            w.writerow([f"{s:.3f}", f"{e_:.3f}", f"{e_ - s:.3f}", m, name])

    voiced = sum(1 for fr in frames if fr.voiced)
    print(f"帧 {len(frames)}, 有声 {voiced} ({voiced / max(1, len(frames)) * 100:.1f}%), "
          f"音符 {len(note_rows)}, 处理耗时 {wall:.2f}s")
    print("\n音符序列:")
    print("  " + "  ".join(n for *_, n in note_rows))

    if stream_sim:
        # 事件延迟(音频时间轴): 事件产出时已推入的音频量 - 事件自身时间戳
        print("\n音符事件流(前 24 条, @ = 产出时的音频进度):")
        shown = 0
        for a, e in events:
            if isinstance(e, NoteEvent):
                print(f"  @{a:7.3f}s  {e.t:7.3f}s  note-{e.kind:<3} {e.note}")
                shown += 1
                if shown >= 24:
                    break
        on_lat = [a - e.t for a, e in events
                  if isinstance(e, NoteEvent) and e.kind == "on"]
        fr_lat = [a - e.t for a, e in events if isinstance(e, FramePitch)]
        if fr_lat:
            print(f"帧事件延迟: 均值 {np.mean(fr_lat)*1000:.0f} ms, "
                  f"最大 {np.max(fr_lat)*1000:.0f} ms")
        if on_lat:
            print(f"音符 on 延迟: 均值 {np.mean(on_lat)*1000:.0f} ms, "
                  f"最大 {np.max(on_lat)*1000:.0f} ms")

    print(f"\n逐帧序列 -> {csv_path}")
    print(f"音符序列 -> {notes_path}")


def main():
    ap = argparse.ArgumentParser(description="音高清码前端")
    ap.add_argument("infile", nargs="?", default=None,
                    help="音频文件; --loopback 时不需要")
    ap.add_argument("--loopback", action="store_true",
                    help="采集系统声音回环(WASAPI loopback, 不装虚拟声卡)")
    ap.add_argument("--device", default=None,
                    help="回环端点: 名字子串或 index; 缺省用系统默认输出")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="回环采集时长(秒), 0=不限直到 Ctrl-C")
    ap.add_argument("--list-devices", action="store_true",
                    help="列出可用的回环端点后退出")
    ap.add_argument("--stream-sim", action="store_true",
                    help="分块推送模拟实时流(默认一次性整块推送)")
    ap.add_argument("--chunk-ms", type=float, default=20.0)
    ap.add_argument("--realtime", action="store_true",
                    help="stream-sim 时按真实时间节奏推送")
    ap.add_argument("--silence-db", type=float, default=None,
                    help="固定静音门限(dB); 缺省用滑动最小统计自适应")
    ap.add_argument("--f0-method", choices=("yin", "peak"), default="yin",
                    help="基频估计: yin=自相关(默认, 抗弱基频); "
                         "peak=旧行为(带内最高谱峰)")
    ap.add_argument("--hp", type=float, default=100.0,
                    help="音域以下高通截止频率(Hz), 0 关闭")
    ap.add_argument("--lo", default="C3", help="乐器最低音(音名, 默认 C3)")
    ap.add_argument("--hi", default="B5", help="乐器最高音(音名, 默认 B5)")
    ap.add_argument("--min-note-ms", type=float, default=100.0)
    ap.add_argument("--csv", default="pitch.csv")
    ap.add_argument("--notes-csv", default="notes.csv")
    args = ap.parse_args()

    if args.list_devices:
        from .audio import list_loopback_devices
        devs = list_loopback_devices()
        if not devs:
            print("没有可用的回环端点(检查系统是否有活动的输出设备)")
        for d in devs:
            mark = " <- 默认" if d["default"] else ""
            print(f"  idx={d['index']:3d} ch={d['channels']} "
                  f"sr={d['sample_rate']:6d} {d['name']}{mark}")
        return

    if not args.loopback and not args.infile:
        ap.error("需要给出音频文件, 或用 --loopback 采集系统声音")
    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    run(args.infile, args.stream_sim, args.chunk_ms, args.realtime,
        args.silence_db, args.min_note_ms, args.csv, args.notes_csv,
        (parse_note(args.lo), parse_note(args.hi)), args.f0_method, args.hp,
        loopback=args.loopback, device=device, duration=args.duration)


if __name__ == "__main__":
    main()
