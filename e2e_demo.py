#!/usr/bin/env python3
"""端到端演示: 游戏内实录音频 -> 音高检测 -> 接收端解码 -> 明文。

用法:
    python3 e2e_demo.py [音频文件] [期望明文]
缺省: recordings/t1_no.mp3, 期望 '有人堵桥' (无纠错档实录)。
"""
import sys

from frontend import FileAudioSource, PitchFrontend, NoteEvent
from frontend.__main__ import engine_frame_hop
from receiver import Receiver


def decode_file(path: str):
    events = []
    src = FileAudioSource(path, chunk_ms=20)

    def collect(e):
        events.append(e)

    frame, hop = engine_frame_hop(src.sample_rate, 4096, 512)
    fe = PitchFrontend(sr=src.sample_rate, frame=frame, hop=hop, on_event=collect)
    for chunk in src:
        fe.process(chunk)
    fe.flush()
    notes = [ev.note for ev in events
             if isinstance(ev, NoteEvent) and ev.kind == "on"]
    rx = Receiver()
    for n in notes:
        rx.feed(n)
    return notes, rx.result


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "recordings/t1_no.mp3"
    expect = sys.argv[2] if len(sys.argv) > 2 else "有人堵桥"
    notes, res = decode_file(path)
    print(f"检测到 {len(notes)} 个音符:", " ".join(notes))
    print(f"解码: {res.text!r} complete={res.complete}")
    print(f"摘要: {res.summary()}")
    mark = "✓ 命中" if res.text == expect else "✗ 不符"
    print(f"明文应为 {expect!r} -> {mark}")
    return 0 if res.text == expect else 1


if __name__ == "__main__":
    sys.exit(main())
