"""解码前端: 音频源抽象 + 流式音高/音符引擎。

快速上手:
    from frontend import PitchFrontend, FileAudioSource

    fe = PitchFrontend(sr=48000, on_event=decoder.consume)  # 解码模块挂回调
    with FileAudioSource("test.mp3") as src:                # 部署时换 WasapiLoopbackSource
        for chunk in src:
            fe.process(chunk)
    fe.flush()

命令行:
    python -m frontend test.mp3                 # 离线批处理 -> pitch.csv / notes.csv
    python -m frontend test.mp3 --stream-sim    # 分块推送模拟实时, 打印事件流与延迟
    python -m frontend --list-devices           # 列出可回环采集的输出端点
    python -m frontend --loopback               # 采集系统声音(Ctrl-C 收尾)
    python -m frontend --loopback --device Realtek --duration 30
"""
from .audio import (AudioSource, FileAudioSource, WasapiLoopbackSource,
                    list_loopback_devices)
from .engine import (Event, FramePitch, NoteEvent, PitchFrontend,
                     hz_to_midi, midi_to_hz, midi_to_note)

__all__ = [
    "AudioSource", "FileAudioSource", "WasapiLoopbackSource",
    "list_loopback_devices",
    "PitchFrontend", "FramePitch", "NoteEvent", "Event",
    "hz_to_midi", "midi_to_hz", "midi_to_note",
]
