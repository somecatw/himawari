"""解码前端: 音频源抽象。

部署环境是 Windows(游戏音频采集), 开发环境是 WSL。因此这里只定义接口和
基于文件/数组/回环的实现, 接入新音源只需实现 AudioSource, 引擎代码零改动。
"""
from __future__ import annotations

import queue
import time
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

import numpy as np


class AudioSource(ABC):
    """单声道音频源: 迭代产出 float32/float64 的一维采样块。

    约定:
    - 采样率在打开后可用(sample_rate 属性);
    - 迭代结束即流结束; 支持上下文管理器;
    - 线程模型由具体实现自行保证, 引擎假设单线程顺序 push。
    """

    sample_rate: int

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FileAudioSource(AudioSource):
    """文件音频源: 用 soundfile 分块解码, 混为单声道。开发/回归测试用。"""

    def __init__(self, path: str, chunk_ms: float = 20.0):
        import soundfile as sf  # 延迟导入, 引擎本身不依赖

        self._file = sf.SoundFile(path)
        self.sample_rate = int(self._file.samplerate)
        self._chunk = max(1, int(self.sample_rate * chunk_ms / 1000.0))

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            block = self._file.read(self._chunk, dtype="float64", always_2d=True)
            if len(block) == 0:
                break
            yield block.mean(axis=1)

    def close(self) -> None:
        self._file.close()


def list_loopback_devices() -> List[dict]:
    """列出可用的回环端点(供 CLI --list-devices 与排错用)。"""
    import pyaudiowpatch as pa

    p = pa.PyAudio()
    try:
        default = None
        try:
            default = p.get_default_wasapi_loopback()["index"]
        except Exception:
            pass
        out = []
        for d in p.get_loopback_device_info_generator():
            out.append({
                "index": int(d["index"]),
                "name": str(d["name"]),
                "channels": int(d["maxInputChannels"]),
                "sample_rate": int(d["defaultSampleRate"]),
                "default": d["index"] == default,
            })
        return out
    finally:
        p.terminate()


class WasapiLoopbackSource(AudioSource):
    """Windows 系统声音回环采集(WASAPI loopback)。

    **不注册任何虚拟音频设备**: WASAPI loopback 是 Windows 自带的用户态 API,
    在**已有的**渲染端点上开一条输入流(OBS 的"桌面音频"、Game Bar 录屏同源)。
    不装驱动、不注入进程、不 hook、不改音频拓扑。

    依赖 PyAudioWPatch(PyAudio + PortAudio v19 的 fork)。注意 sounddevice
    做不了这件事: 它的高层 API 没有 loopback, `PaWasapi_IsLoopback()` 只在
    其 cdef 里且注明 "high-level interface not yet available"。

    为什么用 callback + 队列而不是直接阻塞 read() —— 两条实测结论:
      1. 端点上**没有任何东西在渲染**时, PortAudio 的 read() **无限阻塞**
         (无播放时 8 秒内 0 次返回; 恢复播放后阻塞中的读才返回)。若直接阻塞读,
         系统静音期间解码器时间轴会整个停住: 帧序号不再前进、音符不收尾、
         恢复后时间戳错乱。故迭代器取不到数据时按墙钟补一块静音。
      2. 自建线程阻塞在 read()、另一线程去 stop_stream()/close() 会直接
         **SIGSEGV**(实测 exit 139)。改用 callback 流让 PortAudio 自己管线程,
         关闭路径就只剩标准操作。
    静音按**墙钟记账**补齐(不是"每次队列超时补一块" —— Windows 等待粒度约
    15.6ms 会把 20ms 超时拖成 ~31ms, 实测静音期间时间轴比真实时间慢 35%),
    时间轴匀速前进, 静音如实当静音。

    采样率取端点自身的 defaultSampleRate(通常 48000, 不写死); 调用方需按
    sr/48000 等比缩放引擎的 frame/hop, 否则时间窗会偏。
    """

    def __init__(self, device: Optional[object] = None, chunk_ms: float = 20.0):
        import pyaudiowpatch as pa

        self._pa = pa.PyAudio()
        self._closed = False
        try:
            info = self._resolve(device)
            self.device_index = int(info["index"])
            self.device_name = str(info["name"])
            self.channels = int(info["maxInputChannels"])
            self.sample_rate = int(info["defaultSampleRate"])
            self._chunk = max(1, int(self.sample_rate * chunk_ms / 1000.0))
            self._q: queue.Queue = queue.Queue(maxsize=64)
            # 用 callback 流: PortAudio 自己管读线程。
            # 不用"自建线程阻塞 read()"是因为关闭时会崩 —— 实测一个线程阻塞在
            # read()、另一线程调 stop_stream()/close(), 直接 SIGSEGV(exit 139)。
            self._stream = self._pa.open(
                format=pa.paFloat32, channels=self.channels,
                rate=self.sample_rate, input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self._chunk,
                stream_callback=self._callback)
        except Exception:
            self._pa.terminate()
            raise

    def _resolve(self, device: Optional[object]) -> dict:
        """None=默认回环端点; int=按 index; str=按名字子串(不区分大小写)。"""
        p = self._pa
        if device is None:
            try:
                return p.get_default_wasapi_loopback()
            except Exception as e:
                raise RuntimeError(
                    "找不到默认回环端点。请确认系统有活动的输出设备, "
                    "或用 list_loopback_devices() 查看可用端点。"
                ) from e
        if isinstance(device, int):
            for d in p.get_loopback_device_info_generator():
                if int(d["index"]) == device:
                    return d
            raise ValueError(f"没有 index={device} 的回环端点")
        key = str(device).lower()
        for d in p.get_loopback_device_info_generator():
            if key in str(d["name"]).lower():
                return d
        raise ValueError(f"没有名字含 {device!r} 的回环端点")

    def _callback(self, in_data, frame_count, time_info, status):
        """PortAudio 回调: 只做格式转换 + 入队, 绝不阻塞。"""
        import pyaudiowpatch as pa

        a = np.frombuffer(in_data, np.float32)
        if self.channels > 1 and a.size % self.channels == 0:
            a = a.reshape(-1, self.channels).mean(axis=1)
        try:
            self._q.put_nowait(a)
        except queue.Full:
            pass                            # 消费端跟不上就丢块, 不阻塞回调
        return (None, pa.paContinue)

    def __iter__(self) -> Iterator[np.ndarray]:
        chunk_s = self._chunk / self.sample_rate
        # 补静音按墙钟记账, 而不是"每次超时补一块": Windows 的等待粒度约 15.6ms,
        # 会让 20ms 的队列超时实际等 ~31ms, 于是静音期间时间轴比真实时间慢 ~35%,
        # 静音越久后续时间戳偏得越多。记账后补齐速率严格跟随墙钟。
        next_t = time.perf_counter()
        while not self._closed:
            try:
                a = self._q.get(timeout=chunk_s)
                next_t = time.perf_counter()    # 有真实数据就以它为准重新对齐
                yield a
                continue
            except queue.Empty:
                pass
            now = time.perf_counter()
            n = int((now - next_t) / chunk_s)
            if n <= 0:
                continue
            next_t += n * chunk_s
            for _ in range(min(n, 50)):         # 上限防长时间静音后一次喷一大串
                yield np.zeros(self._chunk)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True                 # 先让迭代器退出, 不再取队列
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass
        self._pa.terminate()
