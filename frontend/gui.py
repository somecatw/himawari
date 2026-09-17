"""实时监视 GUI: 频谱瀑布 + 当前音高。

用途是**肉眼验证跟踪对不对**: 瀑布图上能直接看到谐波列, 一眼看出程序锁的是
基频还是某个谐波(这正是 test2.mp3 当初识别错的原因)。

    python -m frontend.gui test2.mp3 --hi C6      # 文件按原速播放
    python -m frontend.gui --loopback --hi C6     # 采集系统声音

实现要点:
- **纯 stdlib**: tkinter + numpy, 不引入 Pillow/matplotlib。每帧重建整幅
  RGB PhotoImage 实测 1.80ms(555fps), 远够 30fps 用。
- **线程**: 音源迭代是阻塞的, 引擎只能在单个线程里跑 -> 工作线程负责
  采集+分析, 结果丢队列; tkinter 只能在主线程 -> root.after 轮询重绘。
- **配色**: sequential 单色相(蓝 700->100, 深色底上"接近零"退向底色),
  不用彩虹色图。取自 dataviz skill 的参考调色板。
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk

import numpy as np

from . import WasapiLoopbackSource, FileAudioSource, FramePitch, PitchFrontend
from .engine import midi_to_note
from .__main__ import engine_frame_hop, parse_note

# ---- 取自 dataviz skill 参考调色板(深色模式) ----
SURFACE = "#1a1a19"      # 图表底色
PLANE = "#0d0d0d"        # 页面底
INK = "#ffffff"          # 主文本
INK_2 = "#c3c2b7"        # 次文本
MUTED = "#898781"        # 轴/次要标签
GRID = "#2c2c2a"         # 网格发丝线
AXIS = "#383835"         # 基线

# sequential 蓝 700 -> 100(light->dark 的反向: 深色底上亮 = 能量大)
_RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
         "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6",
         "#cde2fb"]


def _lut() -> np.ndarray:
    """把 ramp 插值成 256 级 RGB 查找表。"""
    pts = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in _RAMP],
                   dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(pts))
    dst = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(dst, src, pts[:, c]) for c in range(3)],
                    axis=1).astype(np.uint8)


LUT = _lut()


class _Spectrum:
    """音频 -> 对数频率轴的显示列。"""

    def __init__(self, sr: int, nbins: int = 160, n_fft: int = 8192,
                 fmin: float = 60.0, fmax: float = 4000.0):
        self.n_fft = n_fft
        self.nbins = nbins
        self.fmin, self.fmax = fmin, fmax
        self.win = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        edges = fmin * (fmax / fmin) ** (np.arange(nbins + 1) / nbins)
        idx = np.clip(np.searchsorted(freqs, edges), 0, len(freqs) - 1)
        self.bands = []
        lo = int(idx[0])
        for b in range(nbins):
            hi = max(int(idx[b + 1]), lo + 1)
            self.bands.append((lo, hi))
            lo = hi
        self._ref = 1e-3          # 自适应参考电平(缓慢衰减)

    def column(self, x: np.ndarray) -> np.ndarray:
        """返回 (nbins,) 的 0~1 强度, 低频在前。"""
        seg = x[-self.n_fft:]
        if len(seg) < self.n_fft:
            seg = np.pad(seg, (self.n_fft - len(seg), 0))
        spec = np.abs(np.fft.rfft(seg * self.win))
        col = np.array([spec[lo:hi].max() for lo, hi in self.bands])
        db = 20.0 * np.log10(col + 1e-12)
        peak = float(db.max())
        # 参考电平跟随峰值但缓慢回落, 免得强弱段互相压制
        self._ref = max(peak, self._ref - 0.4)
        v = (db - (self._ref - 70.0)) / 70.0
        return np.clip(v, 0.0, 1.0)

    def y_of(self, f: float, height: int) -> float:
        """频率 -> 画布 y(低频在下)。"""
        t = np.log(f / self.fmin) / np.log(self.fmax / self.fmin)
        return height * (1.0 - t)


class _Worker(threading.Thread):
    """后台: 音源 -> 引擎 -> 显示队列。"""

    def __init__(self, src, fe, spec, out: queue.Queue, pace: bool):
        super().__init__(daemon=True)
        self.src, self.fe, self.spec, self.out, self.pace = src, fe, spec, out, pace
        self.stop = threading.Event()
        self.error: str | None = None

    def run(self) -> None:
        buf = np.zeros(8192)
        t0 = time.perf_counter()
        used = 0.0
        try:
            for chunk in self.src:
                if self.stop.is_set():
                    break
                self.fe.process(chunk)
                used += len(chunk) / self.src.sample_rate
                buf = np.concatenate([buf, chunk])[-self.spec.n_fft:]
                self.out.put(("col", self.spec.column(buf)))
                if self.pace:                      # 文件按原速播, 才看得出过程
                    dt = (t0 + used) - time.perf_counter()
                    if dt > 0:
                        time.sleep(dt)
        except Exception as e:                     # 别让线程静默死掉
            self.error = f"{type(e).__name__}: {e}"
        finally:
            try:
                self.fe.flush()
            except Exception:
                pass
            self.out.put(("end", self.error))


class App:
    W, H = 440, 170         # 瀑布图尺寸
    NBINS = 160
    TRAJ_H = 34

    def __init__(self, root: tk.Tk, src, fe, sr: int, pace: bool, title: str):
        self.root = root
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.spec = _Spectrum(sr, self.NBINS, fmax=4000.0)

        root.title(title)
        root.configure(bg=PLANE)

        wrap = tk.Frame(root, bg=PLANE, padx=10, pady=8)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="频谱瀑布 (最近几秒)", bg=PLANE, fg=MUTED,
                 font=("Consolas", 9)).pack(anchor="w")
        self.canvas = tk.Canvas(wrap, width=self.W, height=self.H,
                                bg=SURFACE, highlightthickness=0)
        self.canvas.pack()
        self._draw_chrome()

        read = tk.Frame(wrap, bg=PLANE, pady=6)
        read.pack(fill="x")
        self.note_lbl = tk.Label(read, text="---", bg=PLANE, fg=INK,
                                 font=("Consolas", 26, "bold"))
        self.note_lbl.pack(side="left")
        self.det_lbl = tk.Label(read, text="", bg=PLANE, fg=INK_2,
                                font=("Consolas", 10), justify="left")
        self.det_lbl.pack(side="left", padx=12)

        tk.Label(wrap, text="音高轨迹", bg=PLANE, fg=MUTED,
                 font=("Consolas", 9)).pack(anchor="w")
        self.traj = tk.Canvas(wrap, width=self.W, height=self.TRAJ_H,
                              bg=SURFACE, highlightthickness=0)
        self.traj.pack()

        self.hist: list[float] = []
        self.img_arr = np.zeros((self.NBINS, self.W, 3), dtype=np.uint8)
        self.img_arr[:, :] = LUT[0]
        self._photo = None
        self._item = self.canvas.create_image(0, 0, anchor="nw")
        self._render()

        # 线程在 start() 里才启动 —— 要等 fe.on_event 接好, 否则开头几帧丢失
        self.worker = _Worker(src, fe, self.spec, self.q, pace)
        self._ended = False
        self._got_end = False
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- 静态图层: 频率刻度 + 网格(画在图像之上) ----
    def _draw_chrome(self) -> None:
        for f in (125, 250, 500, 1000, 2000, 4000):
            y = self.spec.y_of(f, self.H)
            if 2 < y < self.H - 2:
                self.canvas.create_line(0, y, self.W, y, fill=GRID)
                lab = f"{f // 1000}k" if f >= 1000 else str(f)
                self.canvas.create_text(4, y - 7, anchor="w", text=lab,
                                        fill=MUTED, font=("Consolas", 8))
        self.canvas.create_text(self.W - 4, self.H - 8, anchor="e",
                                text="Hz", fill=MUTED, font=("Consolas", 8))

    def _render(self) -> None:
        rgb = np.ascontiguousarray(self.img_arr[::-1]).tobytes()
        ppm = b"P6\n%d %d\n255\n" % (self.W, self.NBINS) + rgb
        self._photo = tk.PhotoImage(data=ppm)
        self.canvas.itemconfig(self._item, image=self._photo)

    # ---- 主线程轮询 ----
    def tick(self) -> None:
        dirty = ui = False
        for _ in range(64):
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "col":
                self.img_arr = np.roll(self.img_arr, -1, axis=1)
                self.img_arr[:, -1] = LUT[(payload * 255).astype(np.uint8)]
                dirty = True
            elif kind == "ui":
                # 音高历史只在主线程维护: 工作线程只投递数值, 不碰共享列表
                self.hist.append(payload[2] if payload else np.nan)
                del self.hist[:-self.W]
                if payload:
                    note, freq, midi_v, pw = payload
                    self.note_lbl.config(text=note, fg=INK)
                    self.det_lbl.config(text=f"{freq:7.1f} Hz\n"
                                             f"MIDI {midi_v:5.2f}\n"
                                             f"{pw:5.1f} dB", fg=INK_2)
                else:
                    self.note_lbl.config(text="---", fg=MUTED)
                    self.det_lbl.config(text="无声", fg=MUTED)
                ui = True
            elif kind == "end":
                self._got_end = True
                if payload:
                    self.det_lbl.config(text=f"工作线程异常: {payload}",
                                        fg="#ff6b6b")
                else:
                    self.det_lbl.config(text="输入结束", fg=MUTED)
                ui = True
        if dirty:
            self._render()
        if ui:
            self._render_traj()
        if not self.running_ok():
            return
        self.root.after(33, self.tick)

    def running_ok(self) -> bool:
        """工作线程若**意外**退出, 界面会静默冻住 —— 显式提示。

        正常收尾时线程也会退出, 且已经发过 "end"; 那种情况不能再报错,
        否则文件放完会显示一行红色的"工作线程已退出", 看着像出故障。
        """
        if self.worker.is_alive() or self._got_end:
            return True
        if not self._ended:
            self._ended = True
            self.det_lbl.config(text="工作线程意外退出", fg="#ff6b6b")
        return True

    def _render_traj(self) -> None:
        self.traj.delete("all")
        vals = [v for v in self.hist if not np.isnan(v)]
        if not vals:
            return
        lo, hi = min(vals) - 1, max(vals) + 1
        if hi - lo < 6:
            mid = (hi + lo) / 2
            lo, hi = mid - 3, mid + 3
        pts, x0 = [], self.W - len(self.hist)
        for i, v in enumerate(self.hist):
            if np.isnan(v):
                continue
            y = self.TRAJ_H - (v - lo) / (hi - lo) * self.TRAJ_H
            pts += [x0 + i, max(1, min(self.TRAJ_H - 1, y))]
        if len(pts) >= 4:
            self.traj.create_line(*pts, fill="#86b6ef", width=2,
                                  capstyle="round", joinstyle="round")

    def on_frame(self, fp: FramePitch) -> None:
        """引擎回调(在工作线程里): 只投递数值, 不碰任何 GUI 对象。"""
        if fp.voiced and fp.midi_smooth is not None:
            self.q.put(("ui", (midi_to_note(fp.midi_smooth), fp.freq,
                               fp.midi_smooth, fp.power_db)))
        else:
            self.q.put(("ui", None))

    def on_close(self) -> None:
        self.worker.stop.set()
        try:
            self.worker.src.close()
        except Exception:
            pass
        self.worker.join(timeout=1.0)
        self.root.destroy()

    def start(self) -> None:
        self.worker.start()
        self.root.after(33, self.tick)


def main() -> None:
    ap = argparse.ArgumentParser(description="音高识别监视 GUI")
    ap.add_argument("infile", nargs="?", default=None)
    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="文件不按原速播放(默认按原速, 便于观察)")
    ap.add_argument("--lo", default="C3")
    ap.add_argument("--hi", default="B5")
    ap.add_argument("--hp", type=float, default=100.0)
    ap.add_argument("--f0-method", choices=("yin", "peak"), default="yin")
    args = ap.parse_args()

    if args.loopback:
        src = WasapiLoopbackSource(device=args.device)
        title = f"音高监视 — 回环 {src.device_name}"
    else:
        if not args.infile:
            ap.error("需要音频文件, 或用 --loopback")
        src = FileAudioSource(args.infile)
        title = f"音高监视 — {args.infile}"

    frame, hop = engine_frame_hop(src.sample_rate, 4096, 512)
    fe = PitchFrontend(sr=src.sample_rate, frame=frame, hop=hop,
                       range_midi=(parse_note(args.lo), parse_note(args.hi)),
                       f0_method=args.f0_method, hp_hz=args.hp)

    root = tk.Tk()
    app = App(root, src, fe, src.sample_rate, pace=not args.fast, title=title)
    fe.on_event = lambda e: (app.on_frame(e)
                             if isinstance(e, FramePitch) else None)
    app.start()
    root.mainloop()
    src.close()


if __name__ == "__main__":
    main()
