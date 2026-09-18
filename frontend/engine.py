"""实时流式音高清码前端(解码器第一级)。

推模式设计: 音频块进来 -> FramePitch/NoteEvent 事件出去, 全部处理只用因果
信息(不依赖未来整段数据), 延迟有界, 可对接任意 AudioSource, 供后续解码模块
通过 on_event 回调消费。

DSP 与已验证的离线脚本 pitch_track.py 同源: 4096 点 Hann 窗 / 512 点帧移 /
4 倍零填充 STFT, MIDI 域 7 帧中值 + 9 帧 2 阶局部拟合平滑
(Savitzky-Golay 等价), 按半音量化合并出音符事件。

基频估计默认用自相关(YIN), 见 f0_method。原"带内最高谱峰"法(f0_method=
"peak")在簧片乐器上会系统性出错: 口琴基频常比 3 次谐波弱十几 dB(实测
test2.mp3 的 C3 基频比最强峰低 17.1dB), "最强分量即基频"的前提不成立,
输出会整体变成 3 次谐波(八度+五度的错误)。自相关只看波形的周期重复性,
与谐波幅度分布无关, 对弱基频免疫。

乐器音域 C3~B5(range_midi): 搜索带默认由此推导(各向外扩 1 半音), 带外
频率既不出音也不成音符; 对 YIN 而言音域同时构成自相关 lag 的搜索范围,
比基频更低的八度下错候选天然出局。

与离线脚本的两处差异(流式化必然结果, 参数已对齐回归样本):
- 静音门限不再用全文件 Otsu, 改为滑动最小统计估计噪声底 + SNR 偏置(因果);
  也可用 silence_db 钉死固定门限。
- 平滑/分段需要少量前瞻帧, 事件比帧中心晚约 (窗长 + 前瞻 + 确认帧 + 最短
  音长), 数量级 0.2~0.3s, 时间戳本身无偏。

用法:
    fe = PitchFrontend(sr=48000, on_event=decoder.consume)
    for chunk in audio_source:      # FileAudioSource 或将来的 WasapiLoopbackSource
        fe.process(chunk)
    fe.flush()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def midi_to_note(m: float) -> str:
    n = int(round(m))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


@dataclass
class FramePitch:
    """逐帧音高事件。"""

    t: float                       # 帧中心时间(音频时间轴, 秒)
    power_db: float                # 带内谱峰幅度(dB)
    voiced: bool                   # 通过能量/噪音门限
    freq: Optional[float]          # 原始峰频(Hz); 未出音为 None
    midi: Optional[float]          # 原始峰频 MIDI
    midi_smooth: Optional[float]   # 平滑后 MIDI(前瞻帧就绪后产生)


@dataclass
class NoteEvent:
    """音符级事件, 解码模块的基本输入。"""

    kind: str    # "on" / "off"
    t: float
    midi: int
    note: str


Event = Union[FramePitch, NoteEvent]


def _parabolic_refine(spec: np.ndarray, k: int) -> float:
    """对 log 幅度谱做三点抛物线插值, 返回亚 bin 峰位偏移。"""
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-12)
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def _yin_f0(seg: np.ndarray, sr: int, tau_min: int, tau_max: int,
            nfft: int, threshold: float = 0.15,
            octave_check: bool = True) -> float:
    """YIN 基频估计(自相关/累积均值归一化差分), 返回 Hz 或 nan。

    谱峰法假设"最强分量是基频", 而口琴这类簧片乐器基频常比 3 次谐波弱
    十几 dB, 该假设不成立。自相关只看波形的周期重复性, 与谐波幅度分布
    无关, 因此对弱基频免疫。

    tau_min/tau_max 由乐器音域反推: 基频搜索被限制在音域内, 比基频更低
    的候选(八度下错)天然出局。

    返回 nan 表示该帧没有明确的周期结构(噪声/静音/瞬态)。这个"弃权"是
    有信息量的: 实测 test.mp3 的强瞬态段(能量最大但频谱散乱)严格 YIN 会
    弃权, 而谱峰法给出的值是对的; 若强行退回 CMND 全局最小, 反而会输出
    低八度的错误值。故调用方在 nan 时应退回谱峰法, 而不是继续猜周期。
    """
    n = len(seg)
    if tau_max <= tau_min + 2:
        return np.nan
    seg = seg - seg.mean()
    X = np.fft.rfft(seg, nfft)
    acf = np.fft.irfft(X * np.conj(X), nfft)[: tau_max + 1]
    c1 = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    e_head = c1[n - np.arange(tau_max + 1)]
    e_tail = c1[n] - c1[np.arange(tau_max + 1)]
    d = np.maximum(e_head + e_tail - 2.0 * acf[: tau_max + 1], 0.0)
    d[0] = 0.0
    cmnd = np.ones(tau_max + 1)
    taus = np.arange(1, tau_max + 1)
    cmnd[1:] = d[1:] * taus / (np.cumsum(d[1:]) + 1e-12)

    # YIN 第 4 步: 第一个低于阈值的局部极小(取最短 lag 也顺带压制八度下错);
    # 找不到 -> 弃权, 由调用方退回谱峰法
    best = None
    for tau in range(tau_min, tau_max):
        if cmnd[tau] < threshold and cmnd[tau] <= cmnd[tau + 1]:
            best = tau
            break
    if best is None:
        return np.nan
    if octave_check:
        # 半个/三分之一周期处若谷底同样深, 则真实周期更短(防止八度下错)
        for div in (2, 3):
            cand = int(round(best / div))
            if cand >= tau_min and cmnd[cand] < min(0.35, cmnd[best] * 2.5):
                best = cand
                break
    a, b, c = cmnd[best - 1], cmnd[best], cmnd[best + 1]
    den = a - 2.0 * b + c
    off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    return sr / (best + float(np.clip(off, -0.5, 0.5)))


class _HighPass:
    """因果二阶高通(RBJ/Butterworth), 跨音频块保持状态。

    乐器最低音是 C3(130.8Hz), 音域以下只可能是噪声: 空调/桌面震动/电源
    哼声/近场气流。这些低频能量对 YIN 是实打实的干扰——自相关吃的是时域
    波形, tau 搜索范围虽已限制在音域内, 但低频分量照样进 d(tau) 和 CMND,
    把周期谷底抬高。谱峰路径不受影响(搜索带本就 ≥123Hz)。

    实测(向 test2.mp3 注入 38/57Hz 嗡声, 与干净结果的逐帧一致率):
        注入电平   0dB   -6dB  -12dB  -18dB  -24dB  -30dB
        原始      20.8%  63.6%  81.4%  91.7%  96.0%  99.3%
        本滤波器  85.2%  92.8%  97.3%  98.9%  99.0%  98.8%
    即存在低频污染时收益很大, 干净素材上持平。

    截止取 100Hz 而非贴着 123Hz: C3(131.8Hz) 处只衰减约 1.2dB, 不影响
    判音; 而典型工频/震动噪声(38~57Hz)可压下 10~17dB。
    """

    def __init__(self, fc: float, sr: int, stages: int = 1, q: float = 0.7071):
        w0 = 2.0 * np.pi * fc / sr
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2.0 * q)
        b0, b1, b2 = (1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0
        a0, a1, a2 = 1.0 + alpha, -2.0 * cw, 1.0 - alpha
        self.b = (b0 / a0, b1 / a0, b2 / a0)
        self.a = (a1 / a0, a2 / a0)
        self.stages = stages
        self._z = np.zeros((stages, 2))

    def process(self, x: np.ndarray) -> np.ndarray:
        """滤波一块样本, 状态延续到下次调用(块边界无接缝)。

        二阶 IIR 的反馈是串行的, 只能逐样本递推; 单级情形单独写以减少
        Python 解释开销(这是整个引擎里唯一无法向量化的地方)。
        """
        b0, b1, b2 = self.b
        a1, a2 = self.a
        out = np.empty(len(x), dtype=np.float64)
        if self.stages == 1:
            z1, z2 = self._z[0]
            for i in range(len(x)):
                xi = x[i]
                yi = b0 * xi + z1
                z1 = b1 * xi - a1 * yi + z2
                z2 = b2 * xi - a2 * yi
                out[i] = yi
            self._z[0] = (z1, z2)
            return out
        for s in range(self.stages):
            z1, z2 = self._z[s]
            for i in range(len(x)):
                xi = x[i]
                yi = b0 * xi + z1
                z1 = b1 * xi - a1 * yi + z2
                z2 = b2 * xi - a2 * yi
                out[i] = yi
            self._z[s] = (z1, z2)
            x = out.copy()
        return out


class _MinStatsGate:
    """因果静音门限: 滑动窗口最小统计估计噪声底, 加 SNR 偏置。

    游戏音频里音符(24~40dB)与间隙噪声(<=10dB)动态范围大且随音量浮动,
    固定门限不鲁棒; 最小统计只依赖过去窗口, 流式可用。
    """

    def __init__(self, snr_db: float = 10.0, window_s: float = 6.0,
                 floor_db: float = -80.0, hop_s: float = 0.0107):
        self.snr = snr_db
        self.w = max(1, int(round(window_s / hop_s)))
        self.floor = floor_db
        self._hist: List[float] = []

    def threshold(self, power_db: float) -> float:
        self._hist.append(power_db)
        lo = max(0, len(self._hist) - self.w)
        noise = min(self._hist[lo:])
        return max(noise + self.snr, self.floor)


class _NoteSegmenter:
    """平滑 MIDI 流 -> 音符 on/off 事件。

    - 半音量化, 相邻同音合并;
    - 音高变化需连续 change_confirm 帧确认, 静音需 rest_confirm 帧确认;
    - on 事件延迟到音符存续满 min_note_s 才发出(回填真实起始时间戳),
      过短的音符整体丢弃。
    """

    def __init__(self, t_of_frame: Callable[[int], float], min_note_s: float,
                 change_confirm: int, rest_confirm: int,
                 midi_lo: int = 0, midi_hi: int = 127):
        self._t = t_of_frame
        self.min_note_s = min_note_s
        self.change_confirm = change_confirm
        self.rest_confirm = rest_confirm
        self.midi_lo, self.midi_hi = midi_lo, midi_hi
        self.cur_q: Optional[int] = None
        self.cur_start = 0
        self.cur_last = 0
        self.announced = False
        self.cand_q: Optional[int] = None
        self.cand_start = 0
        self.cand_n = 0
        self.rest = 0
        self.last_voiced = -1

    def _events_for_close(self, off_i: int) -> List[NoteEvent]:
        ev, q, s = [], self.cur_q, self.cur_start
        dur = self._t(off_i) - self._t(s)
        if self.announced:
            ev.append(NoteEvent("off", self._t(off_i), q, midi_to_note(q)))
        elif dur >= self.min_note_s:
            ev.append(NoteEvent("on", self._t(s), q, midi_to_note(q)))
            ev.append(NoteEvent("off", self._t(off_i), q, midi_to_note(q)))
        self.cur_q = None
        self.announced = False
        return ev

    def feed(self, i: int, midi_smooth: Optional[float]) -> List[NoteEvent]:
        ev: List[NoteEvent] = []
        # 音域保险: 平滑值落在乐器音域外(滑音过冲等)按无声处理
        if midi_smooth is not None and not (
                self.midi_lo <= round(midi_smooth) <= self.midi_hi):
            midi_smooth = None
        if midi_smooth is None:
            self.rest += 1
            self.cand_q = None
            if self.rest >= self.rest_confirm and self.cur_q is not None:
                ev += self._events_for_close(off_i=i - self.rest)
            return ev
        self.rest = 0
        self.last_voiced = i
        q = int(round(midi_smooth))
        if self.cur_q is None:
            self.cur_q, self.cur_start, self.cur_last = q, i, i
        elif q == self.cur_q:
            self.cur_last = i
            self.cand_q = None
        else:
            if self.cand_q == q:
                self.cand_n += 1
            else:
                self.cand_q, self.cand_start, self.cand_n = q, i, 1
            if self.cand_n >= self.change_confirm:
                off = self.cand_start
                ev += self._events_for_close(off_i=off)
                self.cur_q, self.cur_start, self.cur_last = q, self.cand_start, i
        # 存续满最短音长后回填发出 on
        if (self.cur_q is not None and not self.announced
                and self._t(i) - self._t(self.cur_start) >= self.min_note_s):
            ev.append(NoteEvent("on", self._t(self.cur_start),
                                self.cur_q, midi_to_note(self.cur_q)))
            self.announced = True
        return ev

    def flush(self) -> List[NoteEvent]:
        ev: List[NoteEvent] = []
        if self.cur_q is not None:
            ev += self._events_for_close(off_i=self.last_voiced + 1)
        return ev


class PitchFrontend:
    """流式音高/音符前端。

    参数默认值按 48kHz 调好; 换采样率时 frame/hop 等样本数参数需按比例调。
    process/flush 返回本轮新产生的事件(FramePitch 与 NoteEvent 混合, 按产生
    顺序); on_event 回调同时被逐个调用, 解码模块挂这里。
    """

    def __init__(self, sr: int, frame: int = 4096, hop: int = 512, zpad: int = 4,
                 fmin: Optional[float] = None, fmax: Optional[float] = None,
                 range_midi: tuple = (48, 83),
                 median: int = 7, smooth: int = 9,
                 harm_thr: float = 0.030, harm_tol: float = 0.025,
                 subharm: bool = False, f0_method: str = "yin",
                 yin_threshold: float = 0.15,
                 hp_hz: float = 100.0, hp_stages: int = 1,
                 silence_db: Optional[float] = None,
                 gate_snr_db: float = 10.0, gate_window_s: float = 6.0,
                 min_note_ms: float = 100.0,
                 change_confirm: int = 3, rest_confirm: int = 3,
                 on_event: Optional[Callable[[Event], None]] = None):
        if median % 2 == 0:
            median += 1
        if smooth % 2 == 0:
            smooth += 1
        self.sr, self.frame, self.hop = sr, frame, hop
        self.median, self.smooth = median, smooth
        self.lookahead = median // 2 + smooth // 2
        self.harm_thr, self.harm_tol, self.subharm = harm_thr, harm_tol, subharm
        # 乐器音域(C3~B5): 搜索带默认取音域各向外扩 1 个半音, 吸收音准偏移
        # 和插值窗口; 带外谐波不参与"最高峰"竞争, 从源头避免八度错误
        self.midi_lo, self.midi_hi = range_midi
        if fmin is None:
            fmin = midi_to_hz(self.midi_lo - 1.0)
        if fmax is None:
            fmax = midi_to_hz(self.midi_hi + 1.0)
        self.fmax = fmax
        self.n_fft = frame * zpad
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sr)
        band = np.where((self.freqs >= fmin) & (self.freqs <= fmax))[0]
        self.f_lo, self.f_hi = int(band[0]), int(band[-1])
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.win = np.hanning(frame)
        # 自相关基频搜索范围 = 乐器音域(比基频更低的八度下错候选天然出局)
        self.f0_method = f0_method
        self.yin_threshold = yin_threshold
        self.tau_min = max(2, int(sr / fmax))
        self.tau_max = min(frame // 2, int(sr / fmin))
        self.acf_nfft = 1 << int(np.ceil(np.log2(2 * frame)))
        self.hp = _HighPass(hp_hz, sr, hp_stages) if hp_hz > 0 else None
        self.on_event = on_event
        self.silence_db = silence_db
        self.gate = None if silence_db is not None else _MinStatsGate(
            snr_db=gate_snr_db, window_s=gate_window_s,
            hop_s=hop / sr)
        self.seg = _NoteSegmenter(self._t_of, min_note_ms / 1000.0,
                                  change_confirm, rest_confirm,
                                  midi_lo=self.midi_lo, midi_hi=self.midi_hi)
        self._buf = np.zeros(0)
        self._g = 0                 # 已分析帧数
        self._raw: List[float] = []      # 原始 MIDI(nan=未出音)
        self._rawf: List[Optional[float]] = []
        self._voiced: List[bool] = []
        self._power: List[float] = []
        self._next_out = 0          # 下一个待输出(平滑)帧索引

    # ---- 时间轴 ----
    def _t_of(self, i: int) -> float:
        return (i * self.hop + self.frame / 2.0) / self.sr

    # ---- 单帧分析 ----
    def _peak_f0(self, spec: np.ndarray):
        """旧路径: 带内最高谱峰(可用 subharm 做子谐波校正)。"""
        b = spec[self.f_lo : self.f_hi + 1]
        loc = np.where((b[1:-1] > b[:-2]) & (b[1:-1] >= b[2:]))[0] + 1
        if not len(loc):
            return None
        vmax = b[loc].max()
        cands = loc[b[loc] >= vmax * 10.0 ** (-10.0 / 20.0)]
        f_cands = self.freqs[self.f_lo + cands]
        mag_cands = b[cands]
        k_max = self.f_lo + cands[int(np.argmax(mag_cands))]
        best_k = k_max
        if self.subharm:
            best_score, best_mag = 0, -np.inf
            for jc, fc in enumerate(f_cands):
                ks = np.round(f_cands / fc).astype(int)
                ok = (ks >= 1) & (
                    np.abs(f_cands - ks * fc) <= self.harm_tol * f_cands)
                score = int(ok.sum())
                if score > best_score or (score == best_score
                                          and mag_cands[jc] > best_mag):
                    best_k, best_score, best_mag = (self.f_lo + cands[jc],
                                                    score, mag_cands[jc])
        delta = (_parabolic_refine(spec, best_k)
                 if self.f_lo < best_k < self.f_hi else 0.0)
        return float(self.freqs[best_k] + delta * (self.freqs[1] - self.freqs[0]))

    def _harm_ratio(self, spec: np.ndarray, b: np.ndarray, f0: float) -> float:
        """能量落在 f0 谐波列上的占比(噪音门限用)。

        该指标对高音结构性不利: 音域上端的音符带内只剩 1~2 根谐波线, 分母却是
        整带 ~340 根 bin 的线性和, 比值必然被稀释。实测(实吹录音 t1)环里最高的
        C5 该指标中位 0.0469, 其他音 0.0637 —— **系统性吃亏 27%**, 正好压在门限上,
        整个音被砍掉, 表现为"整个音符消失"(差分链断, 一个音毁两个数字)。

        harm_thr 因此从 0.045 降到 0.030: 两段实吹录音从解码出错变为全对, 且
        静音段不引入任何多余音符。更根本的改法是分母改用噪声底归一(实测可让
        C5 由其他音的 73% 变为 109%, 偏置消失), 但需要重新标定门限与全量回归,
        留作后续。
        """
        harm_sum = 0.0
        for k in range(1, 9):
            fk = f0 * k
            if fk > self.fmax:
                break
            w = ((self.freqs >= fk * (1 - self.harm_tol))
                 & (self.freqs <= fk * (1 + self.harm_tol)))
            if w.any():
                harm_sum += spec[w].max()
        return harm_sum / (b.sum() + 1e-12)

    def _analyze(self, seg: np.ndarray) -> None:
        spec = np.abs(np.fft.rfft(seg * self.win, n=self.n_fft))
        b = spec[self.f_lo : self.f_hi + 1]
        power = 20.0 * np.log10(float(b.max()) + 1e-12)
        freq = None
        voiced = False
        if self.silence_db is not None:
            thr = self.silence_db
        else:
            thr = self.gate.threshold(power)
        if power >= thr:
            if self.f0_method == "yin":
                f0 = _yin_f0(seg, self.sr, self.tau_min, self.tau_max,
                             self.acf_nfft, self.yin_threshold)
                if f0 is not None and not np.isnan(f0) \
                        and self.fmin <= f0 <= self.fmax:
                    freq = float(f0)
                # YIN 弃权(瞬态/噪声)时退回谱峰法: 此时"最强分量即基频"
                # 的假设通常成立, 沿用旧行为可避免瞬态段出现低八度错误
                if freq is None:
                    freq = self._peak_f0(spec)
            else:
                freq = self._peak_f0(spec)
            if freq is not None:
                # 噪音门限: 能量集中在基频谐波列上的占比(见 _harm_ratio)
                if self.harm_thr > 0:
                    voiced = self._harm_ratio(spec, b, freq) >= self.harm_thr
                else:
                    voiced = True
                if not voiced:
                    freq = None
        self._power.append(power)
        self._voiced.append(voiced)
        self._rawf.append(freq)
        self._raw.append(hz_to_midi(freq) if freq is not None else np.nan)
        self._g += 1

    # ---- 流式平滑(与离线"段内中值+SG"等价的局部实现) ----
    def _run_bounds(self, i: int, half: int) -> tuple[int, int]:
        lo = i
        while lo > 0 and self._voiced[lo - 1] and i - lo < half:
            lo -= 1
        hi = i
        while hi < self._g - 1 and self._voiced[hi + 1] and hi - i < half:
            hi += 1
        return lo, hi

    def _median_at(self, i: int) -> float:
        lo, hi = self._run_bounds(i, self.median // 2)
        vals = [v for v in self._raw[lo : hi + 1] if not np.isnan(v)]
        return float(np.median(vals)) if vals else np.nan

    def _smooth_at(self, i: int) -> Optional[float]:
        lo, hi = self._run_bounds(i, self.smooth // 2)
        xs, ys = [], []
        for j in range(lo, hi + 1):
            m = self._median_at(j)
            if not np.isnan(m):
                xs.append(j - i)
                ys.append(m)
        if len(xs) >= 3:
            coef = np.polyfit(xs, ys, 2)
            return float(np.polyval(coef, 0.0))
        m0 = self._median_at(i)
        return None if np.isnan(m0) else float(m0)

    def _drain(self, final: bool = False) -> List[Event]:
        ev: List[Event] = []
        limit = self._g if final else self._g - self.lookahead
        while self._next_out < limit:
            i = self._next_out
            ms = self._smooth_at(i) if self._voiced[i] else None
            fp = FramePitch(
                t=self._t_of(i), power_db=self._power[i],
                voiced=self._voiced[i], freq=self._rawf[i],
                midi=self._raw[i] if self._voiced[i] else None,
                midi_smooth=ms)
            ev.append(fp)
            ev += self.seg.feed(i, ms)
            self._next_out += 1
        return ev

    # ---- 对外接口 ----
    def process(self, chunk: np.ndarray) -> List[Event]:
        chunk = np.asarray(chunk, np.float64)
        if self.hp is not None:
            # 音域以下的高通, 状态跨块保持(见 _HighPass)
            chunk = self.hp.process(chunk)
        self._buf = np.concatenate([self._buf, chunk])
        while len(self._buf) >= self.frame:
            self._analyze(self._buf[: self.frame])
            self._buf = self._buf[self.hop :]
        ev = self._drain()
        for e in ev:
            if self.on_event:
                self.on_event(e)
        return ev

    def flush(self) -> List[Event]:
        """流结束: 丢掉不满一窗的尾部样本, 收尾平滑/分段, 返回剩余事件。"""
        ev = self._drain(final=True)
        ev += self.seg.flush()
        for e in ev:
            if self.on_event:
                self.on_event(e)
        return ev
