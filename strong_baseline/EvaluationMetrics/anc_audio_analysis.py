# -*- coding: utf-8 -*-
"""
ANC音频频谱与1/3倍频程分析程序

功能：
1. 读取WAV音频；
2. 提取ANC关闭和ANC开启时间段；
3. 使用STFT计算平均功率谱；
4. 转换为1/3倍频程频带级；
5. 计算平均降噪量、最大噪声反弹和最大降噪量；
6. 绘制并保存1/3倍频程降噪曲线。

降噪量定义：
    NR = L_off - L_on

因此：
    NR > 0：ANC开启后噪声降低；
    NR < 0：ANC开启后噪声增加，即出现噪声反弹。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class SpectrumResult:
    """STFT平均功率谱结果。"""

    frequencies: np.ndarray
    power_spectrum: np.ndarray
    level_db: np.ndarray
    sample_rate: int


@dataclass
class ThirdOctaveResult:
    """1/3倍频程分析结果。"""

    center_frequencies: np.ndarray
    lower_frequencies: np.ndarray
    upper_frequencies: np.ndarray
    band_power: np.ndarray
    band_level_db: np.ndarray
    valid_mask: np.ndarray


class AudioSpectrumAnalyzer:
    """完成音频加载、分段和STFT平均功率谱计算。"""

    def __init__(
        self,
        n_fft: int = 8192,
        hop_length: Optional[int] = None,
        window_type: str = "hann",
        eps: float = 1e-20,
    ) -> None:
        if not isinstance(n_fft, int) or n_fft <= 0:
            raise ValueError("n_fft必须为正整数。")

        if hop_length is None:
            hop_length = n_fft // 4

        if not isinstance(hop_length, int) or hop_length <= 0:
            raise ValueError("hop_length必须为正整数。")

        if hop_length > n_fft:
            raise ValueError("hop_length不应大于n_fft。")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_type = window_type.lower()
        self.eps = float(eps)

    def _create_window(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """根据设置创建窗函数。"""
        if self.window_type == "hann":
            return torch.hann_window(
                self.n_fft,
                periodic=True,
                dtype=dtype,
                device=device,
            )

        if self.window_type == "hamming":
            return torch.hamming_window(
                self.n_fft,
                periodic=True,
                dtype=dtype,
                device=device,
            )

        if self.window_type == "blackman":
            return torch.blackman_window(
                self.n_fft,
                periodic=True,
                dtype=dtype,
                device=device,
            )

        raise ValueError(
            f"不支持的窗函数：{self.window_type}。"
            "可选值为hann、hamming或blackman。"
        )

    @staticmethod
    def load_audio(
        file_path: Union[str, Path],
        channel: Optional[int] = None,
        mono: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        读取音频。

        Parameters
        ----------
        file_path:
            WAV音频路径。
        channel:
            指定分析的通道编号，从0开始。指定后mono参数不再生效。
        mono:
            对多通道音频取平均，默认True。

        Returns
        -------
        waveform:
            形状为[1, samples]或[channel, samples]。
        sample_rate:
            采样率。
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"找不到音频文件：{file_path}")

        sample_rate, waveform_np = wavfile.read(str(file_path))
        # 转换为[channel, samples]形状的float张量
        if waveform_np.ndim == 1:
            waveform_np = waveform_np.reshape(1, -1)
        else:
            waveform_np = waveform_np.T  # [samples, channel] -> [channel, samples]
        
        # 归一化到[-1, 1]
        if np.issubdtype(waveform_np.dtype, np.integer):
            max_val = float(np.iinfo(waveform_np.dtype).max)
            waveform_np = waveform_np.astype(np.float64) / max_val
        else:
            waveform_np = waveform_np.astype(np.float64)
        
        waveform = torch.from_numpy(waveform_np)

        if channel is not None:
            if channel < 0 or channel >= waveform.shape[0]:
                raise ValueError(
                    f"通道编号{channel}无效。该音频共有"
                    f"{waveform.shape[0]}个通道。"
                )
            waveform = waveform[channel : channel + 1, :]
        elif mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, int(sample_rate)

    @staticmethod
    def extract_segment(
        waveform: torch.Tensor,
        sample_rate: int,
        time_range: Tuple[float, float],
    ) -> torch.Tensor:
        """按时间范围提取音频片段。"""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 2:
            raise ValueError(
                "waveform形状必须为[samples]或[channel, samples]。"
            )

        start_time, end_time = time_range
        duration = waveform.shape[-1] / sample_rate

        if start_time < 0:
            raise ValueError("起始时间不能小于0。")

        if end_time <= start_time:
            raise ValueError("结束时间必须大于起始时间。")

        if end_time > duration:
            raise ValueError(
                f"时间范围{time_range}超过音频总时长"
                f"{duration:.3f}秒。"
            )

        start_sample = int(round(start_time * sample_rate))
        end_sample = int(round(end_time * sample_rate))

        segment = waveform[:, start_sample:end_sample]

        if segment.numel() == 0:
            raise ValueError("提取出的音频片段为空。")

        return segment

    def compute_power_spectrum(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        remove_mean: bool = True,
        reference_power: float = 1.0,
    ) -> SpectrumResult:
        """
        使用STFT计算时间平均单边功率谱。

        说明：
        1. 先计算各帧FFT模平方；
        2. 对时间帧和通道求平均；
        3. 使用窗函数平方和进行能量归一化；
        4. 对单边谱进行能量补偿。
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate必须为正数。")

        if reference_power <= 0:
            raise ValueError("reference_power必须为正数。")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 2:
            raise ValueError(
                "waveform形状必须为[samples]或[channel, samples]。"
            )

        waveform = waveform.to(torch.float64)

        if remove_mean:
            waveform = waveform - waveform.mean(dim=-1, keepdim=True)

        if waveform.shape[-1] < self.n_fft:
            pad_length = self.n_fft - waveform.shape[-1]
            waveform = torch.nn.functional.pad(
                waveform,
                (0, pad_length),
            )

        window = self._create_window(
            device=waveform.device,
            dtype=waveform.dtype,
        )

        stft_result = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        power = stft_result.abs().square()
        mean_power = power.mean(dim=(0, 2))

        # 窗能量归一化。
        mean_power = mean_power / window.square().sum()

        # 单边谱补偿。
        if self.n_fft % 2 == 0:
            if mean_power.numel() > 2:
                mean_power[1:-1] *= 2.0
        else:
            if mean_power.numel() > 1:
                mean_power[1:] *= 2.0

        frequencies = torch.fft.rfftfreq(
            self.n_fft,
            d=1.0 / sample_rate,
            device=waveform.device,
        )

        level_db = 10.0 * torch.log10(
            torch.clamp(
                mean_power / reference_power,
                min=self.eps,
            )
        )

        return SpectrumResult(
            frequencies=frequencies.cpu().numpy(),
            power_spectrum=mean_power.cpu().numpy(),
            level_db=level_db.cpu().numpy(),
            sample_rate=sample_rate,
        )


class ThirdOctaveAnalyzer:
    """将FFT线性功率谱转换为1/3倍频程频谱。"""

    NOMINAL_CENTER_FREQUENCIES = np.array(
        [
            12.5,
            16.0,
            20.0,
            25.0,
            31.5,
            40.0,
            50.0,
            63.0,
            80.0,
            100.0,
            125.0,
            160.0,
            200.0,
            250.0,
            315.0,
            400.0,
            500.0,
            630.0,
            800.0,
            1000.0,
            1250.0,
            1600.0,
            2000.0,
            2500.0,
            3150.0,
            4000.0,
            5000.0,
            6300.0,
            8000.0,
            10000.0,
            12500.0,
            16000.0,
            20000.0,
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        center_frequencies: Optional[Sequence[float]] = None,
        eps: float = 1e-30,
    ) -> None:
        if center_frequencies is None:
            center_frequencies = self.NOMINAL_CENTER_FREQUENCIES

        self.center_frequencies = np.asarray(
            center_frequencies,
            dtype=np.float64,
        )

        if self.center_frequencies.ndim != 1:
            raise ValueError("中心频率必须是一维数组。")

        if np.any(self.center_frequencies <= 0):
            raise ValueError("中心频率必须全部大于0。")

        if np.any(np.diff(self.center_frequencies) <= 0):
            raise ValueError("中心频率必须按升序排列。")

        self.eps = float(eps)

    def fft_to_third_octave(
        self,
        frequencies: ArrayLike,
        power_spectrum: ArrayLike,
        sample_rate: Optional[int] = None,
        reference_power: float = 1.0,
    ) -> ThirdOctaveResult:
        """
        将FFT线性功率谱转换为1/3倍频程。

        注意：
        power_spectrum必须是线性功率，不能传入dB频谱。
        """
        frequencies = np.asarray(frequencies, dtype=np.float64)
        power_spectrum = np.asarray(power_spectrum, dtype=np.float64)

        if frequencies.ndim != 1 or power_spectrum.ndim != 1:
            raise ValueError("频率和功率谱必须是一维数组。")

        if frequencies.shape != power_spectrum.shape:
            raise ValueError("频率和功率谱长度必须一致。")

        if not np.all(np.isfinite(frequencies)):
            raise ValueError("频率数组包含非有限值。")

        if np.any(power_spectrum < 0):
            raise ValueError("功率谱不能包含负数。")

        if reference_power <= 0:
            raise ValueError("reference_power必须为正数。")

        if sample_rate is None:
            nyquist_frequency = float(np.max(frequencies))
        else:
            nyquist_frequency = sample_rate / 2.0

        edge_factor = 2.0 ** (1.0 / 6.0)
        lower_frequencies = self.center_frequencies / edge_factor
        upper_frequencies = self.center_frequencies * edge_factor

        band_power = np.full(
            self.center_frequencies.shape,
            np.nan,
            dtype=np.float64,
        )

        # 只有整个频带不超过Nyquist频率时才视为完整有效频带。
        valid_mask = upper_frequencies <= nyquist_frequency

        for index, (lower, upper) in enumerate(
            zip(lower_frequencies, upper_frequencies)
        ):
            if not valid_mask[index]:
                continue

            frequency_mask = (
                (frequencies >= lower)
                & (frequencies < upper)
            )

            if not np.any(frequency_mask):
                continue

            band_power[index] = np.sum(
                power_spectrum[frequency_mask]
            )

        finite_mask = np.isfinite(band_power)
        band_level_db = np.full_like(band_power, np.nan)

        band_level_db[finite_mask] = 10.0 * np.log10(
            np.maximum(
                band_power[finite_mask] / reference_power,
                self.eps,
            )
        )

        return ThirdOctaveResult(
            center_frequencies=self.center_frequencies.copy(),
            lower_frequencies=lower_frequencies,
            upper_frequencies=upper_frequencies,
            band_power=band_power,
            band_level_db=band_level_db,
            valid_mask=valid_mask & finite_mask,
        )

    @staticmethod
    def _range_mask(
        center_frequencies: np.ndarray,
        frequency_range: Tuple[float, float],
    ) -> np.ndarray:
        lower, upper = frequency_range

        if lower <= 0 or upper <= 0:
            raise ValueError("频率范围必须大于0。")

        if lower >= upper:
            raise ValueError("频率范围下限必须小于上限。")

        return (
            (center_frequencies >= lower)
            & (center_frequencies <= upper)
        )

    @staticmethod
    def calculate_noise_reduction(
        off_result: ThirdOctaveResult,
        on_result: ThirdOctaveResult,
    ) -> Dict[str, np.ndarray]:
        """
        计算ANC开启前后的1/3倍频程级差。

        noise_reduction_db = L_off - L_on
        level_change_db = L_on - L_off
        """
        if not np.allclose(
            off_result.center_frequencies,
            on_result.center_frequencies,
        ):
            raise ValueError("两组倍频程结果的中心频率不一致。")

        valid_mask = (
            off_result.valid_mask
            & on_result.valid_mask
        )

        noise_reduction_db = (
            off_result.band_level_db
            - on_result.band_level_db
        )
        level_change_db = -noise_reduction_db

        valid_mask &= np.isfinite(noise_reduction_db)

        return {
            "center_frequencies":
                off_result.center_frequencies.copy(),
            "noise_reduction_db":
                noise_reduction_db,
            "level_change_db":
                level_change_db,
            "valid_mask":
                valid_mask,
        }

    def calculate_metrics(
        self,
        off_result: ThirdOctaveResult,
        on_result: ThirdOctaveResult,
        average_range: Tuple[float, float] = (50.0, 5000.0),
        rebound_range: Tuple[float, float] = (1000.0, 8000.0),
    ) -> Dict[str, float]:
        """
        计算评价指标。

        返回：
        1. 频带降噪量的算术平均值；
        2. 整个指定频段的总能量降噪量；
        3. 最大噪声反弹；
        4. 最差降噪量及对应频率；
        5. 最大降噪量及对应频率。
        """
        reduction = self.calculate_noise_reduction(
            off_result,
            on_result,
        )

        center_frequencies = reduction["center_frequencies"]
        noise_reduction_db = reduction["noise_reduction_db"]
        valid_mask = reduction["valid_mask"]

        average_mask = (
            self._range_mask(
                center_frequencies,
                average_range,
            )
            & valid_mask
        )

        rebound_mask = (
            self._range_mask(
                center_frequencies,
                rebound_range,
            )
            & valid_mask
        )

        if not np.any(average_mask):
            raise ValueError(
                f"{average_range} Hz内没有有效的1/3倍频程数据。"
            )

        if not np.any(rebound_mask):
            raise ValueError(
                f"{rebound_range} Hz内没有有效的1/3倍频程数据。"
            )

        # 指标1：各频带降噪量的算术平均。
        average_noise_reduction = float(
            np.mean(noise_reduction_db[average_mask])
        )

        # 指标2：整个频段总能量的降噪量。
        total_off_power = float(
            np.sum(off_result.band_power[average_mask])
        )
        total_on_power = float(
            np.sum(on_result.band_power[average_mask])
        )

        total_band_noise_reduction = float(
            10.0
            * np.log10(
                max(total_off_power, self.eps)
                / max(total_on_power, self.eps)
            )
        )

        frequencies_in_rebound_range = (
            center_frequencies[rebound_mask]
        )
        reductions_in_rebound_range = (
            noise_reduction_db[rebound_mask]
        )

        worst_index = int(
            np.argmin(reductions_in_rebound_range)
        )
        best_index = int(
            np.argmax(reductions_in_rebound_range)
        )

        worst_noise_reduction = float(
            reductions_in_rebound_range[worst_index]
        )
        maximum_noise_reduction = float(
            reductions_in_rebound_range[best_index]
        )

        maximum_level_increase = -worst_noise_reduction
        maximum_rebound = max(0.0, maximum_level_increase)

        return {
            "average_noise_reduction_db":
                average_noise_reduction,
            "total_band_noise_reduction_db":
                total_band_noise_reduction,
            "maximum_rebound_db":
                maximum_rebound,
            "maximum_level_change_db":
                maximum_level_increase,
            "worst_noise_reduction_db":
                worst_noise_reduction,
            "rebound_frequency_hz":
                float(
                    frequencies_in_rebound_range[worst_index]
                ),
            "maximum_noise_reduction_db":
                maximum_noise_reduction,
            "maximum_reduction_frequency_hz":
                float(
                    frequencies_in_rebound_range[best_index]
                ),
        }


def plot_noise_reduction_curve(
    center_frequencies: ArrayLike,
    noise_reduction_db: ArrayLike,
    valid_mask: Optional[ArrayLike] = None,
    average_range: Tuple[float, float] = (50.0, 5000.0),
    rebound_range: Tuple[float, float] = (1000.0, 8000.0),
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """绘制1/3倍频程降噪曲线。"""
    center_frequencies = np.asarray(
        center_frequencies,
        dtype=np.float64,
    )
    noise_reduction_db = np.asarray(
        noise_reduction_db,
        dtype=np.float64,
    )

    if center_frequencies.shape != noise_reduction_db.shape:
        raise ValueError("中心频率与降噪量数组长度不一致。")

    if valid_mask is None:
        valid_mask = np.isfinite(noise_reduction_db)
    else:
        valid_mask = (
            np.asarray(valid_mask, dtype=bool)
            & np.isfinite(noise_reduction_db)
        )

    frequencies = center_frequencies[valid_mask]
    reductions = noise_reduction_db[valid_mask]

    if frequencies.size == 0:
        raise ValueError("没有可绘制的有效频带。")

    fig, ax = plt.subplots(figsize=(11, 5.8))

    ax.semilogx(
        frequencies,
        reductions,
        marker="o",
        linewidth=1.8,
        label="1/3-octave noise reduction",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="No level change",
    )

    ax.axvspan(
        average_range[0],
        average_range[1],
        alpha=0.10,
        label=(
            f"Mean range: {average_range[0]:g}-"
            f"{average_range[1]:g} Hz"
        ),
    )

    ax.axvspan(
        rebound_range[0],
        rebound_range[1],
        alpha=0.10,
        label=(
            f"Rebound range: {rebound_range[0]:g}-"
            f"{rebound_range[1]:g} Hz"
        ),
    )

    ax.set_xlabel("Center frequency (Hz)")
    ax.set_ylabel("Noise reduction, Loff - Lon (dB)")
    ax.set_title("ANC 1/3-Octave Noise-Reduction Curve")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_third_octave_spectrum(
    center_frequencies: ArrayLike,
    off_level_db: ArrayLike,
    on_level_db: ArrayLike,
    valid_mask: Optional[ArrayLike] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """绘制ANC开启/关闭的1/3倍频程声压级对比柱状图。"""
    center_frequencies = np.asarray(
        center_frequencies,
        dtype=np.float64,
    )
    off_level_db = np.asarray(
        off_level_db,
        dtype=np.float64,
    )
    on_level_db = np.asarray(
        on_level_db,
        dtype=np.float64,
    )

    if valid_mask is None:
        valid_mask = np.isfinite(off_level_db) & np.isfinite(on_level_db)
    else:
        valid_mask = (
            np.asarray(valid_mask, dtype=bool)
            & np.isfinite(off_level_db)
            & np.isfinite(on_level_db)
        )

    frequencies = center_frequencies[valid_mask]
    off_levels = off_level_db[valid_mask]
    on_levels = on_level_db[valid_mask]

    if frequencies.size == 0:
        raise ValueError("没有可绘制的有效频带。")

    n_bands = len(frequencies)
    x = np.arange(n_bands)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.8))

    ax.bar(
        x - bar_width / 2,
        off_levels,
        width=bar_width,
        label="ANC OFF",
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.bar(
        x + bar_width / 2,
        on_levels,
        width=bar_width,
        label="ANC ON",
        color="#ff7f0e",
        edgecolor="white",
        linewidth=0.5,
    )

    # 格式化频率标签：整数Hz，kHz以上保留1位小数
    tick_labels = []
    for f in frequencies:
        if f >= 1000:
            tick_labels.append(f"{f / 1000:.1f}k")
        else:
            tick_labels.append(f"{int(f)}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("1/3-Octave center frequency (Hz)")
    ax.set_ylabel("Sound pressure level (dB)")
    ax.set_title("ANC On/Off 1/3-Octave Band Level Comparison")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def analyze_anc_audio(
    audio_path: Union[str, Path],
    anc_off_range: Tuple[float, float],
    anc_on_range: Tuple[float, float],
    n_fft: int = 8192,
    hop_length: Optional[int] = None,
    window_type: str = "hann",
    channel: Optional[int] = None,
    mono: bool = True,
    remove_mean: bool = True,
    average_range: Tuple[float, float] = (50.0, 5000.0),
    rebound_range: Tuple[float, float] = (1000.0, 8000.0),
    plot: bool = True,
    show_plot: bool = True,
    plot_path: Optional[Union[str, Path]] = None,
    octave_plot_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    完成ANC音频的全部分析流程。

    Parameters
    ----------
    audio_path:
        WAV音频文件路径。
    anc_off_range:
        ANC关闭时间段，单位为秒，例如(0.0, 5.0)。
    anc_on_range:
        ANC开启时间段，单位为秒，例如(6.0, 11.0)。
    n_fft:
        FFT点数。
    hop_length:
        STFT帧移，None表示n_fft // 4。
    window_type:
        hann、hamming或blackman。
    channel:
        指定单个分析通道，从0开始。None表示按mono参数处理。
    mono:
        多通道是否平均为单通道。
    remove_mean:
        是否去除每段音频的直流分量。
    average_range:
        平均降噪量统计范围。
    rebound_range:
        最大反弹统计范围。
    plot:
        是否绘制曲线。
    show_plot:
        是否显示绘图窗口。
    plot_path:
        降噪曲线保存路径。
    octave_plot_path:
        ANC开/关倍频程声压级对比图保存路径。

    Returns
    -------
    一个字典，包含FFT频谱、1/3倍频程结果和评价指标。
    """
    spectrum_analyzer = AudioSpectrumAnalyzer(
        n_fft=n_fft,
        hop_length=hop_length,
        window_type=window_type,
    )
    octave_analyzer = ThirdOctaveAnalyzer()

    waveform, sample_rate = spectrum_analyzer.load_audio(
        audio_path,
        channel=channel,
        mono=mono,
    )

    off_waveform = spectrum_analyzer.extract_segment(
        waveform,
        sample_rate,
        anc_off_range,
    )
    on_waveform = spectrum_analyzer.extract_segment(
        waveform,
        sample_rate,
        anc_on_range,
    )

    off_spectrum = spectrum_analyzer.compute_power_spectrum(
        off_waveform,
        sample_rate,
        remove_mean=remove_mean,
    )
    on_spectrum = spectrum_analyzer.compute_power_spectrum(
        on_waveform,
        sample_rate,
        remove_mean=remove_mean,
    )

    off_octave = octave_analyzer.fft_to_third_octave(
        off_spectrum.frequencies,
        off_spectrum.power_spectrum,
        sample_rate=sample_rate,
    )
    on_octave = octave_analyzer.fft_to_third_octave(
        on_spectrum.frequencies,
        on_spectrum.power_spectrum,
        sample_rate=sample_rate,
    )

    reduction = octave_analyzer.calculate_noise_reduction(
        off_octave,
        on_octave,
    )

    metrics = octave_analyzer.calculate_metrics(
        off_result=off_octave,
        on_result=on_octave,
        average_range=average_range,
        rebound_range=rebound_range,
    )

    if plot:
        plot_noise_reduction_curve(
            center_frequencies=(
                reduction["center_frequencies"]
            ),
            noise_reduction_db=(
                reduction["noise_reduction_db"]
            ),
            valid_mask=reduction["valid_mask"],
            average_range=average_range,
            rebound_range=rebound_range,
            output_path=plot_path,
            show=show_plot,
        )

        if octave_plot_path is not None:
            plot_third_octave_spectrum(
                center_frequencies=off_octave.center_frequencies,
                off_level_db=off_octave.band_level_db,
                on_level_db=on_octave.band_level_db,
                valid_mask=off_octave.valid_mask & on_octave.valid_mask,
                output_path=octave_plot_path,
                show=show_plot,
            )

    return {
        "sample_rate":
            sample_rate,
        "anc_off_time_range":
            anc_off_range,
        "anc_on_time_range":
            anc_on_range,
        "anc_off_spectrum":
            off_spectrum,
        "anc_on_spectrum":
            on_spectrum,
        "anc_off_third_octave":
            off_octave,
        "anc_on_third_octave":
            on_octave,
        "center_frequencies":
            reduction["center_frequencies"],
        "noise_reduction_db":
            reduction["noise_reduction_db"],
        "level_change_db":
            reduction["level_change_db"],
        "valid_mask":
            reduction["valid_mask"],
        "metrics":
            metrics,
    }


def print_analysis_results(
    results: Dict[str, object],
) -> None:
    """以便于阅读的格式打印分析结果。"""
    metrics = results["metrics"]

    print("=" * 60)
    print("ANC音频1/3倍频程分析结果")
    print("=" * 60)
    print(f"采样率：{results['sample_rate']} Hz")
    print(
        "ANC关闭时间段："
        f"{results['anc_off_time_range'][0]:.3f} - "
        f"{results['anc_off_time_range'][1]:.3f} s"
    )
    print(
        "ANC开启时间段："
        f"{results['anc_on_time_range'][0]:.3f} - "
        f"{results['anc_on_time_range'][1]:.3f} s"
    )
    print("-" * 60)
    print(
        "50 Hz-5 kHz频带降噪量算术平均："
        f"{metrics['average_noise_reduction_db']:.2f} dB"
    )
    print(
        "50 Hz-5 kHz总能量降噪量："
        f"{metrics['total_band_noise_reduction_db']:.2f} dB"
    )
    print(
        "1 kHz-8 kHz最大噪声反弹："
        f"{metrics['maximum_rebound_db']:.2f} dB"
    )
    print(
        "最差降噪量："
        f"{metrics['worst_noise_reduction_db']:.2f} dB，"
        "对应频率："
        f"{metrics['rebound_frequency_hz']:.1f} Hz"
    )
    print(
        "最大降噪量："
        f"{metrics['maximum_noise_reduction_db']:.2f} dB，"
        "对应频率："
        f"{metrics['maximum_reduction_frequency_hz']:.1f} Hz"
    )
    print("=" * 60)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "分析同一WAV文件中ANC关闭和开启时间段的"
            "1/3倍频程降噪效果。"
        )
    )

    parser.add_argument(
        "audio_path",
        type=str,
        help="WAV音频文件路径。",
    )
    parser.add_argument(
        "--off-start",
        type=float,
        required=True,
        help="ANC关闭片段起始时间，单位秒。",
    )
    parser.add_argument(
        "--off-end",
        type=float,
        required=True,
        help="ANC关闭片段结束时间，单位秒。",
    )
    parser.add_argument(
        "--on-start",
        type=float,
        required=True,
        help="ANC开启片段起始时间，单位秒。",
    )
    parser.add_argument(
        "--on-end",
        type=float,
        required=True,
        help="ANC开启片段结束时间，单位秒。",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=8192,
        help="FFT点数，默认8192。",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=None,
        help="STFT帧移，默认n_fft//4。",
    )
    parser.add_argument(
        "--window",
        choices=["hann", "hamming", "blackman"],
        default="hann",
        help="窗函数，默认hann。",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="指定音频通道编号，从0开始。",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="anc_third_octave_result.png",
        help="降噪曲线保存路径。",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不显示绘图窗口，仅保存图片。",
    )

    return parser


def main() -> None:
    """命令行程序入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    results = analyze_anc_audio(
        audio_path=args.audio_path,
        anc_off_range=(args.off_start, args.off_end),
        anc_on_range=(args.on_start, args.on_end),
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        window_type=args.window,
        channel=args.channel,
        plot=True,
        show_plot=not args.no_show,
        plot_path=args.plot_path,
    )

    print_analysis_results(results)


if __name__ == "__main__":
    main()
