import os
from itertools import product

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def apply_dynamic_path(signal_batch, path_batch):
    """Apply a batch of secondary-path impulse responses as causal convolutions."""
    batch_size, signal_length = signal_batch.shape
    path_length = path_batch.shape[1]

    signal_reshaped = signal_batch.view(1, batch_size, signal_length)
    path_flipped = torch.flip(path_batch, dims=[1])
    path_reshaped = path_flipped.view(batch_size, 1, path_length)

    signal_padded = F.pad(signal_reshaped, (path_length - 1, 0))
    output = F.conv1d(signal_padded, path_reshaped, groups=batch_size)
    return output.squeeze(0)


def load_secondary_paths(dataset_dir, path_indices):
    """Load sh.npy and normalize its layout to [num_paths, path_length]."""
    sh_path = os.path.join(dataset_dir, "sh.npy")
    sh_array = np.load(sh_path)

    if sh_array.ndim != 2:
        raise ValueError(f"sh.npy 必须是二维数组，实际形状为 {sh_array.shape}")

    max_path_index = max(path_indices) if path_indices else 0

    if sh_array.shape[0] > max_path_index and sh_array.shape[0] <= sh_array.shape[1]:
        paths = sh_array
    elif sh_array.shape[1] > max_path_index and sh_array.shape[1] < sh_array.shape[0]:
        paths = sh_array.T
    else:
        raise ValueError(
            "无法判断 sh.npy 的路径维度。"
            f"sh.npy 形状={sh_array.shape}, 最大路径索引={max_path_index}"
        )

    return paths.astype(np.float32, copy=False)


def infer_num_paths(dataset_dir):
    """Infer the number of acoustic paths from the shorter dimension of sh.npy."""
    sh_array = np.load(os.path.join(dataset_dir, "sh.npy"))
    if sh_array.ndim != 2:
        raise ValueError(f"sh.npy 必须是二维数组，实际形状为 {sh_array.shape}")
    return int(min(sh_array.shape))


class AudioSliceReader:
    """Shared helper that reads exactly the requested number of audio samples."""

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self._info_cache = {}

    def info(self, filepath):
        if filepath not in self._info_cache:
            self._info_cache[filepath] = sf.info(filepath)
        return self._info_cache[filepath]

    def read(self, filepath, start_index, num_frames):
        info = self.info(filepath)
        if info.samplerate != self.sample_rate:
            raise ValueError(
                f"采样率错误: {filepath} 是 {info.samplerate} Hz，"
                f"但配置要求 {self.sample_rate} Hz"
            )

        start_index = int(max(0, start_index))
        num_frames = int(num_frames)

        audio, read_sr = sf.read(
            filepath,
            start=start_index,
            frames=num_frames,
            dtype="float32",
            always_2d=False,
        )

        if read_sr != self.sample_rate:
            raise ValueError(f"读取采样率不一致: {read_sr} != {self.sample_rate}")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if len(audio) < num_frames:
            audio = np.pad(audio, (0, num_frames - len(audio)), mode="constant")
        elif len(audio) > num_frames:
            audio = audio[:num_frames]

        return audio.astype(np.float32, copy=False)


class BalancedTrainANCDataset(Dataset):
    """
    Training dataset with exact path-noise balance in every complete data cycle.

    One training item contains:
        x: raw reference noise
        sh: secondary-path impulse response
        d: disturbance measured at the error microphone before control
    """

    def __init__(
        self,
        dataset_dir,
        noise_names,
        path_indices,
        segment_duration=1.0,
        sample_rate=48000,
        repeats_per_combination=32,
        train_start_seconds=30.0,
        fallback_skip_seconds=20.0,
    ):
        if not noise_names:
            raise ValueError("训练噪声列表为空")
        if not path_indices:
            raise ValueError("训练路径列表为空")
        if repeats_per_combination <= 0:
            raise ValueError("repeats_per_combination 必须大于 0")

        self.dataset_dir = dataset_dir
        self.noise_names = list(noise_names)
        self.path_indices = list(path_indices)
        self.sample_rate = sample_rate
        self.segment_length = int(round(segment_duration * sample_rate))
        self.train_start_sample = int(round(train_start_seconds * sample_rate))
        self.fallback_skip_sample = int(round(fallback_skip_seconds * sample_rate))

        self.secondary_paths = load_secondary_paths(dataset_dir, self.path_indices)
        self.expected_dir = os.path.join(dataset_dir, "EXPECTED_NOISE")
        self.raw_noise_dir = os.path.join(dataset_dir, "NOISE")
        self.reader = AudioSliceReader(sample_rate)

        self.combinations = list(product(self.path_indices, self.noise_names))
        self.repeats_per_combination = int(repeats_per_combination)
        self.samples_per_cycle = len(self.combinations) * self.repeats_per_combination

    def __len__(self):
        return self.samples_per_cycle

    def _choose_random_start(self, raw_path, expected_path):
        available_frames = min(
            self.reader.info(raw_path).frames,
            self.reader.info(expected_path).frames,
        )
        max_start = available_frames - self.segment_length

        preferred_start = self.train_start_sample
        if max_start < preferred_start:
            preferred_start = self.fallback_skip_sample
        if max_start < preferred_start:
            preferred_start = 0

        if max_start <= preferred_start:
            return max(0, max_start)

        return int(np.random.randint(preferred_start, max_start + 1))

    def __getitem__(self, index):
        combination_index = index % len(self.combinations)
        path_index, noise_name = self.combinations[combination_index]

        raw_path = os.path.join(self.raw_noise_dir, f"{noise_name}.wav")
        expected_path = os.path.join(
            self.expected_dir,
            f"{noise_name}_scene_{path_index + 1:02d}.wav",
        )

        start_index = self._choose_random_start(raw_path, expected_path)
        x = self.reader.read(raw_path, start_index, self.segment_length)
        d = self.reader.read(expected_path, start_index, self.segment_length)
        sh = self.secondary_paths[path_index]

        return (
            torch.from_numpy(x),
            torch.from_numpy(sh.copy()),
            torch.from_numpy(d),
        )


class GridSpliceANCDataset(Dataset):
    """
    Deterministic path x time-window evaluation dataset.

    Every sample is exactly one second long. The first half comes from noise A,
    and the second half comes from noise B, so the acoustic noise changes at 0.5 s.
    All paths use exactly the same time windows, removing the old path/time confound.
    """

    def __init__(
        self,
        dataset_dir,
        noise_pair,
        path_indices,
        window_starts_seconds,
        segment_duration=1.0,
        sample_rate=48000,
    ):
        if len(noise_pair) != 2:
            raise ValueError("noise_pair 必须正好包含两个噪声名称")
        if not path_indices:
            raise ValueError("评估路径列表为空")
        if not window_starts_seconds:
            raise ValueError("评估时间窗口列表为空")

        self.dataset_dir = dataset_dir
        self.noise_pair = tuple(noise_pair)
        self.path_indices = list(path_indices)
        self.window_starts_seconds = [float(x) for x in window_starts_seconds]
        self.sample_rate = sample_rate
        self.segment_length = int(round(segment_duration * sample_rate))
        self.first_half_length = self.segment_length // 2
        self.second_half_length = self.segment_length - self.first_half_length

        self.secondary_paths = load_secondary_paths(dataset_dir, self.path_indices)
        self.expected_dir = os.path.join(dataset_dir, "EXPECTED_NOISE")
        self.raw_noise_dir = os.path.join(dataset_dir, "NOISE")
        self.reader = AudioSliceReader(sample_rate)

        self.grid = list(product(self.path_indices, range(len(self.window_starts_seconds))))

    def __len__(self):
        return len(self.grid)

    def _clamp_start(self, filepaths, desired_start, num_frames):
        available_frames = min(self.reader.info(path).frames for path in filepaths)
        max_start = max(0, available_frames - num_frames)
        return int(min(max(0, desired_start), max_start))

    def __getitem__(self, index):
        path_index, window_index = self.grid[index]
        noise_a, noise_b = self.noise_pair
        desired_start = int(round(
            self.window_starts_seconds[window_index] * self.sample_rate
        ))

        raw_a = os.path.join(self.raw_noise_dir, f"{noise_a}.wav")
        raw_b = os.path.join(self.raw_noise_dir, f"{noise_b}.wav")
        expected_a = os.path.join(
            self.expected_dir,
            f"{noise_a}_scene_{path_index + 1:02d}.wav",
        )
        expected_b = os.path.join(
            self.expected_dir,
            f"{noise_b}_scene_{path_index + 1:02d}.wav",
        )

        start_a = self._clamp_start(
            [raw_a, expected_a], desired_start, self.first_half_length
        )
        start_b = self._clamp_start(
            [raw_b, expected_b], desired_start, self.second_half_length
        )

        x_a = self.reader.read(raw_a, start_a, self.first_half_length)
        x_b = self.reader.read(raw_b, start_b, self.second_half_length)
        d_a = self.reader.read(expected_a, start_a, self.first_half_length)
        d_b = self.reader.read(expected_b, start_b, self.second_half_length)

        x = np.concatenate([x_a, x_b]).astype(np.float32, copy=False)
        d = np.concatenate([d_a, d_b]).astype(np.float32, copy=False)
        sh = self.secondary_paths[path_index]

        return (
            torch.from_numpy(x),
            torch.from_numpy(sh.copy()),
            torch.from_numpy(d),
            torch.tensor(path_index, dtype=torch.long),
            torch.tensor(window_index, dtype=torch.long),
        )
