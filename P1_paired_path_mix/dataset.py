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
    Balanced training dataset with optional physically paired path interpolation.

    For every training item:
        x: raw reference noise
        sh: secondary-path impulse response used by the differentiable ANC loop
        d: disturbance measured at the error microphone before control

    When path mix is enabled, two training scenes i and j are read using:
        - the same noise name
        - the same sample start
        - the same interpolation coefficient alpha

    The virtual path sample is:
        d_mix  = alpha * d_i  + (1 - alpha) * d_j
        sh_mix = alpha * sh_i + (1 - alpha) * sh_j

    Keeping x, time position, and alpha aligned is essential. Otherwise the
    synthetic sample no longer represents one internally consistent linear ANC
    system.
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
        path_mix_probability=0.5,
        path_mix_alpha_min=0.2,
        path_mix_alpha_max=0.8,
    ):
        if not noise_names:
            raise ValueError("训练噪声列表为空")
        if not path_indices:
            raise ValueError("训练路径列表为空")
        if repeats_per_combination <= 0:
            raise ValueError("repeats_per_combination 必须大于 0")
        if not 0.0 <= path_mix_probability <= 1.0:
            raise ValueError("path_mix_probability 必须位于 [0, 1]")
        if not 0.0 <= path_mix_alpha_min < path_mix_alpha_max <= 1.0:
            raise ValueError(
                "必须满足 0 <= path_mix_alpha_min < "
                "path_mix_alpha_max <= 1"
            )
        if path_mix_probability > 0.0 and len(path_indices) < 2:
            raise ValueError("启用路径插值时至少需要两条训练路径")

        self.dataset_dir = dataset_dir
        self.noise_names = list(noise_names)
        self.path_indices = list(path_indices)
        self.sample_rate = sample_rate
        self.segment_length = int(round(segment_duration * sample_rate))
        self.train_start_sample = int(round(train_start_seconds * sample_rate))
        self.fallback_skip_sample = int(round(fallback_skip_seconds * sample_rate))

        self.path_mix_probability = float(path_mix_probability)
        self.path_mix_alpha_min = float(path_mix_alpha_min)
        self.path_mix_alpha_max = float(path_mix_alpha_max)

        self.secondary_paths = load_secondary_paths(dataset_dir, self.path_indices)
        self.expected_dir = os.path.join(dataset_dir, "EXPECTED_NOISE")
        self.raw_noise_dir = os.path.join(dataset_dir, "NOISE")
        self.reader = AudioSliceReader(sample_rate)

        # The primary path/noise combination remains exactly balanced, just as in E0.
        self.combinations = list(product(self.path_indices, self.noise_names))
        self.repeats_per_combination = int(repeats_per_combination)
        self.samples_per_cycle = len(self.combinations) * self.repeats_per_combination

        # Use an exact number of mixed repeats in every complete data cycle.
        # With the default 32 repeats and p=0.5, every path-noise combination
        # contains exactly 16 real-path samples and 16 mixed-path samples.
        self.mixed_repeats_per_combination = int(
            round(self.repeats_per_combination * self.path_mix_probability)
        )
        self.original_repeats_per_combination = (
            self.repeats_per_combination - self.mixed_repeats_per_combination
        )

        self.partner_candidates = {
            path_index: [
                candidate
                for candidate in self.path_indices
                if candidate != path_index
            ]
            for path_index in self.path_indices
        }

    def __len__(self):
        return self.samples_per_cycle

    def _expected_path(self, noise_name, path_index):
        return os.path.join(
            self.expected_dir,
            f"{noise_name}_scene_{path_index + 1:02d}.wav",
        )

    def _choose_random_start(self, filepaths):
        """
        Choose one common start index that is valid for every file.

        A common start is required because d_i and d_j must correspond to the
        exact same reference-noise samples x.
        """
        available_frames = min(
            self.reader.info(filepath).frames
            for filepath in filepaths
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

    def _select_partner_path(
        self,
        primary_path_index,
        combination_index,
        repeat_index,
    ):
        """
        Select a different training path with a deterministic balanced schedule.

        Random audio starts and alpha values still change on every retrieval, but
        partner identities do not depend on DataLoader order. This makes runs
        reproducible and prevents one path from being chosen as partner far more
        often merely by chance.
        """
        candidates = self.partner_candidates[primary_path_index]
        partner_position = (combination_index + repeat_index) % len(candidates)
        return candidates[partner_position]

    def __getitem__(self, index):
        num_combinations = len(self.combinations)
        combination_index = index % num_combinations
        repeat_index = index // num_combinations

        primary_path_index, noise_name = self.combinations[combination_index]
        raw_path = os.path.join(self.raw_noise_dir, f"{noise_name}.wav")
        expected_primary_path = self._expected_path(
            noise_name,
            primary_path_index,
        )

        use_path_mix = repeat_index < self.mixed_repeats_per_combination

        if not use_path_mix:
            start_index = self._choose_random_start(
                [raw_path, expected_primary_path]
            )
            x = self.reader.read(
                raw_path,
                start_index,
                self.segment_length,
            )
            d = self.reader.read(
                expected_primary_path,
                start_index,
                self.segment_length,
            )
            sh = self.secondary_paths[primary_path_index].copy()
        else:
            partner_path_index = self._select_partner_path(
                primary_path_index,
                combination_index,
                repeat_index,
            )
            expected_partner_path = self._expected_path(
                noise_name,
                partner_path_index,
            )

            # One common time position is used for x, d_i, and d_j.
            start_index = self._choose_random_start(
                [
                    raw_path,
                    expected_primary_path,
                    expected_partner_path,
                ]
            )

            x = self.reader.read(
                raw_path,
                start_index,
                self.segment_length,
            )
            d_primary = self.reader.read(
                expected_primary_path,
                start_index,
                self.segment_length,
            )
            d_partner = self.reader.read(
                expected_partner_path,
                start_index,
                self.segment_length,
            )

            alpha = np.float32(
                np.random.uniform(
                    self.path_mix_alpha_min,
                    self.path_mix_alpha_max,
                )
            )
            one_minus_alpha = np.float32(1.0) - alpha

            d = (
                alpha * d_primary
                + one_minus_alpha * d_partner
            ).astype(np.float32, copy=False)

            sh = (
                alpha * self.secondary_paths[primary_path_index]
                + one_minus_alpha * self.secondary_paths[partner_path_index]
            ).astype(np.float32, copy=False)

        return (
            torch.from_numpy(x),
            torch.from_numpy(sh),
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
