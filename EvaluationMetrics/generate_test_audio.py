# -*- coding: utf-8 -*-
"""
生成ANC测试音频文件
前5秒：ANC关闭（噪声较大）
6-11秒：ANC开启（噪声较小，模拟降噪效果）
"""
import numpy as np
from scipy.io import wavfile

def generate_test_wav(output_path: str, sample_rate: int = 16000, duration: float = 12.0):
    """生成测试音频"""
    total_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    
    # 生成白噪声
    np.random.seed(42)
    white_noise = np.random.randn(total_samples) * 0.3
    
    # 生成多个音调分量（模拟实际噪声）
    freq_components = [100, 250, 500, 1000, 2000, 4000]
    tones = np.zeros_like(t)
    for freq in freq_components:
        tones += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # ANC关闭阶段（0-5秒）：噪声较大
    anc_off_mask = (t >= 0) & (t < 5.0)
    # ANC开启阶段（6-11秒）：噪声较小（模拟降噪）
    anc_on_mask = (t >= 6.0) & (t < 11.0)
    # 过渡阶段
    transition_mask = (t >= 5.0) & (t < 6.0)
    
    # 构建音频信号
    audio = np.zeros_like(t)
    
    # ANC关闭：全量噪声
    audio[anc_off_mask] = (white_noise[anc_off_mask] + tones[anc_off_mask]) * 1.0
    
    # 过渡段：线性衰减
    transition_t = t[transition_mask]
    alpha = 1.0 - (transition_t - 5.0) / 1.0  # 1.0 -> 0.4
    audio[transition_mask] = (white_noise[transition_mask] + tones[transition_mask]) * (0.4 + 0.6 * alpha)
    
    # ANC开启：噪声降低约10-15dB（幅值乘以0.25）
    audio[anc_on_mask] = (white_noise[anc_on_mask] + tones[anc_on_mask]) * 0.25
    
    # 尾部（11-12秒）：保持低噪声
    tail_mask = t >= 11.0
    audio[tail_mask] = (white_noise[tail_mask] + tones[tail_mask]) * 0.25
    
    # 归一化到[-1, 1]范围
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # 转换为16位整数并保存
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)
    
    print(f"测试音频已生成: {output_path}")
    print(f"采样率: {sample_rate} Hz")
    print(f"时长: {duration} 秒")
    print(f"ANC关闭时段: 0.0 - 5.0 秒")
    print(f"ANC开启时段: 6.0 - 11.0 秒")
    print(f"预期降噪量: 约12 dB")

if __name__ == "__main__":
    generate_test_wav("anc_recording.wav")
