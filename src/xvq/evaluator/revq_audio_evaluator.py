# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import warnings
from typing import Optional, Mapping, Text
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class AudioEvaluator:
    """通用音频VQ模型评估器，支持ReVQ等多种音频量化模型"""
    
    def __init__(
        self,
        device,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        enable_time_domain: bool = True,      # 时域指标 (MSE, MAE, SNR)
        enable_frequency_domain: bool = True,  # 频域指标 (频谱距离)
        enable_mel_domain: bool = True,       # Mel域指标 (Mel频谱距离)
        enable_codebook_stats: bool = True,   # 码本统计
        num_codebook_entries: int = None,     # 码本大小（支持自动检测）
        num_groups: int = None,               # 分组数（ReVQ专用，支持自动检测）
    ):
        """初始化通用音频评估器
        
        Args:
            device: 计算设备
            sample_rate: 音频采样率
            n_fft: FFT窗口大小
            hop_length: FFT跳跃长度  
            n_mels: Mel滤波器数量
            enable_time_domain: 是否启用时域评估
            enable_frequency_domain: 是否启用频域评估
            enable_mel_domain: 是否启用Mel域评估
            enable_codebook_stats: 是否启用码本统计
            num_codebook_entries: 总码本条目数（None表示自动检测）
            num_groups: 分组数量（ReVQ专用，None表示自动检测）
        """
        self._device = device
        self._sample_rate = sample_rate
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._n_mels = n_mels
        
        # 功能开关
        self._enable_time_domain = enable_time_domain
        self._enable_frequency_domain = enable_frequency_domain
        self._enable_mel_domain = enable_mel_domain
        self._enable_codebook_stats = enable_codebook_stats
        
        # 码本参数（支持自动检测）
        self._num_codebook_entries = num_codebook_entries
        self._num_groups = num_groups
        self._auto_detect_codebook = (num_codebook_entries is None) or (num_groups is None)
        
        # 如果提供了完整参数，计算每组码字数
        if self._num_codebook_entries is not None and self._num_groups is not None:
            self._codes_per_group = self._num_codebook_entries // self._num_groups
        
        # 初始化Mel频谱变换
        if self._enable_mel_domain:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
                normalized=True
            ).to(device)
        
        self.reset_metrics()
    
    def reset_metrics(self):
        """重置所有评估指标"""
        self._num_examples = 0
        self._num_updates = 0
        
        # 时域指标
        self._mse_total = 0.0
        self._mae_total = 0.0
        self._snr_total = 0.0
        
        # 频域指标
        self._spectral_mse_total = 0.0
        self._spectral_mae_total = 0.0
        
        # Mel域指标
        self._mel_mse_total = 0.0
        self._mel_mae_total = 0.0
        
        # 码本统计（只有在确定参数后才初始化）
        if self._enable_codebook_stats and not self._auto_detect_codebook:
            self._init_codebook_stats()
    
    def _init_codebook_stats(self):
        """初始化码本统计"""
        # 全局码本使用情况
        self._used_codes = set()
        # 每组码本使用情况（如果有分组）
        if self._num_groups is not None:
            self._group_used_codes = [set() for _ in range(self._num_groups)]
        # 码本频率统计
        if self._num_codebook_entries is not None:
            self._code_frequencies = torch.zeros(
                self._num_codebook_entries, 
                dtype=torch.float64, 
                device=self._device
            )
    
    def _auto_detect_codebook_structure(self, codebook_indices: torch.Tensor):
        """自动检测码本结构"""
        if not self._auto_detect_codebook:
            return
        
        print(f"🔍 正在自动检测码本结构...")
        
        # 检测码本大小
        if self._num_codebook_entries is None:
            max_index = codebook_indices.max().item()
            self._num_codebook_entries = max_index + 1
            print(f"  - 检测到码本大小: {self._num_codebook_entries}")
        
        # 检测分组结构（ReVQ特有）
        if len(codebook_indices.shape) >= 3 and self._num_groups is None:
            self._num_groups = codebook_indices.shape[1]  # 第二维通常是组数
            if self._num_codebook_entries is not None:
                self._codes_per_group = self._num_codebook_entries // self._num_groups
                print(f"  - 检测到ReVQ结构: {self._num_groups}组，每组{self._codes_per_group}个码字")
        
        # 初始化码本统计
        if self._enable_codebook_stats:
            self._init_codebook_stats()
            print(f"  - 码本统计已初始化")
        
        self._auto_detect_codebook = False  # 只检测一次
    
    def _compute_snr(self, signal: torch.Tensor, noise: torch.Tensor) -> float:
        """计算信噪比 (SNR)"""
        signal_power = torch.mean(signal ** 2)
        noise_power = torch.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    
    def _compute_spectral_metrics(self, real_audio: torch.Tensor, fake_audio: torch.Tensor):
        """计算频谱域指标"""
        # 计算STFT
        real_stft = torch.stft(
            real_audio.squeeze(), 
            n_fft=self._n_fft, 
            hop_length=self._hop_length,
            window=torch.hann_window(self._n_fft).to(self._device),
            return_complex=True
        )
        fake_stft = torch.stft(
            fake_audio.squeeze(), 
            n_fft=self._n_fft, 
            hop_length=self._hop_length,
            window=torch.hann_window(self._n_fft).to(self._device),
            return_complex=True
        )
        
        # 计算幅度谱
        real_mag = torch.abs(real_stft)
        fake_mag = torch.abs(fake_stft)
        
        # 频谱MSE和MAE
        spec_mse = F.mse_loss(real_mag, fake_mag).item()
        spec_mae = F.l1_loss(real_mag, fake_mag).item()
        
        return spec_mse, spec_mae
    
    def _compute_mel_metrics(self, real_audio: torch.Tensor, fake_audio: torch.Tensor):
        """计算Mel域指标"""
        # 确保音频维度正确
        if real_audio.dim() == 1:
            real_audio = real_audio.unsqueeze(0)
        if fake_audio.dim() == 1:
            fake_audio = fake_audio.unsqueeze(0)
        
        # 计算Mel频谱
        real_mel = self._mel_transform(real_audio)
        fake_mel = self._mel_transform(fake_audio)
        
        # 转换为对数刻度（dB）
        real_mel_db = torchaudio.transforms.AmplitudeToDB()(real_mel)
        fake_mel_db = torchaudio.transforms.AmplitudeToDB()(fake_mel)
        
        # Mel频谱MSE和MAE
        mel_mse = F.mse_loss(real_mel_db, fake_mel_db).item()
        mel_mae = F.l1_loss(real_mel_db, fake_mel_db).item()
        
        return mel_mse, mel_mae
    
    def _update_codebook_stats(self, codebook_indices: torch.Tensor):
        """更新码本统计信息"""
        if not self._enable_codebook_stats:
            return
        
        # 如果还没检测过码本结构，先进行检测
        if self._auto_detect_codebook:
            self._auto_detect_codebook_structure(codebook_indices)
        
        # codebook_indices可能有不同的形状
        # ReVQ: [batch_size, groups, spatial_dims...]
        # 其他VQ: [batch_size, spatial_dims...]
        batch_size = codebook_indices.shape[0]
        
        for b in range(batch_size):
            indices_batch = codebook_indices[b]
            
            if self._num_groups is not None and len(indices_batch.shape) >= 2:
                # ReVQ分组情况
                for g in range(min(self._num_groups, indices_batch.shape[0])):
                    group_indices = indices_batch[g].flatten()
                    
                    # 转换为全局索引
                    if hasattr(self, '_codes_per_group'):
                        global_indices = g * self._codes_per_group + group_indices
                    else:
                        global_indices = group_indices
                    
                    # 更新统计
                    unique_indices = torch.unique(global_indices)
                    if hasattr(self, '_used_codes'):
                        self._used_codes.update(unique_indices.cpu().tolist())
                    if hasattr(self, '_group_used_codes') and g < len(self._group_used_codes):
                        self._group_used_codes[g].update(group_indices.cpu().tolist())
                    
                    # 更新频率统计
                    if hasattr(self, '_code_frequencies'):
                        for idx in global_indices:
                            if 0 <= idx < self._code_frequencies.shape[0]:
                                self._code_frequencies[idx] += 1
            else:
                # 传统VQ情况
                all_indices = indices_batch.flatten()
                unique_indices = torch.unique(all_indices)
                
                if hasattr(self, '_used_codes'):
                    self._used_codes.update(unique_indices.cpu().tolist())
                
                if hasattr(self, '_code_frequencies'):
                    for idx in all_indices:
                        if 0 <= idx < self._code_frequencies.shape[0]:
                            self._code_frequencies[idx] += 1
    
    def update(
        self,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor,
        codebook_indices: Optional[torch.Tensor] = None
    ):
        """更新评估指标
        
        Args:
            real_audio: 原始音频 [batch_size, time] 或 [batch_size, channels, time]
            fake_audio: 重建音频 [batch_size, time] 或 [batch_size, channels, time]  
            codebook_indices: 码本索引 [batch_size, groups, spatial_dims...]
        """
        batch_size = real_audio.shape[0]
        self._num_examples += batch_size
        self._num_updates += 1
        
        # 确保音频在正确设备上
        real_audio = real_audio.to(self._device)
        fake_audio = fake_audio.to(self._device)
        
        # 处理每个音频样本
        for i in range(batch_size):
            real_sample = real_audio[i]
            fake_sample = fake_audio[i]
            
            # 如果是多声道，取平均或第一个声道
            if real_sample.dim() > 1:
                real_sample = real_sample.mean(dim=0)
            if fake_sample.dim() > 1:
                fake_sample = fake_sample.mean(dim=0)
            
            # 确保长度一致
            min_len = min(real_sample.shape[-1], fake_sample.shape[-1])
            real_sample = real_sample[..., :min_len]
            fake_sample = fake_sample[..., :min_len]
            
            # 时域指标
            if self._enable_time_domain:
                # MSE和MAE
                mse = F.mse_loss(real_sample, fake_sample).item()
                mae = F.l1_loss(real_sample, fake_sample).item()
                self._mse_total += mse
                self._mae_total += mae
                
                # SNR
                noise = real_sample - fake_sample
                snr = self._compute_snr(real_sample, noise)
                if not np.isinf(snr):
                    self._snr_total += snr
            
            # 频域指标
            if self._enable_frequency_domain:
                spec_mse, spec_mae = self._compute_spectral_metrics(real_sample, fake_sample)
                self._spectral_mse_total += spec_mse
                self._spectral_mae_total += spec_mae
            
            # Mel域指标
            if self._enable_mel_domain:
                mel_mse, mel_mae = self._compute_mel_metrics(real_sample, fake_sample)
                self._mel_mse_total += mel_mse
                self._mel_mae_total += mel_mae
        
        # 码本统计
        if codebook_indices is not None:
            self._update_codebook_stats(codebook_indices)
    
    def result(self) -> Mapping[Text, float]:
        """返回评估结果"""
        if self._num_examples < 1:
            raise ValueError("No examples to evaluate.")
        
        results = {}
        
        # 时域指标
        if self._enable_time_domain:
            results["Time_MSE"] = self._mse_total / self._num_examples
            results["Time_MAE"] = self._mae_total / self._num_examples
            results["Time_RMSE"] = np.sqrt(results["Time_MSE"])
            results["SNR_dB"] = self._snr_total / self._num_examples
        
        # 频域指标
        if self._enable_frequency_domain:
            results["Spectral_MSE"] = self._spectral_mse_total / self._num_examples
            results["Spectral_MAE"] = self._spectral_mae_total / self._num_examples
        
        # Mel域指标
        if self._enable_mel_domain:
            results["Mel_MSE"] = self._mel_mse_total / self._num_examples
            results["Mel_MAE"] = self._mel_mae_total / self._num_examples
        
        # 码本统计
        if self._enable_codebook_stats and hasattr(self, '_used_codes'):
            # 全局码本使用率
            if self._num_codebook_entries is not None:
                total_usage = len(self._used_codes) / self._num_codebook_entries
                results["Codebook_Usage"] = total_usage
            
            # 分组统计（ReVQ特有）
            if hasattr(self, '_group_used_codes') and hasattr(self, '_codes_per_group'):
                group_usages = []
                for g in range(len(self._group_used_codes)):
                    group_usage = len(self._group_used_codes[g]) / self._codes_per_group
                    group_usages.append(group_usage)
                
                if group_usages:
                    results["Avg_Group_Usage"] = np.mean(group_usages)
                    results["Min_Group_Usage"] = np.min(group_usages)
                    results["Max_Group_Usage"] = np.max(group_usages)
            
            # 码本熵
            if hasattr(self, '_code_frequencies') and self._code_frequencies.sum() > 0:
                probs = self._code_frequencies / self._code_frequencies.sum()
                entropy = (-torch.log2(probs + 1e-12) * probs).sum().item()
                results["Codebook_Entropy"] = entropy
                
                # 归一化熵
                if self._num_codebook_entries is not None:
                    max_entropy = np.log2(self._num_codebook_entries)
                    results["Normalized_Entropy"] = entropy / max_entropy
            else:
                results["Codebook_Entropy"] = 0.0
                if self._num_codebook_entries is not None:
                    results["Normalized_Entropy"] = 0.0
        
        return results


if __name__ == "__main__":
    # 测试通用音频评估器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎵 测试通用音频评估器...")
    
    # 测试1: ReVQ模式（提供完整参数）
    print("\n1. ReVQ模式测试:")
    evaluator_revq = AudioEvaluator(
        device=device,
        sample_rate=22050,
        num_codebook_entries=65536,
        num_groups=256
    )
    
    # 生成ReVQ测试数据
    batch_size = 2
    audio_length = 22050  # 1秒
    groups = 256
    spatial_dim = 32
    
    real_audio = torch.randn(batch_size, audio_length).to(device)
    fake_audio = real_audio + torch.randn_like(real_audio) * 0.1
    revq_indices = torch.randint(0, 256, (batch_size, groups, spatial_dim)).to(device)
    
    evaluator_revq.update(real_audio, fake_audio, revq_indices)
    results_revq = evaluator_revq.result()
    
    print("ReVQ评估结果:")
    for metric, value in results_revq.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    # 测试2: 自动检测模式
    print("\n2. 自动检测模式测试:")
    evaluator_auto = AudioEvaluator(
        device=device,
        sample_rate=22050,
        # 不提供码本参数，让它自动检测
    )
    
    # 生成传统VQ测试数据
    vq_indices = torch.randint(0, 1024, (batch_size, 64, 64)).to(device)
    
    evaluator_auto.update(real_audio, fake_audio, vq_indices)
    results_auto = evaluator_auto.result()
    
    print("自动检测评估结果:")
    for metric, value in results_auto.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    # 测试3: 仅音频质量评估（无码本统计）
    print("\n3. 纯音频质量评估:")
    evaluator_audio_only = AudioEvaluator(
        device=device,
        sample_rate=22050,
        enable_codebook_stats=False  # 关闭码本统计
    )
    
    evaluator_audio_only.update(real_audio, fake_audio)  # 不传码本索引
    results_audio_only = evaluator_audio_only.result()
    
    print("纯音频质量评估结果:")
    for metric, value in results_audio_only.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    print("\n✅ 所有测试完成！")
