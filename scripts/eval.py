# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

#%%
import os
import sys
sys.path.append('..')
import math
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse
import time, datetime
import logging
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn

from xvq.dataset import data_loader, load_preprocessor, load_frozen_vae, AudioPreprocessor
from xvq.utils import reconstruct_sample, seed_everything, check_rank_zero, to_ddp_model, set_train, set_eval
from xvq.models import setup_models
from xvq.evaluator import AudioEvaluator
from xvq.config import get_config

def log_and_print(message, logger=None):
    """同时输出到logger和控制台"""
    print(message)
    if logger is not None:
        logger.info(message)



def main_worker(rank, config):
    # setup devices
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    config.rank = rank

    # setup distribution
    if config.world_size > 1:
        dist.init_process_group(
            backend=config.backend, init_method=config.init_method,
            rank=config.rank, world_size=config.world_size
        )

    if check_rank_zero():
        os.makedirs(config.log_path, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(config.log_path, "eval.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'  
        )

        logger = logging.getLogger(__name__)
        logger.info(" ReVQ eval Started")

 
    val_dataset, val_loader = data_loader(
        root_dir=config.data.data_val_path,
        split='val', 
        sample_rate= config.data.sample_rate, 
        acoustic_dim = config.data.acoustic_dim,
        win_size = config.data.win_size,
        hop_size = config.data.hop_size,
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False
    )
    
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # 🔥 创建音频预处理器（与训练时一致）
    audio_preprocessor = AudioPreprocessor(
        sample_rate=config.model.audio_preprocessor.sample_rate,
        n_fft=config.model.audio_preprocessor.n_fft,
        hop_length=config.model.audio_preprocessor.hop_length,
        n_mels=config.model.audio_preprocessor.n_mels,
        target_length=config.model.audio_preprocessor.target_length
    ).to(device)
    
    # setup the model
    decoder, quantizer, viewer = setup_models(config.model, device)
    vae_encoder, vae_decoder = load_frozen_vae(device=device, config=config.model, if_decoder=True)

    if config.world_size > 1:
        decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
    
    if check_rank_zero():
        get_param_num = lambda x: sum(p.numel() for p in x.parameters() if p.requires_grad)
        logger.info(f"Quantizer: {get_param_num(quantizer) / 1e6:.2f}M")
        logger.info(f"Decoder: {get_param_num(decoder) / 1e6:.2f}M")
        total_params = get_param_num(quantizer) + get_param_num(decoder)
        logger.info(f"Total params: {total_params / 1e6:.2f}M")
    
    # 🔥 加载训练好的检查点（与train.py保存格式一致）
    checkpoint_path = os.path.join(config.log_path, "ckpt.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        quantizer.load_state_dict(checkpoint["quantizer"])
        decoder.load_state_dict(checkpoint["decoder"])
        audio_preprocessor.load_state_dict(checkpoint["audio_preprocessor"])
        if check_rank_zero():
            logger.info(f"✓ 模型加载完成，来自epoch {checkpoint['epoch']}")
            logger.info(f"  从 {checkpoint_path} 加载")
    else:
        if check_rank_zero():
            logger.info(f"⚠️  检查点文件不存在: {checkpoint_path}")
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    # start evaluation - 创建音频评估器
    quantizer, decoder = to_ddp_model(rank, quantizer, decoder)
    evaluator = AudioEvaluator(
        device=device,
        sample_rate=config.model.audio_preprocessor.sample_rate,
        n_fft=config.model.audio_preprocessor.n_fft,
        hop_length=config.model.audio_preprocessor.hop_length,
        n_mels=config.model.audio_preprocessor.n_mels,
        enable_time_domain=True,        # 时域指标 (MSE, MAE, SNR)
        enable_frequency_domain=True,   # 频域指标
        enable_mel_domain=True,         # Mel域指标
        enable_codebook_stats=True,     # 码本统计
    )

    def audio_vq_infer(raw_audio):
        with torch.no_grad():
            data = audio_preprocessor(raw_audio)
            data = vae_encoder(data)
            data_shuffle = viewer.shuffle(data)
            quant_result = quantizer(data_shuffle)
            quant_shuffle = quant_result["x_quant"]
            quant = viewer.unshuffle(quant_shuffle)
            data_rec = decoder(quant)
            reconstructed_audio = vae_decoder(data_rec)
            
            return {
                'original_latent': data,
                'reconstructed_latent': data_rec,
                'quantized_latent': quant,
                'reconstructed_audio': reconstructed_audio,
                'codebook_indices': quant_result.get('indices', None)  # 码本索引（如果有）
            }
        
    set_eval(audio_preprocessor, vae_encoder, quantizer, decoder)  # 设置所有模型为评估模式
    pbar = tqdm(val_loader)     
    with torch.no_grad():
        total_loss = 0
        for i, (raw_audio, _) in enumerate(pbar):
            raw_audio = raw_audio.to(device)
            
            # 使用完整的音频推理流程
            results = audio_vq_infer(raw_audio)
            
            # 计算重建损失（与训练验证一致）
            loss_quant = F.mse_loss(
                results['original_latent'], 
                results['quantized_latent']
            )
            loss_decoder = F.mse_loss(
                results['original_latent'], 
                results['reconstructed_latent']
            )
            total_loss_batch = loss_quant + loss_decoder
            total_loss += total_loss_batch.item()
            
            # 🔥 更新音频评估器（传入原始音频和重建音频）
            evaluator.update(
                real_audio=raw_audio,
                fake_audio=results['reconstructed_audio'],
                codebook_indices=results['codebook_indices']
            )
            
            pbar.set_description(
                f"Eval {i}/{len(val_loader)} | "
                f"Loss: {total_loss_batch.item():.4f} | "
                f"Avg: {total_loss/(i+1):.4f}"
            )

            if (i+1) % 50 == 0:
                avg_loss = total_loss / (i+1)
                results_eval = evaluator.result()
                if check_rank_zero():
                    log_and_print(f"📊 Progress: {i+1}/{len(val_loader)} batches, Average loss: {avg_loss:.4f}", logger)
                    log_and_print("🎵 当前音频评估结果:", logger)
                    for key, value in results_eval.items():
                        log_and_print(f"  {key}: {value:.4f}", logger)

    # 最终评估结果
    final_avg_loss = total_loss / len(val_loader)
    final_results = evaluator.result()
    
    if check_rank_zero():
        log_and_print(f"🎯 评估完成! 最终结果:", logger)
        log_and_print(f"  - Final average loss: {final_avg_loss:.4f}", logger)
        log_and_print(f"  - Processed {len(val_loader)} batches", logger)
        log_and_print(f"  - Total samples: {len(val_dataset)}", logger)
        log_and_print("", logger)
        log_and_print("🎵 最终音频评估指标:", logger)
        for key, value in final_results.items():
            log_and_print(f"  {key}: {value:.4f}", logger)
    else:
        print(f"Final average loss: {final_avg_loss:.4f}")
        print("Final audio evaluation results:")
        for key, value in final_results.items():
            print(f"  {key}: {value:.4f}")

    if dist.is_available() and dist.is_initialized():
        # destroy the process group
        dist.destroy_process_group()
    

def main():
    # setup config
    config = get_config()
    seed_everything(config.seed)
    
    # launch
    if config.world_size > 1:
        torch.multiprocessing.spawn(main_worker, args=(config,), nprocs=config.world_size)
    else:
        main_worker(0, config)
    
if __name__ == "__main__":
    main()
