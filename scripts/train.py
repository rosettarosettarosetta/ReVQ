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
from diffusers import AutoencoderDC
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse
import time, datetime
import logging
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from xvq.dataset import data_loader, load_frozen_vae, AudioPreprocessor
from xvq.utils import reconstruct_sample, seed_everything, check_rank_zero, to_ddp_model, set_train, set_eval
from xvq.models import setup_models
from xvq.config import get_config

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
            filename=os.path.join(config.log_path, "training.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'  
        )

        logger = logging.getLogger(__name__)
        logger.info(" ReVQ Training Started")
        logger.info(f" Device: {device}")
        logger.info(f" Random Seed: {config.seed}")



    # setup dataloaders using raw images instead of cached features
    train_dataset, train_loader = data_loader(
        root_dir=config.data.data_train_path,
        split='train',
        sample_rate= config.data.sample_rate, 
        acoustic_dim = config.data.acoustic_dim,
        win_size = config.data.win_size,
        hop_size = config.data.hop_size,
        #......
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
        shuffle=True if config.world_size == 1 else False
    )
    
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
    
    # Setup distributed sampler if needed
    if config.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, 
            num_replicas=config.world_size, 
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, 
            num_replicas=config.world_size, 
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.train.batch_size,
            sampler=train_sampler,
            num_workers=config.data.num_workers,  
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if config.data.num_workers > 0 else False
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config.train.batch_size,
            sampler=val_sampler,
            num_workers=config.data.num_workers,  
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if config.data.num_workers > 0 else False
        )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # 根据配置数据类型设置预处理器
        # 音频模式：使用AudioPreprocessor
    audio_preprocessor = AudioPreprocessor(
        sample_rate=config.model.audio_preprocessor.sample_rate,
        n_fft=config.model.audio_preprocessor.n_fft,
        hop_length=config.model.audio_preprocessor.hop_length,
        n_mels=config.model.audio_preprocessor.n_mels,
        target_length=config.model.audio_preprocessor.target_length
    ).to(device)
       
    if check_rank_zero():
        logger.info(f"✓ 音频预处理器配置:")
        logger.info(f"  - 采样率: {audio_preprocessor.sample_rate}")
        logger.info(f"  - Mel滤波器数: {config.model.audio_preprocessor.n_mels}")
        logger.info(f"  - 目标长度: {config.model.audio_preprocessor.target_length}")
        


    if check_rank_zero():
        OmegaConf.save(config, os.path.join(config.log_path, "config.yaml"))
    
  
    # setup the model (在设置预处理器后)
    decoder, quantizer, viewer = setup_models(config.model, device)
    vae_encoder, _ = load_frozen_vae(device=device, config=config.model,if_decoder=False)

    # 🔥 使用随机初始化codebook，无需预计算数据
    logger.info("🔄 使用随机初始化codebook...")
    with torch.no_grad():
        # 使用标准正态分布初始化embeddings，标准差设为0.02
        nn.init.normal_(quantizer.embeddings.data, mean=0.0, std=0.02)
        
        # 初始化计数器为1
        nn.init.ones_(quantizer.count)
        
        logger.info(f"✓ Codebook随机初始化完成:")
        logger.info(f"  - 组数: {quantizer.num_group}")
        logger.info(f"  - 每组码本大小: {quantizer.num_code}")
        logger.info(f"  - 码字维度: {quantizer.code_dim}")
        logger.info(f"  - 总码字数: {quantizer.num_group * quantizer.num_code:,}")

    if config.world_size > 1:
        decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder) 

    quantizer_optimizer = torch.optim.AdamW(
        params=quantizer.parameters(),
        lr=config.train.lr,           # 2e-4
        weight_decay=0.0
    )

    decoder_optimizer = torch.optim.AdamW(
        params=decoder.parameters(),
        lr=config.train.lr * 0.05,   # 1e-5
        weight_decay=1e-4
    )

    # 分别创建调度器
    quantizer_gamma = math.pow(0.05, 1 / config.train.num_epochs)
    decoder_gamma = math.pow(0.1, 1 / config.train.num_epochs)    # 可以设置不同的衰减率

    quantizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=quantizer_optimizer, gamma=quantizer_gamma
    )

    decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=decoder_optimizer, gamma=decoder_gamma
    )

    # print information
    if check_rank_zero():
        get_param_num = lambda x: sum(p.numel() for p in x.parameters() if p.requires_grad)
        logger.info(f"Quantizer: {get_param_num(quantizer) / 1e6:.2f}M")
        logger.info(f"Decoder: {get_param_num(decoder) / 1e6:.2f}M")
        total_params = get_param_num(quantizer) + get_param_num(decoder)
        logger.info(f"Total params: {total_params / 1e6:.2f}M")
        eq_num_images = int(total_params / 2048)
        logger.info(f"Model equivalent num images: {eq_num_images}")
        logger.info(f"Model Learning rate: {config.train.lr}")    
        tb_writer = SummaryWriter(config.log_path)
    
    # auto resume
    if os.path.exists(os.path.join(config.log_path, "ckpt.pth")):
        checkpoint = torch.load(os.path.join(config.log_path, "ckpt.pth"), map_location=device, weights_only=True)
        quantizer.load_state_dict(checkpoint["quantizer"])
        decoder.load_state_dict(checkpoint["decoder"])
        quantizer_optimizer.load_state_dict(checkpoint["quantizer_optimizer"])
        decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
        quantizer_scheduler.load_state_dict(checkpoint["quantizer_scheduler"])
        decoder_scheduler.load_state_dict(checkpoint["decoder_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        if check_rank_zero():
            print(f"Resume training from epoch {start_epoch} in {config.log_path}/ckpt.pth")
    else:
        start_epoch = 0
        
    # start training
    quantizer, decoder = to_ddp_model(rank, quantizer, decoder)
    
    steps_count = 0
    for epoch in range(start_epoch, config.train.num_epochs+1):
        if config.world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # 根据配置决定是否显示进度条
        if check_rank_zero():
            pbar = tqdm(train_loader, total=len(train_loader))
        else:
            pbar = iter(train_loader)
            
        set_train(quantizer, decoder)
        for idx, (batch, _) in enumerate(pbar):
            steps_count += 1

            quantizer_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # 数据预处理（根据数据类型）

            # 音频数据处理：batch是原始音频波形
            raw_audio = batch.to(device)  # [B, channels, samples]
            data = audio_preprocessor(raw_audio)  # 使用AudioPreprocessor
            data = vae_encoder(data)
            data_shuffle = viewer.shuffle(data)
            quant_shuffle = quantizer(data_shuffle)["x_quant"]
            quant = viewer.unshuffle(quant_shuffle)
            data_rec = decoder(quant.detach())
            loss_quant = F.mse_loss(data_shuffle, quant_shuffle)
            loss_decoder = F.mse_loss(data, data_rec)
            loss = loss_quant + loss_decoder
            loss.backward()
            torch.nn.utils.clip_grad_norm_(quantizer.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
            quantizer_optimizer.step()
            decoder_optimizer.step()

            # log
            if config.world_size > 1: dist.barrier()
            if check_rank_zero():
                lr_quantizer = quantizer_optimizer.param_groups[0]["lr"]
                lr_decoder = decoder_optimizer.param_groups[0]["lr"]
                to_log_dict = {
                    "train/loss_quant": loss_quant.item(),
                    "train/loss_decoder": loss_decoder.item(),
                    "lr/lr_quantizer": lr_quantizer,
                    "lr/lr_decoder": lr_decoder,
                }
                for k, v in to_log_dict.items():
                    tb_writer.add_scalar(k, v, steps_count)
                pbar.set_description(
                    f"epoch {epoch}/{config.train.num_epochs}, " +
                    f"batch {idx}/{len(train_loader)}, " +
                    f"lr: {lr_quantizer:.6f}, {lr_decoder:.6f}, " +
                    f"loss-quant={loss_quant.item():.4f}, loss-decoder={loss_decoder.item():.4f}"
                )
                # Write log to file
                logger.info(
                    f"Epoch {epoch} | Batch {idx}/{len(train_loader)} | "
                    f"lr_quantizer={lr_quantizer:.6f}, lr_decoder={lr_decoder:.6f} | "
                    f"loss_quant={loss_quant.item():.4f}, loss_decoder={loss_decoder.item():.4f}"
                )
        # 根据配置的epoch间隔进行验证
        if epoch % config.train.eval_interval == 0:
            set_eval(quantizer, decoder)
            with torch.no_grad():
                loss_list = []
                # 验证阶段也根据配置决定是否显示进度条
                if check_rank_zero():
                    val_pbar = tqdm(val_loader, desc="Validation")
                else:
                    val_pbar = iter(val_loader)
                
                for idx, (batch, _) in enumerate(val_pbar):

                        # 音频验证数据处理
                    raw_audio = batch.to(device)
                    data = audio_preprocessor(raw_audio)  # 使用AudioPreprocessor
                    data = vae_encoder(data)
                    data_shuffle = viewer.shuffle(data)
                    quant_shuffle = quantizer(data_shuffle)["x_quant"]
                    quant = viewer.unshuffle(quant_shuffle)
                    data_rec = decoder(quant.detach())
                    loss_quant = F.mse_loss(data_shuffle, quant_shuffle)
                    loss_decoder = F.mse_loss(data, data_rec)
                    loss = loss_quant + loss_decoder
                    loss_list.append(loss.item())
                loss_mean = sum(loss_list) / len(loss_list)
                if check_rank_zero():
                    logger.info(f"Epoch {epoch} validation loss: {loss_mean:.4f}")
                    tb_writer.add_scalar("val/loss", loss_mean, epoch)

        quantizer.reset()

        # update the learning rate
        quantizer_scheduler.step()
        decoder_scheduler.step()

        # save the model
        if check_rank_zero():
            to_save_dict = {
                "quantizer": quantizer.state_dict(),
                "decoder": decoder.state_dict(),
                "audio_preprocessor": audio_preprocessor.state_dict(), 
                "quantizer_optimizer": quantizer_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "quantizer_scheduler": quantizer_scheduler.state_dict(),
                "decoder_scheduler": decoder_scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                to_save_dict, os.path.join(config.log_path, f"ckpt.pth")
            )
            logger.info(f"Epoch {epoch} saved to {config.log_path}/ckpt.pth")

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
