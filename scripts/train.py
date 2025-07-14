# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

#%%
import os
import math
from tqdm import trange, tqdm
from diffusers import AutoencoderDC
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse
import time, datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from xvq.dataset import get_imagenet_loader, load_preprocessor, load_frozen_vae
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
            backend="nccl", init_method="tcp://localhost:23456",
            rank=config.rank, world_size=config.world_size
        )
    
    # setup dataloaders
    raw_data = torch.load("/path/to/imagenet_train.pth", map_location="cpu", weights_only=True)
    train_set = torch.utils.data.TensorDataset(raw_data)
    val_data = torch.load("/path/to/imagenet_val.pth", map_location="cpu", weights_only=True)
    val_set = torch.utils.data.TensorDataset(val_data)
    if config.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=config.world_size, rank=rank
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=config.data.batch_size,
            sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=config.data.batch_size,
            shuffle=True, num_workers=8, pin_memory=True, drop_last=True
        )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=config.data.batch_size,
        shuffle=False, num_workers=8, pin_memory=True, drop_last=False
    )

    if check_rank_zero():
        OmegaConf.save(config, os.path.join(config.log_path, "config.yaml"))
    
    # load the preprocessor, index-projector, and vae from ckpt files
    preprocessor = load_preprocessor(device=device, config=config.model)
    config.model.data_dim = preprocessor.data_dim
    
    # setup the model
    quantizer, decoder, viewer = setup_models(config.model, device)
    code_bank = torch.load("/path/to/subset.pth", map_location=device, weights_only=True)
    code_bank = viewer.shuffle(code_bank)
    quantizer.prepare_codebook(code_bank, method="random")
    del code_bank
    torch.cuda.empty_cache()

    if config.world_size > 1:
        decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
    optimizer = torch.optim.AdamW(params=[
        {"params": quantizer.parameters(), "lr": config.train.lr, "weight_decay": 0.0},
        {"params": decoder.parameters(), "lr": config.train.lr * 0.05, "weight_decay": 1e-4},
    ])
    gamma = math.pow(0.05, 1 / config.train.num_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=gamma
    )

    # print information
    if check_rank_zero():
        get_param_num = lambda x: sum(p.numel() for p in x.parameters() if p.requires_grad)
        print(f"Quantizer: {get_param_num(quantizer) / 1e6:.2f}M")
        print(f"Decoder: {get_param_num(decoder) / 1e6:.2f}M")
        total_params = get_param_num(quantizer) + get_param_num(decoder)
        print(f"Total params: {total_params / 1e6:.2f}M")
        eq_num_images = int(total_params / 2048)
        print(f"Model equivalent num images: {eq_num_images}")
        print(f"Model Learning rate: {config.train.lr}")    
        print(f"Scheduler gamma = {gamma:.6f}")
        tb_writer = SummaryWriter(config.log_path)
    
    # auto resume
    if os.path.exists(os.path.join(config.log_path, "ckpt.pth")):
        checkpoint = torch.load(os.path.join(config.log_path, "ckpt.pth"), map_location=device, weights_only=True)
        quantizer.load_state_dict(checkpoint["quantizer"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        if check_rank_zero():
            print(f"Resume training from epoch {start_epoch} in {config.log_path}/ckpt.pth")
    else:
        start_epoch = 0
        
    # start training
    quantizer, decoder = to_ddp_model(rank, quantizer, decoder)
    steps_count = 0
    for epoch in range(start_epoch, config.train.num_epochs):
        if check_rank_zero():
            pbar = tqdm(train_loader, total=len(train_loader))
            if config.world_size > 1:
                train_sampler.set_epoch(epoch)
        else:
            pbar = iter(train_loader)
        set_train(quantizer, decoder)
        for idx, (batch,) in enumerate(pbar):
            steps_count += 1

            optimizer.zero_grad()
            data = batch.to(device)
            data = preprocessor(data)
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
            optimizer.step()

            # log
            if config.world_size > 1: dist.barrier()
            if check_rank_zero():
                lr_quantizer = optimizer.param_groups[0]["lr"]
                lr_decoder = optimizer.param_groups[1]["lr"]
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
        quantizer.reset()
        
        set_eval(quantizer, decoder)
        with torch.no_grad():
            loss_list = []
            for idx, (batch,) in enumerate(tqdm(val_loader)):
                data = batch.to(device)
                data = preprocessor(data)
                data_shuffle = viewer.shuffle(data)
                quant_shuffle = quantizer(data_shuffle)["x_quant"]
                quant = viewer.unshuffle(quant_shuffle)
                data_rec = decoder(quant)
                loss = F.mse_loss(data, data_rec)
                loss_list.append(loss.item())
            loss_mean = sum(loss_list) / len(loss_list)
            if check_rank_zero():
                print(f"Epoch {epoch} validation loss: {loss_mean:.4f}")
                tb_writer.add_scalar("val/loss", loss_mean, epoch)

        # update the learning rate
        scheduler.step()

        # save the model
        if check_rank_zero():
            to_save_dict = {
                "quantizer": quantizer.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                to_save_dict, os.path.join(config.log_path, f"ckpt.pth")
            )
            print(f"Epoch {epoch} saved to {config.log_path}/ckpt.pth")

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
