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

from xvq.dataset import get_imagenet_loader, load_preprocessor, load_frozen_vae
from xvq.utils import reconstruct_sample, seed_everything, check_rank_zero, to_ddp_model, set_train, set_eval
from xvq.models import setup_models
from xvq.evaluator import VQGANEvaluator
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
    
    _, val_loader = get_imagenet_loader(
        root_dir="/path/to/imagenet",
        batch_size=64,
        num_workers=8,
        image_size=256,
        split="val",
        shuffle=False
    )
    
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
    
    if check_rank_zero():
        get_param_num = lambda x: sum(p.numel() for p in x.parameters() if p.requires_grad)
        print(f"Quantizer: {get_param_num(quantizer) / 1e6:.2f}M")
        print(f"Decoder: {get_param_num(decoder) / 1e6:.2f}M")
        total_params = get_param_num(quantizer) + get_param_num(decoder)
        print(f"Total params: {total_params / 1e6:.2f}M")
    
    # auto resume

    if os.path.exists(os.path.join(config.log_path, f"{config.name}_ckpt.pth")):
        checkpoint = torch.load(os.path.join(config.log_path, f"{config.name}_ckpt.pth"), map_location=device, weights_only=True)
        quantizer.load_state_dict(checkpoint["quantizer"])
        decoder.load_state_dict(checkpoint["decoder"])
        if check_rank_zero():
            print(f"loading from {config.log_path}/ckpt.pth")
    else:
        start_epoch = 0
        
    # start training
    quantizer, decoder = to_ddp_model(rank, quantizer, decoder)
    evaluator = VQGANEvaluator(device=device,
                                 enable_rfid=True,
                                 enable_inception_score=True,
                                 enable_codebook_entropy_measure=False,
                                 enable_codebook_usage_measure=False,
                                 enable_lpips=True,
                                 enable_ssim_psnr=True
    )
    vae_encode, vae_decode = load_frozen_vae(device=device, config=config.model)

    def vq_infer(x):
        data = x.contiguous()
        data = preprocessor(data)
        data_shuffle = viewer.shuffle(data)
        quant_shuffle = quantizer(data_shuffle)["x_quant"]
        quant = viewer.unshuffle(quant_shuffle)
        data_rec = quant
        data_rec = decoder(data_rec)
        data_rec = data_rec.contiguous()
        data_rec = preprocessor.inverse(data_rec)

        return data_rec
        
    set_eval(quantizer, decoder)
    pbar = tqdm(val_loader)     
    with torch.no_grad():
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            ori = torch.clone(x)
            x_hat = x * 2 - 1
            lat = vae_encode(x_hat)
            rec = vq_infer(lat)
            rec = vae_decode(rec)
            rec = (rec + 1) / 2
            
            ori = torch.clamp(ori, 0.0, 1.0)
            rec = torch.clamp(rec, 0.0, 1.0)
            rec = torch.round(rec * 255.0) / 255.0
            evaluator.update(ori, rec)
            
            pbar.set_description(f"{i}/{len(val_loader)}")

            if (i+1) % 50 == 0:
                for key, value in evaluator.result().items():
                    print(f"{key}: {value}")

    for key, value in evaluator.result().items():
        print(f"{key}: {value}")

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
