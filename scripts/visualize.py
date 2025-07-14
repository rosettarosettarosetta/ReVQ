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
from xvq.utils import seed_everything, check_rank_zero, to_ddp_model, set_eval
from xvq.models import setup_models
import matplotlib.pyplot as plt

def reconstruct_sample(device, dataset, vae_encode, vae_decode, vq_infer, index):
    img, _ = dataset[index]
    img = img.unsqueeze(0).to(device)

    ori = torch.clone(img)
    x_hat = img * 2 - 1
    lat = vae_encode(x_hat)
    rec = vq_infer(lat)
    rec = vae_decode(rec)
    rec = (rec + 1) / 2

    return ori.squeeze(0).cpu(), rec.squeeze(0).cpu()

def save_single_image(tensor_img, log_path, title, epoch, idx, rec=False):
    img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img.clip(0, 1)  

    plt.figure(figsize=(2, 2), dpi=200)
    plt.imshow(img)
    plt.axis("off")
    name = "revq" if rec else "gt"
    save_path = os.path.join(log_path, f"{name}_{idx:02d}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_config():
    # setup config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/path/to/pretrained_model/config.yaml")
    parser.add_argument("--name", type=str, default="/path/to/pretrained_model")
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config)

    ###############################
    config.name = opt.name
    config.world_size = 1
    ###############################
    os.makedirs(config.log_path, exist_ok=True)
    return config

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
    
    val_dataset, _ = get_imagenet_loader(
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
    
    # auto resume
    if os.path.exists(os.path.join(config.log_path, "ckpt.pth")):
        checkpoint = torch.load(os.path.join(config.log_path, "ckpt.pth"), map_location=device, weights_only=True)
        quantizer.load_state_dict(checkpoint["quantizer"])
        decoder.load_state_dict(checkpoint["decoder"])
        if check_rank_zero():
            print(f"loading from {config.log_path}/ckpt.pth")
    else:
        start_epoch = 0
        
    # start training
    quantizer, decoder = to_ddp_model(rank, quantizer, decoder)
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

    # visualize
    idx = torch.randint(0, len(val_dataset), (20,)).tolist()
    rec_log_path = os.path.join(config.workspace, "rec_log_path")
    ori_log_path = os.path.join(config.workspace, "ori_log_path")

    os.makedirs(rec_log_path, exist_ok=True)
    os.makedirs(ori_log_path, exist_ok=True)
    for i in idx:
        ori, rec = reconstruct_sample(device, val_dataset, vae_encode, vae_decode, vq_infer, i)
        save_single_image(ori, ori_log_path, "original", 0, i, rec=False)
        save_single_image(rec, rec_log_path, "reconstructed", 0, i, rec=True)

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

# %%
