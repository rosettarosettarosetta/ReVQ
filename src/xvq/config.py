# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import argparse
from omegaconf import OmegaConf
import os

def get_config():
    # setup config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--world_size", type=int, default=1)
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config)

    ###############################
    config.name = opt.name
    config.world_size = opt.world_size
    config.train.num_epochs = opt.num_epochs
    config.train.lr = opt.lr
    config.data.batch_size = opt.batch_size
    ###############################
    os.makedirs(config.log_path, exist_ok=True)
    return config