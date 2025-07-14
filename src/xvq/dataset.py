# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import webdataset as wds
import math
from tqdm import tqdm

import torch
from torch.utils.data import default_collate
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusers import AutoencoderDC

def compute_stats(num_samples, num_gpus, batch_size_per_gpu, num_workers_per_gpu):
    global_batch_size = batch_size_per_gpu * num_gpus
    num_batches = math.ceil(num_samples / global_batch_size)
    num_worker_batches = math.ceil(num_samples / 
        (global_batch_size * num_workers_per_gpu))
    num_batches = num_worker_batches * num_workers_per_gpu
    num_samples = num_batches * global_batch_size
    return num_batches, num_samples, num_worker_batches

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}
    return _f

def np_to_tensor(x):
    return torch.from_numpy(x).float()

def identity(x):
    return x

def get_train_webdataset(train_shards_path, num_gpus = 1,
                         num_samples: int = 50000,
                         batch_size_per_gpu = 16,
                         num_workers_per_gpu = 8):
    num_batches, num_samples, num_worker_batches = compute_stats(
        num_samples=num_samples, num_gpus=num_gpus,
        batch_size_per_gpu=batch_size_per_gpu, 
        num_workers_per_gpu=num_workers_per_gpu
    )

    # create train dataset and loader
    train_dataset = wds.DataPipeline(
        wds.ResampledShards(train_shards_path),
        wds.tarfile_to_samples(),
        wds.shuffle(bufsize=5000, initial=1000),
        wds.decode(wds.autodecode.basichandlers),
        wds.rename(
            feature="npy", 
            label="cls"
        ),
        wds.map_dict(
            feature=np_to_tensor,
            label=identity,
        ),
        wds.batched(batch_size_per_gpu, partial=False, collation_fn=default_collate),
    ).with_epoch(num_worker_batches)

    train_loader = wds.WebLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers_per_gpu,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader.num_batches = num_batches
    train_loader.num_samples = num_samples
    
    return train_loader

def get_val_webdataset(val_shards_path, 
                       batch_size_per_gpu = 16,
                       num_workers_per_gpu = 8):
    """
    This loader will return the same samples for each node. If you want to have different samples for each node, you need to refer to `https://github.com/webdataset/webdataset?tab=readme-ov-file#the-webdataset-pipeline-api`.
    """
    # create val dataset and loader
    val_dataset = wds.DataPipeline(
        wds.SimpleShardList(val_shards_path),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode(wds.autodecode.basichandlers),
        wds.rename(
            feature="npy", 
            label="cls"
        ),
        wds.map_dict(
            feature=np_to_tensor,
            label=identity,
        ),
        wds.batched(batch_size_per_gpu, partial=True, collation_fn=default_collate),
    )

    val_loader = wds.WebLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers_per_gpu,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return val_loader

# preprocessor
@torch.no_grad()
def load_data_and_stats(device, loader, downsample_ratio: int = 2):
    """
    Load data from the loader and compute the statistics (mean and var) of the data.
    """
    data, current_mean, current_var = None, None, None
    n_samples = 0
    pbar = tqdm(loader)
    for idx, batch in enumerate(pbar):
        sub_data = batch["feature"]
        sub_data = sub_data.to(device) # (BS, C, H, W)
        B, C, H, W = sub_data.size()
        sub_data = sub_data.view(B, -1) # (BS, C*H*W)

        # initiate the mean and var
        if (current_mean is None) or (current_var is None):
            current_mean = torch.zeros(sub_data.shape[1], device=device)
            current_var = torch.zeros(sub_data.shape[1], device=device)

        # update the mean and std
        batch_mean = sub_data.mean(dim=0)
        batch_var = sub_data.var(dim=0, unbiased=False)
        delta = batch_mean - current_mean
        current_mean += delta * B / (n_samples + B)
        current_var += (
            (delta * (batch_mean - current_mean) + batch_var - current_var) 
            * B / (n_samples + B)
        )
        n_samples += B

        # downsample the data
        downsample_sub_data = sub_data[torch.randperm(B)[:B // downsample_ratio]]
        downsample_sub_data = downsample_sub_data.view(-1, C, H, W)
        if data is None:
            data = downsample_sub_data
        else:
            data = torch.cat([data, downsample_sub_data], dim=0)
        pbar.set_description(f"Loading data {idx}: {data.shape}")
    
    return dict(
        data=data,
        mean=current_mean,
        var=current_var,
    )

class Preprocessor(nn.Module):
    def __init__(self, input_data_size):
        super(Preprocessor, self).__init__()
        self.input_data_size = input_data_size
        C, H, W = input_data_size
        self.data_dim = C * H * W
        self.register_buffer("mean", torch.empty(self.data_dim))
        self.register_buffer("std", torch.empty(self.data_dim))
    
    def prepare(self, mean, var):
        self.mean = mean.to(self.mean.device)
        self.std = torch.sqrt(var.to(self.std.device))

    def forward(self, x):
        # normalize: (B, C, H, W) -> (B, C, H, W)
        B, C, H, W = x.size()
        x = x.view(B, self.data_dim) # (B, C*H*W)
        x = (x - self.mean) / self.std
        x = x.view(B, C, H, W).detach()
        return x

    def inverse(self, x):
        # un-normalize: (B, C, H, W) -> (B, C, H, W)
        B = x.size(0)
        C, H, W = self.input_data_size
        x = x.view(B, self.data_dim) # (B, C*H*W)
        x = x * self.std + self.mean
        x = x.view(B, C, H, W)
        return x
    
def load_preprocessor(device, is_eval: bool = True,
    ckpt_path: str = "../ckpt/preprocessor.pth"):
    preprocessor = Preprocessor(
        input_data_size=[32, 8, 8]
    ).to(device)
    preprocessor.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    if is_eval:
        preprocessor.eval()
    return preprocessor

def load_frozen_vae(device, config, is_eval: bool = True):
    vae = AutoencoderDC.from_pretrained(config.vae_path, torch_dtype=torch.float32).to(device)
    if is_eval:
        vae.eval()
    def vae_encode(x):
        with torch.no_grad():
            return vae.encode(x).latent
    def vae_decode(x):
        with torch.no_grad():
            return vae.decode(x).sample
    return vae_encode, vae_decode
    
def get_imagenet_loader(
    root_dir,
    split='train',
    batch_size=64,
    num_workers=8,
    image_size=256,
    shuffle=True
):
    assert split in ['train', 'val'], "split must be 'train' or 'val'"

    split_dir = os.path.join(root_dir, split)

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    dataset = datasets.ImageFolder(split_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataset, loader