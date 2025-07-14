# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------
import torch
from diffusers import AutoencoderDC
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import webdataset as wds
import typer

from torchvision.datasets import ImageFolder
from torchvision import transforms

app = typer.Typer()

@app.command()
def convert(save_split: str = "val", root: str = "/path/to/imagenet"):
    BATCH_SIZE = 64
    device = torch.device("cuda:0")
    data_path = os.path.join(root, save_split)
    ckpt_path = "/path/to/dcae"
    save_path = "/path/to/save/converted_imagenet"
    save_path = os.path.join(save_path, f"{save_split}")
    maxcount = {
        "train": 4000,
        "val": 1000
    }[save_split]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=8)
    vae = AutoencoderDC.from_pretrained(ckpt_path, torch_dtype=torch.float32).to(device)

    # start the convert
    os.makedirs(save_path, exist_ok=True)
    opat = os.path.join(save_path, f"imagenet_{save_split}_" + "%06d.tar")
    wds_writer = wds.ShardWriter(pattern=opat, maxcount=maxcount)
    total_len = len(dataloader)
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            x = x * 2 - 1
            lat = vae.encode(x).latent
            pbar.set_description(f"{i}/{total_len}")

            for j in range(lat.shape[0]):
                sub_index = i * BATCH_SIZE + j
                sub_data = lat[j].detach().cpu().numpy()
                sub_label = int(y[j].item())
                wds_writer.write({
                    "__key__": f"{sub_index:08d}",
                    "cls": sub_label,
                    "npy": sub_data
                })
    wds_writer.close()
    print(f"Save to {save_path}")

if __name__ == "__main__":
    app()