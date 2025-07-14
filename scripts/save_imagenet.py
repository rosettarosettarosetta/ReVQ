# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from tqdm import tqdm
import torch
from xvq.dataset import get_val_webdataset

data_loader = get_val_webdataset(
    val_shards_path="/path/to/imagenet_train_{000000..000320}.tar",
    batch_size_per_gpu=4096,
    num_workers_per_gpu=8
)

data_list = []
pbar = tqdm(data_loader)
for data in pbar:
    data = data["feature"]
    data_list.append(data)
data = torch.cat(data_list, dim=0)
print(data.shape)
torch.save(data, "/path/to/imagenet_train.pth")