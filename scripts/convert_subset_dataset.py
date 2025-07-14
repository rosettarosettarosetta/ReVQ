# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

#%%
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from xvq.dataset import Preprocessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
raw_data = torch.load("/path/to/imagenet_train.pth", map_location=device)
ckpt_path = "../ckpt/preprocessor.pth"
preprocessor = Preprocessor(
    input_data_size=[32, 8, 8]
).to(device)
preprocessor.load_state_dict(
    torch.load(ckpt_path, map_location=device)
)
preprocessor.eval()

num_subset = 65536 * 8
indices = torch.randint(0, raw_data.size(0), (num_subset,))
subset_raw_data = raw_data[indices]
subset_data = preprocessor(subset_raw_data)
print(subset_data.size())
torch.save(subset_data, f"/path/to/save/subset.pth")