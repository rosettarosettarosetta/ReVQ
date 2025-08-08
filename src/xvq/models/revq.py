# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

# Convert a Pytorch model to a Hugging Face model

import torch.nn as nn
import torch
import os

from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from xvq.models import DCAEDecoder
from xvq.models import Quantizer

class Viewer:
    def __init__(self, input_size, code_dim):
        self.input_size = input_size
        self.code_dim = code_dim

    def shuffle(self, x: torch.Tensor):
        # (B, C, H, W) -> (B, T, D)
        B, C, H, W = x.size()
        x = x.view(B, -1, self.code_dim)
        return x

    def unshuffle(self, x: torch.Tensor):
        # (B, T, D) -> (B, C, H, W)
        B, T, D = x.size()
        C, H, W = self.input_size
        x = x.view(B, C, H, W)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, x):
        return x

# class ReVQ(PyTorchModelHubMixin, nn.Module):
#     @classmethod
#     def _from_pretrained(cls, model_id: str, **kwargs):
#         print(f"Loading ReVQ model from {model_id}...")
#         config_path = hf_hub_download(repo_id=model_id, filename="512T_NC=16384.yaml")
#         ckpt_path = hf_hub_download(repo_id=model_id, filename="ckpt.pth")

#         full_cfg = OmegaConf.load(config_path)
#         model_cfg = full_cfg.get("model", {})
#         decoder_config = model_cfg.get("decoder", {})
#         quantize_config = model_cfg.get("quantizer", {})

#         model = cls(decoder=decoder_config, 
#                     quantize=quantize_config, 
#                     ckpt_path=ckpt_path,
#                     )

#         return model
    
#     def __init__(self, 
#             decoder: dict = {},
#             quantize: dict = {},
#             ckpt_path: str = None,
#         ):
#         super(ReVQ, self).__init__()
#         self.decoder = DCAEDecoder(**decoder)
#         self.quantizer = self.setup_quantizer(quantize)
#         self.viewer = Viewer(input_size=[32, 8, 8], code_dim=4)

#         if ckpt_path is not None and os.path.exists(ckpt_path):
#             # Load the model checkpoint
#             checkpoint = torch.load(ckpt_path, map_location="cpu")
#             self.decoder.load_state_dict(checkpoint["decoder"])
#             self.quantizer.load_state_dict(checkpoint["quantizer"])
    
#     def setup_quantizer(self, quantizer_config):
#         quantizer = Quantizer(**quantizer_config)
#         return quantizer
    
#     def quantize(self, x):
#         x_shuffle = self.viewer.shuffle(x)
#         quant_shuffle = self.quantizer(x_shuffle)["x_quant"]
#         quant = self.viewer.unshuffle(quant_shuffle)
#         return quant

#     def decode(self, x):
#         x_rec = self.decoder(x)
#         return x_rec

#     def forward(self, x):
#         quant = self.quantize(x)
#         rec = self.decode(quant)
#         return quant, rec