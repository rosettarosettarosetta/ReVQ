# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn

from .dcae import DCAEDecoder
from .mlp import MLPDecoderv2 as MLPDecoder

from .quantizer import Quantizer

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

def get_params(config, key: str = "TYPE"):
    name = getattr(config, key)
    params = OmegaConf.to_container(config, resolve=True)
    params.pop(key)
    return name, params

def setup_models(config, device):
    # setup quantizer
    config_quantizer = config.quantizer
    name, params = get_params(config_quantizer)
    data_dim = np.prod(config.input_data_size)
    code_dim = int(data_dim / params["tokens_per_data"])
    params["code_dim"] = code_dim
    quantizer = Quantizer(**params)
    quantizer = quantizer.to(device)
    viewer = Viewer(config.input_data_size, code_dim)
    
    # setup decoder
    config_decoder = config.decoder
    name, params = get_params(config_decoder)
    if name == "identity":
        decoder = nn.Identity()
    elif name == "mlp":
        decoder = MLPDecoder(**params)
    elif name == "dc_ae":
        decoder = DCAEDecoder(**params)
    else:
        raise NotImplementedError(f"Decoder {name} not implemented")
    decoder = decoder.to(device)

    return quantizer, decoder, viewer