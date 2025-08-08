# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn

from .mlp import MLPDecoderv2 as MLPDecoder
from .revq import Viewer,Decoder
from .quantizer import Quantizer

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
    decoder_model = Decoder(config.decoder)
    decoder_model = decoder_model.to(device)
    return decoder_model, quantizer, viewer