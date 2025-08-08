# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

def find_nearest(x, y):
    """
    Args:
        x (tensor): size (N, D)
        y (tensor): size (M, D)
    """
    dist_mat = torch.cdist(x, y)
    min_dist, indices = torch.min(dist_mat, dim=-1)
    return min_dist, indices

class Quantizer(nn.Module):
    def __init__(self, code_dim: int = 128, num_code: int = 1024,
                 num_group: int = None, tokens_per_data: int = 4,
                 auto_reset: bool = True, reset_ratio: float = 0.2,
                 reset_noise: float = 0.1
        ):
        super(Quantizer, self).__init__()
        self.code_dim = code_dim
        self.num_code = num_code
        self.tokens_per_data = tokens_per_data
        self.num_group = num_group if num_group is not None else tokens_per_data

        self.auto_reset = auto_reset
        self.reset_ratio = reset_ratio
        self.reset_size = max(int(reset_ratio * num_code), 1)
        self.reset_noise = reset_noise

        assert self.tokens_per_data % self.num_group == 0, "tokens_per_data must be divisible by num_group"

        self.find_nearest = torch.vmap(find_nearest, in_dims=0, out_dims=0, chunk_size=2)
        index_offset = torch.tensor(
            [i * self.num_code for i in range(self.num_group)]
        ).long()
        self.register_buffer("index_offset", index_offset)
        self.register_buffer("count", torch.zeros(self.num_group, self.num_code))
        self.embeddings = nn.Parameter(
            torch.randn(self.num_group, self.num_code, self.code_dim)
        )

    def prepare_codebook(self, data, method: str = "random"):
        """
        Args:
            data (tensor): size (N, TPD, D)
        """
        # sample num-code samples from the data
        N, TPD, D = data.size()
        assert TPD == self.tokens_per_data and D == self.code_dim, \
            f"input size {data.size()} does not match the expected size {(N, self.tokens_per_data, self.code_dim)}"
        assert N >= self.num_code, f"num_code {self.num_code} is larger than the data size {N}"

        for i in range(self.num_group):
            if method == "random":
                # sample num_code samples from the data
                indices = torch.randint(0, N, (self.num_code,))
                # sample the codebook from the data
                self.embeddings.data[i] = data[indices][:, i].clone().detach()
            elif method == "kmeans":
                # use kmeans to sample the codebook: sklearn
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.num_code, n_init=10, max_iter=100)
                kmeans.fit(data[:, i].reshape(-1, self.code_dim).cpu().numpy())
                # sample the codebook from the data
                self.embeddings.data[i] = torch.tensor(kmeans.cluster_centers_).to(data.device)
        nn.init.ones_(self.count)

    def init_random_codebook(self):
        """
        Initialize codebook with random values (no data required).
        This method provides pure random initialization without requiring training data.
        """
        # Reset embeddings to random values
        nn.init.normal_(self.embeddings, mean=0.0, std=1.0)
        # Reset count buffer
        nn.init.ones_(self.count)
        print(f"Initialized codebook randomly: {self.num_group} groups × {self.num_code} codes × {self.code_dim} dim")

    def forward(self, x):
        """
        Args:
            x (tensor): size (BS, TPD, D)
        Returns:
            x_quant: size (BS, TPD, D)
            indices: size (BS, TPD)
        """
        BS, TPD, D = x.size()
        assert TPD == self.tokens_per_data and D == self.code_dim, \
            f"input size {x.size()} does not match the expected size {(BS, self.tokens_per_data, self.code_dim)}"

        # compute the indices
        x = x.view(-1, self.num_group, self.code_dim) # (BS*nS, nG, D)
        x = x.permute(1, 0, 2) # (nG, BS*nS, D)
        with torch.no_grad():
            # find the nearest codebook
            dist_min, indices = self.find_nearest(x, self.embeddings)
            
            # update count buffer
            self.count.scatter_add_(dim=-1, index=indices, 
                                    src=torch.ones_like(indices).float())

        # compute the embedding
        indices = indices.permute(1, 0) # (BS*nS, nG)
        indices_offset = indices + self.index_offset
        indices_offset = indices_offset.flatten()
        x_quant = self.embeddings.view(-1, self.code_dim)[indices_offset] # (BS*nS*nG, D)
        x_quant = x_quant.reshape(BS, self.tokens_per_data, self.code_dim)
        indices = indices.reshape(BS, -1)

        return dict(
            x_quant=x_quant,
            indices=indices
        )
    
    def reset_code(self):
        """
        Update the codebook to update the un-activated codebook
        """
        for i in range(self.num_group):
            # sort the codebook via the count numbers
            _, sorted_idx = torch.sort(self.count[i], descending=True)
            reset_size = int(min(self.reset_size, torch.sum(self.count[i] < 0.5)))
            if reset_size > 0.5:
                largest_topk = sorted_idx[:reset_size]
                smallest_topk = sorted_idx[-reset_size:]

                # move the un-activated codebook to the most frequent codebook
                self.embeddings.data[i][smallest_topk] = (
                    self.embeddings.data[i][largest_topk] + 
                    self.reset_noise * torch.randn_like(self.embeddings.data[i][largest_topk]) 
                )

    def reset_count(self):
        # reset the count buffer
        self.count.zero_()
    
    def reset(self):
        if self.auto_reset:
            self.reset_code()
            self.reset_count()
        
    def get_usage(self):
        usage = (self.count > 0.5).sum(dim=-1).float() / self.num_code
        return usage.mean().item()