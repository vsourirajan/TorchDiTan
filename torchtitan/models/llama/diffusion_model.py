# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.



import torch
import torch.nn as nn
from typing import Optional

from torchtitan.models.norms import build_norm
from torchtitan.models.llama.diffusion_config import DiffusionModelArgs
from torchtitan.models.llama.diffusion_blocks import (
    PatchEmbed,
    LabelEmbedder,
    TimestepEmbedder,
    DiffusionTransformerBlock,
    DiffusionTransformerBlockWithContext
)

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def precompute_freqs_cis_2d(dim: int, end_x: int, end_y: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    out = torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    return out


# Move mode_to_block here from config
mode_to_block = {
    "context": DiffusionTransformerBlockWithContext,
    "adaLN": DiffusionTransformerBlock
}

class DiffusionTransformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, 
                 model_args: DiffusionModelArgs,
                 label_dropout_prob: float = 0.05,
                 ):
        
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.patch_size = model_args.patch_size

        self.input_channels = self.model_args.input_channels
        self.input_image_size = model_args.image_size
        self.patch_size = model_args.patch_size
        self.num_classes = model_args.num_classes
        self.condition_mode = model_args.condition_mode

        self.num_x_patches = self.input_image_size[0] // self.patch_size
        self.num_y_patches = self.input_image_size[1] // self.patch_size

        print("[MODEL] num_x_patches: ", self.num_x_patches, 'num_y_patches: ', self.num_y_patches)

        self.x_embedder = PatchEmbed(self.input_image_size, (self.patch_size, self.patch_size), self.input_channels, model_args.dim, bias=True)
        self.y_embedder = LabelEmbedder(self.num_classes, model_args.dim, dropout_prob=label_dropout_prob)
        self.t_embedder = TimestepEmbedder(model_args.dim)

        assert model_args.condition_mode in ["adaLN", "context"]


        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        dit_block = mode_to_block[model_args.condition_mode]
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = dit_block(layer_id, model_args)

        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, self.patch_size * self.patch_size * self.input_channels)
        self.init_weights()

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
            # print("[MODEL]freqs cis shape: ", self.freqs_cis.shape)

        # if self.tok_embeddings is not None:
        #     nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d) (from DiT paper)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02) #(from DiT paper)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02) #(from DiT paper)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.output is not None:
            # nn.init.trunc_normal_(
            #     self.output.weight,
            #     mean=0.0,
            #     std=final_out_std,
            #     a=-cutoff_factor * final_out_std,
            #     b=cutoff_factor * final_out_std,
            # )

            # the output layers of diffusion models are initialized to 0
            nn.init.constant_(self.output.weight, 0)
            nn.init.constant_(self.output.bias, 0)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis_2d(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            end_x=self.num_x_patches,
            end_y=self.num_y_patches,
            theta=self.model_args.rope_theta,
        )
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.input_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, data_entries: dict):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (dict): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        # h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        
        '''
        data_entries
        {
            "input":Tensor[B, C, H, W], (float)
            "class": Tensor[B, ] (integer)
            "time" Tensor[B, ] (float)
            "param_dtype": torch.dtype
        }
        '''
        
        input = data_entries["input"]
        class_idx = data_entries["class_idx"]
        time = data_entries["time"]
        force_drop_ids = data_entries.get("force_drop_ids", None)

        # print("[MODEL] input shape: ", input.shape)

        #param_dtype = data_entries["param_dtype"]

        h = self.x_embedder(input) # B, num_patches, dim

        # print("[MODEL] h shape: ", h.shape)

        t_embedding = self.t_embedder(time)
        y_embedding = self.y_embedder(class_idx, True, force_drop_ids)

        if self.condition_mode == "context":
            h = torch.cat([h, t_embedding.unsqueeze(1), y_embedding.unsqueeze(1)], dim=1) # B, num_patches + 2, dim

        c = t_embedding + y_embedding #unused for context mode, not many flops so it should be fine

        for layer in self.layers.values():
            h = layer(h, c, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h

        #cut off last 2 dimensions to get (B, num_patches, dim) tensor back
        output = output[:, :-2, :]  if self.condition_mode == "context" else output
        output = self.unpatchify(output)
        return output

    @classmethod
    def from_model_args(cls, model_args: DiffusionModelArgs) -> "DiffusionTransformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
