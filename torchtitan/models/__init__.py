# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama import llama2_configs, llama3_configs, llama3_diffusion_configs, Transformer, DiffusionTransformer

models_config = {
    "llama2": llama2_configs,
    "llama3": llama3_configs,
    "llama3_diffusion": llama3_diffusion_configs,
}

model_name_to_cls = {"llama2": Transformer, "llama3": Transformer, "llama3_diffusion": DiffusionTransformer}

model_name_to_tokenizer = {
    "llama2": "sentencepiece",
    "llama3": "tiktoken",
    "llama3_diffusion": "tiktoken",
}
