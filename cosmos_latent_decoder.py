import os
import torch
import torch.nn as nn
from safetensors.torch import safe_open
from PIL import Image
from cosmos_tokenizer.image_lib import ImageTokenizer
from huggingface_hub import snapshot_download
from typing import Optional, Union

class CosmosDecoder:
    def __init__(
        self,
        is_continuous: bool = True,
        device: str = 'cuda',
        checkpoint_dir: str = './pretrained_ckpts'
    ):
        self.is_continuous = is_continuous
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model_name = "Cosmos-Tokenizer-CI8x8" if is_continuous else "Cosmos-Tokenizer-DI8x8"
        
        # Ensure checkpoint exists
        self._download_checkpoint_if_needed()
        
        # Initialize decoder
        self.decoder = ImageTokenizer(
            checkpoint_dec=f"{self.checkpoint_dir}/{self.model_name}/decoder.jit"
        ).to(self.device)

    def _download_checkpoint_if_needed(self):
        """Downloads the checkpoint if it doesn't exist locally."""
        if not os.path.exists(f"{self.checkpoint_dir}/{self.model_name}"):
            os.makedirs(f"{self.checkpoint_dir}/{self.model_name}", exist_ok=True)
            print(f"Downloading {self.model_name}...")
            hf_repo = f"nvidia/{self.model_name}"
            snapshot_download(repo_id=hf_repo, local_dir=f"{self.checkpoint_dir}/{self.model_name}")

    @torch.no_grad()
    def decode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to images.
        
        Args:
            data: Input tensor of shape [B, 16, 32, 32] for continuous or [B, 32, 32] for discrete
        
        Returns:
            Decoded image tensor of shape [B, 3, H, W]
        """
        if self.is_continuous:
            data = data.to(self.device)
        else:
            data = data.to(self.device).long()
            
        reconstructed = self.decoder.decode(data)
        return reconstructed

if __name__ == "__main__":
    main()