apt-get update
apt-get install -y tmux

pip install --upgrade torch torchvision wandb matplotlib==3.9.2 tqdm scipy torchdata sentencepiece tiktoken datasets tensorboard blobfile torchvision safetensors gpustat torchao
git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git
cd Cosmos-Tokenizer
pip install -e .
cd ..