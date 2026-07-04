# 1. Create a fresh Python 3.9 environment (highly recommended for TorchDrug stability)
# conda create -n fusionprot python=3.9 -y
# conda activate fusionprot

# 2. Install PyTorch 2.4 compiled for CUDA 12.4
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install PyTorch Geometric dependencies matching Torch 2.4.0 + cu124
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# 4. Install TorchDrug, FusionProt requirements, and Biotite
pip install torchdrug easydict pyyaml atom3d transformers==4.14.1 tokenizers==0.10.3 wandb biotite pandas numpy tqdm scipy