#!/usr/bin/env bash
# =============================================================================
# Setup Dedicated 'qwen_env' for CUDA 12.2 / 12.1
# =============================================================================
set -e

ENV_NAME="qwen_env"

echo "============================================================================="
echo "🚀 Creating dedicated Conda Environment: ${ENV_NAME} (CUDA 12.2 compatible)"
echo "============================================================================="

# 1. Create clean Python 3.11 environment
conda create -n ${ENV_NAME} python=3.11 -y

# 2. Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# 3. Install PyTorch with CUDA 12.1/12.2 support
echo "[*] Installing PyTorch with CUDA 12.x support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install latest Transformers & Sentence-Transformers (Qwen3 native support)
echo "[*] Installing latest Transformers, Sentence-Transformers & IR dependencies..."
pip install "transformers>=5.0.0" sentence-transformers accelerate rank-bm25 pandas scipy tabulate python-dotenv tqdm huggingface_hub

# 5. Sanity check Qwen3 and CUDA
echo "[*] Verifying Qwen3 and CUDA on GPU..."
python -c "
import torch, transformers, sentence_transformers
print(' PyTorch Version       :', torch.__version__)
print(' CUDA Available        :', torch.cuda.is_available())
if torch.cuda.is_available():
    print(' Device Name           :', torch.cuda.get_device_name(0))
print(' Transformers Version  :', transformers.__version__)
print(' SentenceTransformers  :', sentence_transformers.__version__)
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
print(' Qwen3 AutoConfig OK   : model_type =', c.model_type)
"

echo "============================================================================="
echo " [✓] Environment '${ENV_NAME}' is ready!"
echo " To run Stage 2 with Qwen3 on gpu04:"
echo "   conda activate ${ENV_NAME}"
echo "   python run_stage2_gpu_retrieval.py --device cuda --top_k 10 --bootstraps 10000 --include_4b"
echo "============================================================================="
