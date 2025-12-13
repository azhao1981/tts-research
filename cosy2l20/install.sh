sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    libasound2-dev

# pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
UV_TORCH_BACKEND=cu121 uv pip install torch==2.3.1 torchaudio==2.3.1
# (tts-research) root@iZ2ze71xvt9twrugnwh1flZ:~/tts/tts-research# UV_TORCH_BACKEND=cu121 uv pip install torch==2.3.1 torchaudio==2.3.1
# Resolved 23 packages in 2.95s
# ⠋ Preparing packages... (3/15)
# nvidia-cuda-cupti-cu12   ------------------------------ 10.99 MiB/13.46 MiB
# nvidia-cuda-nvrtc-cu12   ------------------------------ 11.09 MiB/22.58 MiB
# nvidia-nvjitlink-cu12    ------------------------------ 11.06 MiB/37.91 MiB
# nvidia-curand-cu12       ------------------------------ 10.97 MiB/53.85 MiB
# nvidia-cufft-cu12        ------------------------------ 11.12 MiB/116.00 MiB
# nvidia-cusolver-cu12     ------------------------------ 11.06 MiB/118.41 MiB
# triton                   ------------------------------ 118.72 MiB/160.27 MiB
# nvidia-nccl-cu12         ------------------------------ 11.03 MiB/168.08 MiB
# nvidia-cusparse-cu12     ------------------------------ 11.16 MiB/186.88 MiB
# nvidia-cublas-cu12       ------------------------------ 11.03 MiB/391.57 MiB
# nvidia-cudnn-cu12        ------------------------------ 11.16 MiB/697.83 MiB
# torch                    ------------------------------ 122.04 MiB/744.79 MiB                                                                  


# 为了修复 CUDA 12 的兼容性问题，必须从特定源安装。
uv pip install onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 因为你的系统 CUDA (12.8) 和 Torch 自带的 CUDA (12.1) 版本号不完全一致，DeepSpeed 可能会报错。我们把它单独拿出来，强制跳过版本检查进行安装。
# 使用清华源加速，并跳过 CUDA 版本严格检查
DS_SKIP_CUDA_CHECK=1 uv pip install deepspeed==0.15.1 -i "https://mirrors.aliyun.com/pypi/simple"

uv pip install -r cosy2l20/requirements_final.txt
# uv pip 或是 pip install 都不行，但下面的可以
python -m pip install tensorrt-cu12==10.0.1 tensorrt-cu12-bindings==10.0.1 tensorrt-cu12-libs==10.0.1

python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
# TensorRT version: 10.0.1

pip uninstall -y transformers torchao
pip install transformers==4.44.2

ssh root@39.105.98.32
git submodule update --init --recursive
# Submodule 'third_party/Matcha-TTS' (https://github.com/shivammehta25/Matcha-TTS.git) registered for path 'third_party/Matcha-TTS'
# Cloning into '/root/tts/CosyVoice/third_party/Matcha-TTS'...
# Submodule path 'third_party/Matcha-TTS': checked out 'dd9105b34bf2be2230f4aa1e4769fb586a3c824e'

http://39.105.98.32:13099
http://39.105.98.32:13099/stream

ssh root@123.57.26.77
http://123.57.26.77:13099
http://123.57.26.77:13099/stream

uv pip install ninja packaging
# 1. 强制使用所有核心进行编译 最好不要用完
export MAX_JOBS=15

# 2. 告诉 Flash Attention 这是一个 Ada Lovelace (L20) 架构
# 虽然通常会自动检测，但显式指定可以避免识别错误
export TORCH_CUDA_ARCH_LIST="8.9"

# 3. 开始安装 (指定 --no-build-isolation 以使用当前环境的 torch)
uv pip install flash-attn --no-build-isolation


import torch
import flash_attn

print(f"Flash Attention Version: {flash_attn.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
# 验证是否能调用（如果没有报错，说明安装成功）
print("✅ Flash Attention is installed and importable!")