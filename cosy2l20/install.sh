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