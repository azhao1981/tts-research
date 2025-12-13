import torch
import flash_attn

print(f"Flash Attention Version: {flash_attn.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
# 验证是否能调用（如果没有报错，说明安装成功）
print("✅ Flash Attention is installed and importable!")