# 如果需要用这个来加
import os
import warnings

# 1. 屏蔽 ONNX Runtime 的啰嗦日志 (必须在 import onnxruntime 之前设置)
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

# 2. 屏蔽 Python 库的弃用警告
# 忽略 pkg_resources, diffusers, torch.nn.utils.weight_norm 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 针对特定字符串的模糊匹配屏蔽
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear is deprecated.*")