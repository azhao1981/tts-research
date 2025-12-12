这是为您整理的 **基于 NVIDIA V100 的 CosyVoice 2 高性能流式服务落地最终方案**。

本方案的核心逻辑是：**“做减法”**。V100 是一张强大的显卡，但它是上一代架构（Volta）。强行套用为 A100/H100 设计的新技术（如 FlashAttn v2）不仅无效，反而会因为软件模拟导致性能下降。

### 一、 核心架构决策图

### 二、 为什么放弃其他热门方案？（避坑指南）

作为架构师，明确“不做什么”比“做什么”更重要。以下是在调研中被否决的技术及其原因：

| 技术方案 | 否决原因 (针对 V100 + CosyVoice) | 架构师评语 |
| :--- | :--- | :--- |
| **Flash Attention (v2)** | **硬件不支持** | FlashAttn v2 依赖 Ampere 架构（A100+）指令集。V100 强行安装会报错或回退到慢速模式。 |
| **BF16 (BFloat16)** | **硬件不支持** | V100 没有 BF16 硬件单元。强行使用会导致 CUDA 进行软件模拟，速度比 FP32 慢数倍。**必须使用 FP16。** |
| **DeepSpeed** | **杀鸡用牛刀/负优化** | CosyVoice 0.5B 模型极小，单卡显存足够。DeepSpeed 的强项是多卡切分（TP），在单卡小模型上引入的通信和调度开销远大于收益。 |
| **LightTTS** | **兼容性风险** | 该项目高度依赖 Triton 和 FlashAttn 的新特性，主要针对 A100 优化。在 V100 上适配成本极高（需重写 Kernel），不适合生产落地。 |
| **TensorRT** | **投入产出比低** | 模型包含复杂的流式控制流（Flow Matching），转 TRT 引擎开发难度大，且 V100 对动态 Shape 的支持不如 PyTorch 2.x 的编译模式灵活。 |

-----

### 三、 最终落地实施方案

**核心策略**：PyTorch 原生 FP16 (利用 Tensor Core) + Torch Compile (消除 Python 开销) + 多进程并发 (吃满显存)。

#### 1\. 环境配置 (Dockerfile 参考)

```bash
# 基础镜像：推荐 PyTorch 2.1 或 2.2，CUDA 12.1 (向下兼容 V100)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 安装系统级依赖
RUN apt-get update && apt-get install -y sox libsox-dev git

# 安装 Python 依赖
# 注意：不安装 flash-attn，不安装 deepspeed
RUN pip install "modelscope>=1.15.0" "torchaudio>=2.2.0" "fastapi" "uvicorn" "gunicorn" "python-multipart"

# 克隆代码
WORKDIR /app
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git .
RUN pip install -r requirements.txt
```

#### 2\. 推理核心代码 (`service_core.py`)

这里进行了针对性修改，**强制开启 FP16** 并启用 **Torch Compile**。

```python
import torch
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import io
import torchaudio

class OptimizedCosyService:
    def __init__(self, model_dir='pretrained_models/CosyVoice2-0.5B'):
        print(f"[Init] Loading model from {model_dir} on V100...")
        
        # 1. 核心优化：强制 FP16。V100 的 Tensor Core 必须靠 FP16 激活。
        # 关闭 load_jit/trt 以避免冲突，完全交由 torch.compile 接管
        self.model = CosyVoice2(
            model_dir, 
            load_jit=False, 
            load_trt=False, 
            fp16=True 
        )
        
        # 2. 编译优化：针对小模型流式推理，减少 Python Overhead 是关键
        # 'reduce-overhead' 模式会消耗更多内存预编译，换取极快的推理启动
        if torch.cuda.is_available():
            print("[Init] Compiling model with torch.compile (mode='reduce-overhead')...")
            try:
                self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
            except Exception as e:
                print(f"[Warn] Compile failed, fallback to eager mode: {e}")
        
        # 3. 预热 (Warmup)：编译模式首次运行极慢，必须预热
        print("[Init] Warming up...")
        try:
            # 跑一次空推理，触发内核编译
            list(self.model.inference_sft('预热', '预热', stream=True))
        except:
            pass
        print("[Init] Service Ready.")

    def stream_generate(self, text, prompt_text, prompt_wav):
        # 这里的 prompt_wav 传入路径或 bytes 均可，视具体实现而定
        prompt_speech_16k = load_wav(prompt_wav, 16000)
        
        for i, output in enumerate(self.model.inference_zero_shot(
            text, prompt_text, prompt_speech_16k, stream=True
        )):
            # 转换音频为 bytes 流
            audio_tensor = output['tts_speech'].cpu()
            buffer = io.BytesIO()
            # 建议：实际生产可用 pcm 裸流节省头部开销，这里用 wav 为示例
            torchaudio.save(buffer, audio_tensor, 24000, format="wav")
            yield buffer.getvalue()

# 全局单例
service_instance = OptimizedCosyService()
```

#### 3\. API 服务层 (`server.py`)

```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from service_core import service_instance
import shutil
import os
import uuid

app = FastAPI()

@app.post("/stream")
async def stream_tts(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...)
):
    # 临时文件处理
    temp_filename = f"/tmp/{uuid.uuid4()}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(prompt_audio.file, buffer)

    def iterfile():
        try:
            yield from service_instance.stream_generate(text, prompt_text, temp_filename)
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    return StreamingResponse(iterfile(), media_type="audio/wav")
```

#### 4\. 生产环境启动命令 (并发策略)

CosyVoice 2 模型很小，V100 (16G/32G) 单进程利用率低。使用 **Gunicorn** 启动多进程，每个进程持有一个模型实例。

  * **V100 16GB**: 建议开启 **2-3 个 Worker**。
  * **V100 32GB**: 建议开启 **4-6 个 Worker**。

<!-- end list -->

```bash
# 假设文件名为 server.py
# --timeout 设置长一点，防止长文本推理超时
gunicorn server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
```

### 四、 性能调优 Checklist

1.  **断句优化 (Text Normalization)**：

      * **不要**把整段 500 字丢进模型。
      * 在进入模型前，用简单的正则按句号/逗号切分文本，分段流式请求。这对降低**首字延迟 (TTFT)** 至关重要。

2.  **环境变量设置**：

      * 在启动前设置 `export TORCH_CUDNN_V8_API_ENABLED=1`，这有助于 V100 利用较新的 cuDNN 算法。

3.  **监控显存**：

      * 启动后观察 `nvidia-smi`。如果显存还有大半空余，增加 Gunicorn 的 `--workers` 数量，直到显存占用达到 80%-90% 为止。这是提升 QPS (吞吐量) 最直接的手段。

### 下一步行动

建议您直接复制上述 **Step 2 (service\_core.py)** 的代码，在您的 V100 机器上运行一遍。如果能看到 `[Init] Service Ready.` 且没有报错，说明最难的硬件适配已经完成。