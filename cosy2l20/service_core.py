import torch
import sys

# 1. 导入必要的库
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
import io
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import os
import types

# 2. 显式开启 Flash Attention 优化
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# 3. 引入 torchao
try:
    from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("[Warn] torchao not installed. Quantization will be skipped.")

FINETURED_COSYVOICE2_MODEL_PATH = os.getenv("FINETURED_COSYVOICE2_MODEL_PATH", "pretrained_models/CosyVoice2-0.5B-finetune")

class OptimizedCosyService:
    def __init__(self, model_dir=FINETURED_COSYVOICE2_MODEL_PATH, quantization_mode='none'):
        print(f"[Init] Loading model from {model_dir} on L20...")
        
        # 1. 加载模型 (FP32加载，避免FP16溢出)
        self.model = CosyVoice2(
            model_dir, 
            load_jit=False, 
            load_trt=False, 
            fp16=False,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 2. ✅ 手动将核心模型转为 Bfloat16
        if torch.cuda.is_available():
            print("[Init] Converting sub-modules to Bfloat16...")
            if hasattr(self.model.model, 'llm'):
                self.model.model.llm.to(dtype=torch.bfloat16)
            if hasattr(self.model.model, 'flow'):
                self.model.model.flow.to(dtype=torch.bfloat16)
            if hasattr(self.model.model, 'hift'):
                self.model.model.hift.to(dtype=torch.bfloat16)
            print("✅ All modules converted to Bfloat16.")

        # 3. ✅ 应用 Monkey Patch (修复前端崩溃的关键)
        self._apply_frontend_patch()

        # 4. 量化逻辑 (torchao)
        if TORCHAO_AVAILABLE and quantization_mode in ['int8', 'int4']:
            print(f"[Init] Applying {quantization_mode} quantization to LLM...")
            q_config = int4_weight_only() if quantization_mode == 'int4' else int8_weight_only()
            try:
                quantize_(self.model.model.llm, q_config)
                print(f"✅ LLM module quantized to {quantization_mode}.")
            except Exception as e:
                print(f"[Error] Quantization failed: {e}")

        # 5. 编译优化
        if torch.cuda.is_available():
            print("[Init] Compiling sub-modules...")
            try:
                self.model.model.llm = torch.compile(self.model.model.llm, mode="reduce-overhead", fullgraph=False)
                self.model.model.flow = torch.compile(self.model.model.flow, mode="reduce-overhead", fullgraph=False)
                print("✅ Torch Compile applied.")
            except Exception as e:
                print(f"[Warn] Compile failed: {e}")
        
        # 6. 预热 (Warmup)
        print("[Init] Warming up...")
        try:
            # 构造 FP32 输入 (Patch 会保证它在前端不被乱转)
            dummy_wav = torch.randn(16000, device=self.device)
            # 使用 Autocast 桥接 (Patch 负责局部禁用)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                list(self.model.inference_zero_shot(
                    tts_text='预热', 
                    prompt_text='预热', 
                    prompt_speech_16k=dummy_wav, 
                    stream=True
                ))
            print("✅ Warmup done.")
        except Exception as e:
            print(f"[Warn] Warmup skipped or failed: {e}")
            import traceback
            traceback.print_exc()
            
        print("[Init] Service Ready.")

    def _apply_frontend_patch(self):
        """
        核心修复：给 Frontend 打补丁。
        CosyVoice 的前端 Resampler 会动态生成 FP32 kernel。
        如果外层开启 Autocast，输入会被强转 BF16，导致 conv1d 类型不匹配崩溃。
        此 Patch 强制前端在 Autocast 关闭 (FP32) 模式下运行。
        """
        print("[Init] Applying Frontend Patch (Disable Autocast for Resampler)...")
        
        # 1. 拿到原始绑定的方法
        original_frontend_func = self.model.frontend.frontend_zero_shot

        # 2. 定义 Wrapper
        def patched_frontend_zero_shot(*args, **kwargs):
            # 强制退出 autocast，确保 Resampler 接收 FP32 且运行在 FP32
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', enabled=False):
                    return original_frontend_func(*args, **kwargs)
            else:
                return original_frontend_func(*args, **kwargs)

        # 3. 替换实例方法
        self.model.frontend.frontend_zero_shot = patched_frontend_zero_shot
        print("✅ Frontend Patch applied.")

    def safe_load_prompt(self, wav_path, target_sr=16000):
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform

    def stream_generate(self, text, prompt_text, prompt_wav):
        # 1. 加载音频 (CPU Float32)
        prompt_speech_16k = self.safe_load_prompt(prompt_wav, 16000)
        
        # 2. 移动到 CUDA，保持 Float32！
        if torch.cuda.is_available():
            prompt_speech_16k = prompt_speech_16k.to(self.device)
        
        # 3. 推理 (Autocast 开启)
        #    流程：Autocast ON -> Patch(Autocast OFF) -> Frontend(FP32) -> Patch End -> Autocast ON -> LLM(BF16)
        try:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                generator = self.model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=prompt_text,
                    prompt_speech_16k=prompt_speech_16k, 
                    stream=True
                )
                
                for i, output in enumerate(generator):
                    audio_tensor = output['tts_speech'].cpu()
                    audio_numpy = audio_tensor.squeeze().float().numpy()
                    
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_numpy, 24000, format='RAW', subtype='PCM_16')            
                    yield buffer.getvalue()
                
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            import traceback
            traceback.print_exc()
            yield b""

service_instance = OptimizedCosyService()