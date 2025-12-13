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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 加载模型 (FP32加载)
        self.model = CosyVoice2(
            model_dir, 
            load_jit=False, 
            load_trt=False, 
            fp16=False,
        )
        
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

        # 3. ✅ 应用 Monkey Patch 2.0 (强制前端走 CPU)
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
                # 启用 dynamic=True 防止变长输入导致重编译
                self.model.model.llm = torch.compile(self.model.model.llm, mode="reduce-overhead", fullgraph=False, dynamic=True)
                self.model.model.flow = torch.compile(self.model.model.flow, mode="reduce-overhead", fullgraph=False, dynamic=True)
                print("✅ Torch Compile applied.")
            except Exception as e:
                print(f"[Warn] Compile failed: {e}")
        
        # 6. 预热 (Warmup) - 修正版
        print("[Init] Warming up...")
        try:
            # ✅ 修复点：添加维度 (1, 16000) 而不是 (16000)
            dummy_wav = torch.randn(1, 16000, device=self.device)
            
            # 使用 Autocast 桥接
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
        Monkey Patch 2.0:
        前端（音频处理）强制在 CPU 上以 FP32 运行，彻底解决设备和类型冲突。
        处理完后，自动把结果搬回 GPU 给模型用。
        """
        print("[Init] Applying Frontend Patch (Force CPU & Disable Autocast)...")
        
        # 拿到原始方法
        original_frontend_func = self.model.frontend.frontend_zero_shot

        # 定义补丁函数
        def patched_frontend_zero_shot(i, prompt_text, prompt_speech_16k, sample_rate, zero_shot_spk_id):
            # 1. 输入回退到 CPU
            if isinstance(prompt_speech_16k, torch.Tensor) and prompt_speech_16k.is_cuda:
                prompt_speech_16k = prompt_speech_16k.cpu()

            # 2. 强制禁用 Autocast (FP32模式)
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', enabled=False):
                    model_input = original_frontend_func(i, prompt_text, prompt_speech_16k, sample_rate, zero_shot_spk_id)
            else:
                model_input = original_frontend_func(i, prompt_text, prompt_speech_16k, sample_rate, zero_shot_spk_id)

            # 3. 输出搬运到 GPU (准备喂给 LLM/Flow)
            for key, val in model_input.items():
                if isinstance(val, torch.Tensor):
                    model_input[key] = val.to(self.device)
            
            return model_input

        # 替换方法
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
        
        # 2. 移动到 CUDA (保持 FP32)
        # Monkey Patch 会负责把它暂时挪回 CPU，这里放 GPU 也没事，
        if torch.cuda.is_available():
            prompt_speech_16k = prompt_speech_16k.to(self.device)
        
        # 3. 推理 (Autocast 开启，LLM 需要 BF16)
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