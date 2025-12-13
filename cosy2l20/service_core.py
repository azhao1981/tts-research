import torch
import sys

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
import io
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import os

# 1. 显式开启 Flash Attention 优化上下文
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# 2. 引入 torchao (如果存在)
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
            fp16=False, # 必须 False，防止内部错误转换
        )
        
        # 2. ✅ 手动转为 Bfloat16 (核心修复点)
        if torch.cuda.is_available():
            print("[Init] Converting sub-modules to Bfloat16...")
            self.device = torch.device('cuda')
            
            # 分别转换组件
            if hasattr(self.model.model, 'llm'):
                self.model.model.llm.to(dtype=torch.bfloat16)
            
            if hasattr(self.model.model, 'flow'):
                self.model.model.flow.to(dtype=torch.bfloat16)
            
            if hasattr(self.model.model, 'hift'):
                self.model.model.hift.to(dtype=torch.bfloat16)
                
            print("✅ All modules converted to Bfloat16.")

        # 3. 量化逻辑 (torchao)
        if TORCHAO_AVAILABLE and quantization_mode in ['int8', 'int4']:
            print(f"[Init] Applying {quantization_mode} quantization to LLM...")
            q_config = int4_weight_only() if quantization_mode == 'int4' else int8_weight_only()
            try:
                # 仅量化 LLM，Flow 保持 BF16 以保证音质
                quantize_(self.model.model.llm, q_config)
                print(f"✅ LLM module quantized to {quantization_mode}.")
            except Exception as e:
                print(f"[Error] Quantization failed: {e}")

        # 4. 编译优化 (Torch Compile)
        if torch.cuda.is_available():
            print("[Init] Compiling sub-modules...")
            try:
                self.model.model.llm = torch.compile(self.model.model.llm, mode="reduce-overhead", fullgraph=False)
                self.model.model.flow = torch.compile(self.model.model.flow, mode="reduce-overhead", fullgraph=False)
                print("✅ Torch Compile applied.")
            except Exception as e:
                print(f"[Warn] Compile failed: {e}")
        
        # 5. 预热 (Warmup) - 使用 Zero-shot 路径以测试 BF16 兼容性
        print("[Init] Warming up...")
        try:
            # 构造一个假的 BF16 音频输入进行预热，确保存活
            dummy_wav = torch.randn(16000, dtype=torch.bfloat16, device=self.device)
            list(self.model.inference_zero_shot(
                tts_text='预热', 
                prompt_text='预热', 
                prompt_speech_16k=dummy_wav, 
                stream=True
            ))
            print("✅ Warmup done.")
        except Exception as e:
            print(f"[Warn] Warmup skipped or failed: {e}")
            
        print("[Init] Service Ready.")

    def safe_load_prompt(self, wav_path, target_sr=16000):
        # 这里的 load 默认返回 Float32
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform

    def stream_generate(self, text, prompt_text, prompt_wav):
        # 1. 加载音频 (此时是 Float32 CPU tensor)
        prompt_speech_16k = self.safe_load_prompt(prompt_wav, 16000)
        
        # 2. 【核心修复】转换为 Bfloat16 并移动到 CUDA
        if torch.cuda.is_available():
            prompt_speech_16k = prompt_speech_16k.to(self.device, dtype=torch.bfloat16)
        
        # 3. 推理
        # 注意：不要在外部手动转 text/prompt_text，模型内部 Embedding 层会处理
        try:
            generator = self.model.inference_zero_shot(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech_16k, 
                stream=True
            )
            
            for i, output in enumerate(generator):
                audio_tensor = output['tts_speech'].cpu() # 拿回 CPU
                audio_numpy = audio_tensor.squeeze().float().numpy() # 确保转回 float32 numpy 供 soundfile 写入
                
                buffer = io.BytesIO()
                sf.write(buffer, audio_numpy, 24000, format='RAW', subtype='PCM_16')            
                yield buffer.getvalue()
                
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            import traceback
            traceback.print_exc()
            yield b"" # 防止客户端挂起

service_instance = OptimizedCosyService()