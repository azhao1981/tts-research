import torch
import sys

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
import io
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import os

# 【新增】引入 torchao 的量化函数
try:
    from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("[Warn] torchao not installed. Quantization will be skipped.")

FINETURED_COSYVOICE2_MODEL_PATH = os.getenv("FINETURED_COSYVOICE2_MODEL_PATH", "pretrained_models/CosyVoice2-0.5B-finetune")

class OptimizedCosyService:
    # 【修改】增加 quantization_mode 参数 ('none', 'int8', 'int4')
    def __init__(self, model_dir=FINETURED_COSYVOICE2_MODEL_PATH, quantization_mode='none'):
        print(f"[Init] Loading model from {model_dir} on V100...")
        
        # 1. 加载模型 (FP16)
        self.model = CosyVoice2(
            model_dir, 
            load_jit=False, 
            load_trt=False, 
            fp16=False,
            # fp16=True,
        )
        # 2. ✅ 手动转为 Bfloat16 (L20 专用神器)
        #    Bfloat16 不会溢出，完美解决 "probability tensor contains inf/nan"
        if torch.cuda.is_available():
            print("[Init] Converting model to Bfloat16...")
            self.model.model.to(torch.bfloat16)
        # ----------------------------------------------------------------------
        # 【核心修改】在此处插入量化逻辑 (必须在 compile 之前)
        # ----------------------------------------------------------------------
        if TORCHAO_AVAILABLE and quantization_mode in ['int8', 'int4']:
            print(f"[Init] Applying {quantization_mode} quantization to LLM...")
            
            # 选择量化策略
            # int8_weight_only: 显存减少约 50%，精度损失极小，推理速度快
            # int4_weight_only: 显存减少约 75%，精度有轻微损失
            q_config = int4_weight_only() if quantization_mode == 'int4' else int8_weight_only()
            
            try:
                # 仅对 LLM 部分进行量化 (CosyVoice2 的大头在 LLM)
                # Flow 部分建议保持 FP16 以保证音质细腻度
                quantize_(self.model.model.llm, q_config)
                print(f"✅ LLM module quantized to {quantization_mode} successfully.")
            except Exception as e:
                print(f"[Error] Quantization failed: {e}")
        # ----------------------------------------------------------------------

        # 2. 编译优化
        if torch.cuda.is_available():
            print("[Init] Compiling sub-modules (LLM & Flow) with torch.compile...")
            try:
                # 注意：torch.compile 会自动处理量化后的算子
                self.model.model.llm = torch.compile(
                    self.model.model.llm, 
                    mode="reduce-overhead",
                    fullgraph=False
                )
                self.model.model.flow = torch.compile(
                    self.model.model.flow, 
                    mode="reduce-overhead",
                    fullgraph=False
                )
                print("✅ Torch Compile applied successfully.")
            except Exception as e:
                print(f"[Warn] Compile failed, running in eager mode: {e}")
        
        # 3. 预热
        print("[Init] Warming up with 'male1_trained'...")
        try:
            # 这里的 text 和 prompt 需要确保在你模型词表中存在，简单测试即可
            list(self.model.inference_sft(
                '这是预热文本', 'male1_trained', stream=True
            ))
            print("✅ Warmup done.")
        except Exception as e:
            print(f"[Warn] Warmup skipped: {e}")
            
        print("[Init] Service Ready.")

    def safe_load_prompt(self, wav_path, target_sr=16000):
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform

    def stream_generate(self, text, prompt_text, prompt_wav):
        prompt_speech_16k = self.safe_load_prompt(prompt_wav, 16000)
        
        # 2. 推理
        for i, output in enumerate(self.model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech_16k, 
            stream=True
        )):
            audio_tensor = output['tts_speech'].cpu()
            audio_numpy = audio_tensor.squeeze().numpy()
            buffer = io.BytesIO()
            sf.write(buffer, audio_numpy, 24000, format='RAW', subtype='PCM_16')            
            yield buffer.getvalue()

# 使用示例：你可以传入 'int8' 或 'int4'
# int8 推荐用于 V100，既省显存又利用 Tensor Core
# service_instance = OptimizedCosyService(quantization_mode='int8')
service_instance = OptimizedCosyService()