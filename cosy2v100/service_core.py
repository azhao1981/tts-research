import torch
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
import io
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import os
FINETURED_COSYVOICE2_MODEL_PATH = os.getenv("FINETURED_COSYVOICE2_MODEL_PATH", "pretrained_models/CosyVoice2-0.5B-finetune")

class OptimizedCosyService:
    def __init__(self, model_dir=FINETURED_COSYVOICE2_MODEL_PATH):
        print(f"[Init] Loading model from {model_dir} on V100...")
        
        # 1. 加载模型 (FP16)
        self.model = CosyVoice2(
            model_dir, 
            load_jit=False, 
            load_trt=False, 
            fp16=True 
        )
        
        # 2. 编译优化
        if torch.cuda.is_available():
            print("[Init] Compiling sub-modules (LLM & Flow) with torch.compile...")
            try:
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
        # 1. 加载音频 (CPU Tensor)
        prompt_speech_16k = self.safe_load_prompt(prompt_wav, 16000)
        
        # 调试打印，确保进入 Service 时内容是对的
        print(f"[DEBUG] TTS_TEXT (目标): {text[:20]}...")
        print(f"[DEBUG] PROMPT_TEXT (参考): {prompt_text[:20]}...")

        # 2. 推理 - 【这里是修改的关键】
        # 强制使用关键字参数，防止位置错乱
        for i, output in enumerate(self.model.inference_zero_shot(
            tts_text=text,               # <--- 明确指定：这是要合成的文本
            prompt_text=prompt_text,     # <--- 明确指定：这是参考文本
            prompt_speech_16k=prompt_speech_16k, 
            stream=True
        )):
            audio_tensor = output['tts_speech'].cpu()
            
            # 3. 写入 buffer
            audio_numpy = audio_tensor.squeeze().numpy()
            buffer = io.BytesIO()
            # 【核心修改点】 format='WAV' 改为 'RAW'
            # 这样输出的就是纯粹的音频数据，没有任何文件头
            sf.write(buffer, audio_numpy, 24000, format='RAW', subtype='PCM_16')            
            
            yield buffer.getvalue()

service_instance = OptimizedCosyService()