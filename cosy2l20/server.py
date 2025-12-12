from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles # 1. 导入
from service_core import service_instance
import shutil
import os
import datetime
import soundfile as sf # 务必导入 soundfile
import struct

app = FastAPI()
app.mount("/web", StaticFiles(directory="static", html=True), name="static")
# 辅助函数：手动生成一个 WAV 文件头 (44字节)
# 用于流式传输给客户端，告诉播放器这是个 24k 16bit 单声道的音频
def generate_wav_header(sample_rate, bits_per_sample, channels):
    datasize = 2000*1024*1024 # 设一个伪造的大长度，保证流式播放不中断
    o = bytes("RIFF", 'ascii')
    o += struct.pack('<I', datasize + 36)
    o += bytes("WAVEfmt ", 'ascii')
    o += struct.pack('<I', 16)
    o += struct.pack('<H', 1)
    o += struct.pack('<H', channels)
    o += struct.pack('<I', sample_rate)
    o += struct.pack('<I', sample_rate * channels * bits_per_sample // 8)
    o += struct.pack('<H', channels * bits_per_sample // 8)
    o += struct.pack('<H', bits_per_sample)
    o += bytes("data", 'ascii')
    o += struct.pack('<I', datasize)
    return o

@app.post("/stream")
async def stream_tts(
    text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...)
):
    log_dir = "/tmp/cosyvoice"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    input_prompt_path = f"{log_dir}/input_{timestamp}.wav"
    output_save_path = f"{log_dir}/output_{timestamp}.wav"

    # 保存输入音频
    with open(input_prompt_path, "wb") as buffer:
        shutil.copyfileobj(prompt_audio.file, buffer)
    print(f"✅ [Input] {input_prompt_path}")

    def iterfile():
        # A. 准备服务器本地文件写入 (使用 SoundFile 处理 WAV 头)
        # mode='w' 表示写入, samplerate=24000, channels=1, subtype='PCM_16'
        with sf.SoundFile(output_save_path, mode='w', samplerate=24000, channels=1, subtype='PCM_16') as local_file:

            # B. 延迟发送 WAV header：只在真正有音频数据时发送
            header_sent = False

            try:
                # 调用 Service (现在它吐出的是纯 RAW 数据)
                stream_generator = service_instance.stream_generate(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_wav=input_prompt_path
                )

                for chunk in stream_generator:
                    if chunk:
                        # 第一次有数据时才发送 WAV header
                        if not header_sent:
                            yield generate_wav_header(24000, 16, 1)
                            header_sent = True
                        local_file.buffer_write(chunk, dtype='int16')
                        yield chunk

                print(f"✅ [Output] {output_save_path} (Size: {os.path.getsize(output_save_path)} bytes)")

            except Exception as e:
                print(f"❌ Error: {e}")
                # 出错时确保关闭文件句柄（SoundFile context manager 会自动处理，但双重保险）

    return StreamingResponse(iterfile(), media_type="audio/wav")