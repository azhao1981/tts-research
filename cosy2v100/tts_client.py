import requests
import time
import os
import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 1. 配置服务地址 (注意端口 13099)
url = os.getenv("TTS_SERVICE_URL", "http://127.0.0.1:13099/stream")

# 2. 准备请求数据
# 请确保当前目录下有一个真实的 wav 文件
prompt_wav_path = "xiangyu.wav" 

if not os.path.exists(prompt_wav_path):
    print(f"错误: 找不到参考音频 {prompt_wav_path}，请先准备一个wav文件。")
    exit()

payload = {
    "text": '喂，李先生您好，我这边是先锋教育的王老师。打电话是想跟您同步一个对您家孩子可能有用的信息——我们这周末专门为初二学生安排了一场免费的物理试听课，重点就讲"浮力"这个最容易丢分的模块，帮孩子彻底搞懂原理和解题技巧。',
    # 如果是 Zero-shot 模式，这个其实可以随便填，或者填参考音频的大致内容
    "prompt_text": "咱们这个项目呢属于是投入低，回本快，而且现在加盟呢还有一些政策上的这个优惠，呃，您看我后续让招商经理联系您，给您详细介绍一下可以吗？呃，不会占用您太多时间的，也是给您自己一个赚钱的机会嘛。",
}

files = {
    "prompt_audio": (os.path.basename(prompt_wav_path), open(prompt_wav_path, "rb"), "audio/wav")
}

print(f"正在发送请求到 {url} ...")
start_time = time.time()

# 3. 发送流式请求 (关键是 stream=True)
try:
    with requests.post(url, data=payload, files=files, stream=True) as response:
        response.raise_for_status()
        
        # 4. 接收并写入文件
        output_filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        print("开始接收音频流...")
        
        # 记录首包到达时间 (TTFT)
        first_chunk_received = False
        
        with open(output_filename, "wb") as f:
            # chunk_size=None 表示只要有数据就立即读取，不进行本地缓冲
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    if not first_chunk_received:
                        ttft = time.time() - start_time
                        print(f"⚡ 首包延迟 (TTFT): {ttft:.3f} 秒")
                        first_chunk_received = True
                    
                    f.write(chunk)
                    # 打印一个点表示收到一个数据包
                    print(".", end="", flush=True)

    print(f"\n✅ 生成完成! 音频已保存为: {output_filename}")

except requests.exceptions.ConnectionError:
    print(f"\n❌ 连接失败: 请检查服务是否在 13099 端口启动。")
except Exception as e:
    print(f"\n❌ 发生错误: {e}")