# tts-research

## info

git clone git@github.com:FunAudioLLM/CosyVoice.git

## cosyvoice2

 v100 环境下，cosyvoice2 测试结果

 流式服务 首响 RTF

## cosyvoice2 train

https://www.bilibili.com/video/BV1fZNBeLEeM/

## opz

https://blog.csdn.net/gitblog_00003/article/details/151329543


```bash
screen -S tts
# 按 Ctrl + A 然后按 D 来退出并保留程序运行
screen -r tts
 # 列出所有会话
screen -ls   
screen -r 23986   # 进入 ID 为 1234 的会话
```

flash-attention 要用编译好的

```bash
wget https://mirror.ghproxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```