针对 Python 项目（尤其是像你正在做的 AI/音频处理类项目），`apt` 安装的系统级依赖通常分为三类：**构建工具**、**多媒体/特定库支持**、**通用工具**。

以下是一份比较通用的清单，你可以根据需要选择安装：

### 1\. 核心构建与编译工具 (最重要)

几乎所有复杂的 Python 包（如 pandas, numpy, deepspeed, pyworld）在安装时都需要编译 C/C++ 扩展。

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    git \
    cmake
```

  * `build-essential`: 包含 gcc, g++, make 等编译器，**必须装**。
  * `python3-dev`: 包含 Python 的头文件（.h），编译任何 Python C 扩展（如 pyworld）都**必须装**。
  * `cmake`: 很多现代 C++ 项目（如某些 AI 库）的构建工具。

### 2\. 多媒体与音频处理 (针对 CosyVoice/TTS)

你的项目涉及音频，这些是处理 wav/mp3 等文件底层的编解码库。

```bash
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    libasound2-dev \
    portaudio19-dev
```

  * `ffmpeg`: 视频/音频转码的全能王，`pydub` 或 `librosa` 等库底层常调用它。
  * `libsndfile1`: `soundfile` 库（读取 wav 文件的库）的系统依赖，**必须装**。
  * `libasound2-dev`: Linux 下的 ALSA 音频驱动接口，`pyaudio` 等录音/播放库需要。

### 3\. 高性能计算与网络 (针对 AI/Web 服务)

如果你的项目涉及高性能网络通信或特定计算。

```bash
sudo apt-get install -y \
    curl \
    wget \
    unzip \
    libopenblas-dev \
    pkg-config
```

  * `libopenblas-dev`: 某些数学库（如 numpy）如果没有预编译包，需要这个库进行加速。
  * `pkg-config`: 编译时用来查找库路径的工具，`deepspeed` 可能会用到。

### 总结：你的项目的一键安装命令

针对你当前的 **CosyVoice + L20 显卡** 环境，推荐运行这条“全家桶”命令，涵盖了 99% 的需求：

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev \
    git \
    cmake \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    libasound2-dev \
    ninja-build
```

**特别说明 `ninja-build`**：

  * **用途**：你刚才安装 `flash-attn` 时提到的“并发编译”就是靠它。
  * **建议**：虽然 Python 的 `pip install ninja` 可以装 Python 版的 ninja，但安装一个系统级的 `ninja-build` 也是个好习惯，能避免某些底层 C++ 项目找不到构建工具。