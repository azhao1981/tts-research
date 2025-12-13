# 1. 卸载可能半途安装进去的包
uv pip uninstall flash-attn

# 2. 清理 pip 的构建缓存 (非常重要！)
python -m pip cache purge

# 3. 清理 uv 的缓存 (以防万一)
uv cache clean
