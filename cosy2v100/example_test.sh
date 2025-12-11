#!/bin/bash

# 多进程 TTS 压测示例脚本

# 设置参数
URL="http://127.0.0.1:13099/stream"
AUDIO_FILE="xiangyu.wav"
OUTPUT_DIR="./test_output"

# 示例1: 单进程单并发
echo "📋 示例1: 单进程单并发测试"
python multi_process_test.py \
    --url $URL \
    --audio $AUDIO_FILE \
    --concurrency 1 \
    --batches 1 \
    --output $OUTPUT_DIR/single_test

echo -e "\n" + "="*50 + "\n"

# 示例2: 单进程高并发
echo "📋 示例2: 单进程10并发测试"
python multi_process_test.py \
    --url $URL \
    --audio $AUDIO_FILE \
    --concurrency 10 \
    --batches 1 \
    --output $OUTPUT_DIR/concurrent_test

echo -e "\n" + "="*50 + "\n"

# 示例3: 多进程测试
echo "📋 示例3: 4进程，每进程5并发，每进程2批次"
python run_multi_process_test.py \
    --url $URL \
    --audio $AUDIO_FILE \
    --processes 4 \
    --concurrency 5 \
    --batches 2 \
    --output $OUTPUT_DIR/multi_process_test

echo -e "\n" + "="*50 + "\n"

# 示例4: 高压测试（不保存WAV文件）
echo "📋 示例4: 8进程高压测试（不保存WAV）"
python run_multi_process_test.py \
    --url $URL \
    --audio $AUDIO_FILE \
    --processes 8 \
    --concurrency 10 \
    --batches 3 \
    --no-save-wav \
    --output $OUTPUT_DIR/stress_test

echo -e "\n✅ 所有测试完成！"
echo "📁 结果保存在: $OUTPUT_DIR"