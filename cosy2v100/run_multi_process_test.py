#!/usr/bin/env python3
"""
多进程测试启动器
支持多进程并发测试，每个进程可以运行多个批次
"""

import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from typing import List
import time
import json


def run_process(process_id: int, args: List[str]):
    """运行单个进程"""
    try:
        # 添加进程 ID 到输出目录
        output_dir = Path(args.output) / f"process_{process_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 构建进程特定的参数
        cmd = [
            sys.executable,
            "multi_process_test.py",
            "--url", args.url,
            "--audio", args.audio,
            "--text", args.text,
            "--prompt-text", args.prompt_text,
            "--concurrency", str(args.concurrency),
            "--batches", str(args.batches),
            "--output", str(output_dir),
        ]

        if args.no_save_wav:
            cmd.append("--no-save-wav")

        # 运行子进程
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 输出结果
        if result.stdout:
            print(f"\n[进程 {process_id} 输出]:")
            print(result.stdout)
        if result.stderr:
            print(f"\n[进程 {process_id} 错误]:")
            print(result.stderr)

        return result.returncode

    except Exception as e:
        print(f"进程 {process_id} 出错: {e}")
        return 1


def merge_results(output_dir: Path, process_count: int):
    """合并所有进程的结果"""
    print("\n📊 合并所有进程的结果...")

    all_results = []
    success_count = 0
    failure_count = 0
    total_tasks = 0

    # 遍历所有进程目录
    for pid in range(process_count):
        process_dir = output_dir / f"process_{pid + 1}"
        if not process_dir.exists():
            continue

        # 查找所有结果文件
        for result_file in process_dir.glob("*_results.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 收集结果
                    all_results.extend(data.get('results', []))

                    # 统计信息
                    batch_info = data.get('batch_info', {})
                    success_count += batch_info.get('success_count', 0)
                    failure_count += batch_info.get('failure_count', 0)
                    total_tasks += batch_info.get('total_tasks', 0)

            except Exception as e:
                print(f"  警告：无法读取 {result_file}: {e}")

    # 保存合并后的结果
    if all_results:
        merged_file = output_dir / f"merged_results_{int(time.time())}.json"
        merged_data = {
            "summary": {
                "total_processes": process_count,
                "total_tasks": total_tasks,
                "total_success": success_count,
                "total_failure": failure_count,
                "success_rate": (success_count / total_tasks * 100) if total_tasks > 0 else 0,
            },
            "results": all_results
        }

        with open(merged_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 合并结果已保存到: {merged_file}")
        print(f"\n📈 总体统计:")
        print(f"  总进程数: {process_count}")
        print(f"  总任务数: {total_tasks}")
        print(f"  成功数: {success_count}")
        print(f"  失败数: {failure_count}")
        print(f"  成功率: {merged_data['summary']['success_rate']:.1f}%")

        # 计算平均值
        if success_count > 0:
            success_results = [r for r in all_results if r.get('success', False)]
            avg_ttfb = sum(r.get('ttfb_ms', 0) for r in success_results) / len(success_results)
            avg_rtf = sum(r.get('rtf', 0) for r in success_results) / len(success_results)
            print(f"  平均 TTFB: {avg_ttfb:.0f} ms")
            print(f"  平均 RTF: {avg_rtf:.3f}")


def main():
    parser = argparse.ArgumentParser(description="多进程 TTS 压测启动器")
    parser.add_argument("--url", default="http://127.0.0.1:13099/stream",
                       help="TTS 服务地址")
    parser.add_argument("--audio", required=True,
                       help="参考音频文件路径")
    parser.add_argument("--text",
                       default="喂，李先生您好，我这边是先锋教育的王老师。打电话是想跟您同步一个对您家孩子可能有用的信息。",
                       help="目标文本")
    parser.add_argument("--prompt-text",
                       default="咱们这个项目呢属于是投入低，回本快，而且现在加盟呢还有一些政策上的这个优惠，呃，您看我后续让招商经理联系您，给您详细介绍一下可以吗？呃，不会占用您太多时间的，也是给您自己一个赚钱的机会嘛。",
                       help="提示文本")
    parser.add_argument("--concurrency", type=int, default=1,
                       help="每个进程的并发数")
    parser.add_argument("--batches", type=int, default=1,
                       help="每个进程的批次数")
    parser.add_argument("--processes", type=int, default=2,
                       help="进程数")
    parser.add_argument("--output", default="./test_output",
                       help="输出目录")
    parser.add_argument("--no-save-wav", action="store_true",
                       help="不保存 WAV 文件")

    args = parser.parse_args()

    # 检查音频文件
    audio_file = Path(args.audio)
    if not audio_file.exists():
        print(f"❌ 音频文件不存在: {audio_file}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 启动多进程压测")
    print(f"  URL: {args.url}")
    print(f"  音频: {audio_file}")
    print(f"  进程数: {args.processes}")
    print(f"  每进程并发数: {args.concurrency}")
    print(f"  每进程批次数: {args.batches}")
    print(f"  总请求数: {args.processes * args.concurrency * args.batches}")
    print(f"  输出目录: {output_dir}")
    print(f"  保存 WAV: {'否' if args.no_save_wav else '是'}")

    # 创建进程池
    start_time = time.time()

    print(f"\n⏰ 启动 {args.processes} 个进程...")
    processes = []

    # 启动所有进程
    for pid in range(args.processes):
        p = mp.Process(target=run_process, args=(pid + 1, args))
        p.start()
        processes.append(p)
        print(f"  进程 {pid + 1} 已启动 (PID: {p.pid})")

    # 等待所有进程完成
    print(f"\n⏳ 等待所有进程完成...")
    for p in processes:
        p.join()

    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n✅ 所有进程完成，总耗时: {total_time:.2f} 秒")

    # 合并结果
    merge_results(output_dir, args.processes)


if __name__ == "__main__":
    # Windows 下多进程保护
    mp.freeze_support()
    main()