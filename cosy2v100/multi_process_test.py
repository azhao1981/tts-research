#!/usr/bin/env python3
import asyncio
import aiohttp
import aiofiles
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from dataclasses import dataclass, asdict
import soundfile as sf
import numpy as np


@dataclass
class TestResult:
    """单次测试结果"""
    task_id: int
    batch_id: str
    start_time: str
    ttfb_time: Optional[str] = None
    end_time: Optional[str] = None
    ttfb_ms: float = 0.0
    total_time_ms: float = 0.0
    duration_s: float = 0.0
    rtf: float = 0.0
    success: bool = False
    error_msg: Optional[str] = None
    wav_file: Optional[str] = None


async def run_single_task(
    session: aiohttp.ClientSession,
    task_id: int,
    batch_id: str,
    url: str,
    text: str,
    prompt_text: str,
    audio_file: Path,
    output_dir: Path,
    save_wav: bool = True
) -> TestResult:
    """运行单个测试任务"""
    result = TestResult(
        task_id=task_id,
        batch_id=batch_id,
        start_time=datetime.now().isoformat()
    )

    try:
        # 准备文件上传
        data = aiohttp.FormData()
        data.add_field('text', text)
        data.add_field('prompt_text', prompt_text)
        data.add_field('prompt_audio',
                      open(audio_file, 'rb'),
                      filename=audio_file.name,
                      content_type='audio/wav')

        # 记录开始时间（用于计算 TTFB）
        start_perf = time.perf_counter()
        ttfb_recorded = False

        # 发起请求
        async with session.post(url, data=data) as response:
            if response.status != 200:
                result.error_msg = f"HTTP {response.status}"
                result.end_time = datetime.now().isoformat()
                return result

            # 读取流式响应
            chunks = []
            first_chunk_time = None

            async for chunk in response.content.iter_chunked(8192):
                if not ttfb_recorded:
                    result.ttfb_ms = (time.perf_counter() - start_perf) * 1000
                    result.ttfb_time = datetime.now().isoformat()
                    ttfb_recorded = True

                chunks.append(chunk)

            # 接收完成
            result.total_time_ms = (time.perf_counter() - start_perf) * 1000
            result.end_time = datetime.now().isoformat()

            # 合并音频数据
            audio_data = b''.join(chunks)

            # 保存 WAV 文件
            if save_wav:
                wav_filename = f"{batch_id}_{task_id:03d}.wav"
                wav_path = output_dir / wav_filename
                async with aiofiles.open(wav_path, 'wb') as f:
                    await f.write(audio_data)
                result.wav_file = str(wav_path)

            # 计算音频时长（使用 soundfile）
            try:
                import io
                audio_buffer = io.BytesIO(audio_data)
                data, samplerate = sf.read(audio_buffer)
                result.duration_s = len(data) / samplerate

                # 计算 RTF
                if result.duration_s > 0:
                    result.rtf = (result.total_time_ms / 1000) / result.duration_s

                result.success = True
            except Exception as e:
                result.error_msg = f"音频解析失败: {str(e)}"

    except Exception as e:
        result.error_msg = str(e)
        if not result.end_time:
            result.end_time = datetime.now().isoformat()

    return result


async def run_concurrent_tests(
    batch_id: str,
    concurrency: int,
    url: str,
    text: str,
    prompt_text: str,
    audio_file: Path,
    output_dir: Path,
    save_wav: bool = True
) -> List[TestResult]:
    """运行并发测试"""
    # 创建输出目录
    batch_dir = output_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 创建 HTTP 会话
    connector = aiohttp.TCPConnector(limit=100)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout
    ) as session:
        # 创建所有任务
        tasks = [
            run_single_task(
                session=session,
                task_id=i + 1,
                batch_id=batch_id,
                url=url,
                text=text,
                prompt_text=prompt_text,
                audio_file=audio_file,
                output_dir=batch_dir,
                save_wav=save_wav
            )
            for i in range(concurrency)
        ]

        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed_results.append(TestResult(
                    task_id=i + 1,
                    batch_id=batch_id,
                    start_time=datetime.now().isoformat(),
                    success=False,
                    error_msg=str(r)
                ))
            else:
                processed_results.append(r)

        return processed_results


def save_results(results: List[TestResult], output_file: Path):
    """保存测试结果到 JSON 文件"""
    # 转换为可序列化的字典
    data = {
        "batch_info": {
            "batch_id": results[0].batch_id if results else "",
            "total_tasks": len(results),
            "success_count": sum(1 for r in results if r.success),
            "failure_count": sum(1 for r in results if not r.success),
        },
        "results": [asdict(r) for r in results]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(results: List[TestResult]):
    """打印测试结果摘要"""
    success_results = [r for r in results if r.success]

    if not success_results:
        print(f"\n❌ 所有 {len(results)} 个请求都失败了")
        return

    # 计算统计数据
    avg_ttfb = sum(r.ttfb_ms for r in success_results) / len(success_results)
    avg_rtf = sum(r.rtf for r in success_results) / len(success_results)

    print(f"\n📊 测试结果摘要 ({len(success_results)}/{len(results)} 成功):")
    print(f"  平均 TTFB: {avg_ttfb:.0f} ms")
    print(f"  平均 RTF: {avg_rtf:.3f}")
    print(f"  成功率: {len(success_results)/len(results)*100:.1f}%")


async def main():
    parser = argparse.ArgumentParser(description="多进程 TTS 压测工具")
    parser.add_argument("--url", default="http://127.0.0.1:13099/stream",
                       help="TTS 服务地址")
    parser.add_argument("--audio", required=True,
                       help="参考音频文件路径")
    parser.add_argument("--text",
                       default="是这样的，投资一家麦当劳店目前不算房租，大概投入在六到八万，基本上十万左右就可以开出一家店。那方便问一下您之前有听说过麦当劳吗？麦当劳现在目前在全国已经有2000家门店了。",
                       help="目标文本")
    parser.add_argument("--prompt-text",
                       default="北京是有名额的，我们现在是不收加盟费了，我们是采用抽点的方式和加盟商一同运营成长的。",
                       help="提示文本")
    parser.add_argument("--concurrency", type=int, default=1,
                       help="并发数")
    parser.add_argument("--batches", type=int, default=1,
                       help="批次数（每批会生成新的 batch_id）")
    parser.add_argument("--output", default="./test_output",
                       help="输出目录")
    parser.add_argument("--no-save-wav", action="store_true",
                       help="不保存 WAV 文件")
    parser.add_argument("--processes", type=int, default=1,
                       help="进程数")

    args = parser.parse_args()

    # 检查音频文件
    audio_file = Path(args.audio)
    if not audio_file.exists():
        print(f"❌ 音频文件不存在: {audio_file}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 WAV 文件选项
    save_wav = not args.no_save_wav

    print(f"🚀 开始测试")
    print(f"  URL: {args.url}")
    print(f"  音频: {audio_file}")
    print(f"  并发数: {args.concurrency}")
    print(f"  批次数: {args.batches}")
    print(f"  进程数: {args.processes}")
    print(f"  保存 WAV: {'是' if save_wav else '否'}")

    all_results = []

    # 运行多批次测试
    for batch_idx in range(args.batches):
        batch_id = f"P{mp.current_process().pid}B{batch_idx+1:03d}T{int(time.time())%100000}"

        print(f"\n📦 批次 {batch_idx + 1}/{args.batches} (ID: {batch_id})")

        # 运行当前批次
        results = await run_concurrent_tests(
            batch_id=batch_id,
            concurrency=args.concurrency,
            url=args.url,
            text=args.text,
            prompt_text=args.prompt_text,
            audio_file=audio_file,
            output_dir=output_dir,
            save_wav=save_wav
        )

        all_results.extend(results)

        # 保存当前批次结果
        batch_file = output_dir / f"{batch_id}_results.json"
        save_results(results, batch_file)
        print(f"  结果已保存到: {batch_file}")

        # 打印摘要
        print_summary(results)

    # 保存所有结果
    if args.batches > 1:
        all_results_file = output_dir / f"all_results_{int(time.time())}.json"
        save_results(all_results, all_results_file)
        print(f"\n💾 所有结果已保存到: {all_results_file}")


if __name__ == "__main__":
    asyncio.run(main())