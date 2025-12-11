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
) -> tuple[List[TestResult], Path]:
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

        return processed_results, batch_dir


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


def generate_html_report(results: List[TestResult], output_file: Path, test_params: Dict[str, Any] = None, base_dir: Path = None, is_merged: bool = False):
    """生成 HTML 测试报告"""
    if not results:
        return

    success_results = [r for r in results if r.success]

    # 计算统计数据
    avg_ttfb = sum(r.ttfb_ms for r in success_results) / len(success_results) if success_results else 0
    avg_rtf = sum(r.rtf for r in success_results) / len(success_results) if success_results else 0
    avg_duration = sum(r.duration_s for r in success_results) / len(success_results) if success_results else 0

    success_rate = len(success_results) / len(results) * 100 if results else 0

    # HTML 模板
    html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TTS 压测报告 - {results[0].batch_id if results else 'Unknown'}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f7f9fc;
            line-height: 1.6;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }}
        h1 {{
            color: #1f2937;
            margin-top: 0;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #374151;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        .params {{
            background-color: #f3f4f6;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .param-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .param-item {{
            display: flex;
            justify-content: space-between;
        }}
        .param-label {{
            font-weight: 600;
            color: #6b7280;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.9rem;
        }}
        th {{
            background-color: #f9fafb;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #4b5563;
            border-bottom: 2px solid #e5e7eb;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:hover {{
            background-color: #f9fafb;
        }}
        .success {{
            color: #059669;
            font-weight: 600;
        }}
        .failure {{
            color: #dc2626;
            font-weight: 600;
        }}
        .download-link {{
            color: #3b82f6;
            text-decoration: none;
            font-weight: 600;
        }}
        .download-link:hover {{
            text-decoration: underline;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: #f9fafb;
            border-radius: 8px;
        }}
        .footer {{
            text-align: center;
            color: #6b7280;
            margin-top: 40px;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>TTS 语音合成压测报告</h1>

        <div class="summary">
            <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">总请求数</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="stat-value">{len(success_results)}</div>
                <div class="stat-label">成功请求数</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="stat-value">{success_rate:.1f}%</div>
                <div class="stat-label">成功率</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="stat-value">{avg_ttfb:.0f}</div>
                <div class="stat-label">平均 TTFB (ms)</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <div class="stat-value">{avg_rtf:.3f}</div>
                <div class="stat-label">平均 RTF</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">
                <div class="stat-value">{avg_duration:.2f}</div>
                <div class="stat-label">平均音频时长 (s)</div>
            </div>
        </div>

        <h2>测试参数</h2>
        <div class="params">
            <div class="param-grid">
                <div class="param-item">
                    <span class="param-label">批次 ID:</span>
                    <span>{results[0].batch_id if results else 'N/A'}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">并发数:</span>
                    <span>{test_params.get('concurrency', 'N/A')}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">批次数:</span>
                    <span>{test_params.get('batches', 'N/A')}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">URL:</span>
                    <span>{test_params.get('url', 'N/A')}</span>
                </div>
            </div>
        </div>

        <h2>详细结果</h2>
        <table>
            <thead>
                <tr>
                    <th>任务ID</th>
                    <th>开始时间</th>
                    <th>首响时间</th>
                    <th>结束时间</th>
                    <th>TTFB (ms)</th>
                    <th>总耗时 (ms)</th>
                    <th>音频时长 (s)</th>
                    <th>RTF</th>
                    <th>状态</th>
                    <th>音频文件</th>
                </tr>
            </thead>
            <tbody>
"""

    # 生成表格行
    for r in results:
        status_class = "success" if r.success else "failure"
        status_text = "成功" if r.success else "失败"

        wav_link = ""
        if r.wav_file and r.success:
            wav_path = Path(r.wav_file)
            if is_merged and base_dir:
                # 对于合并报告，计算相对于输出目录的相对路径
                relative_path = wav_path.relative_to(base_dir)
                wav_link = f'<a href="{relative_path}" class="download-link" download>{wav_path.name}</a>'
            else:
                # 对于单批次报告，HTML 和 WAV 在同一目录
                wav_link = f'<a href="{wav_path.name}" class="download-link" download>{wav_path.name}</a>'

        # 预计算格式化值
        ttfb_val = f"{r.ttfb_ms:.0f}" if r.success else '-'
        time_val = f"{r.total_time_ms:.0f}" if r.success else '-'
        duration_val = f"{r.duration_s:.2f}" if r.success else '-'
        rtf_val = f"{r.rtf:.3f}" if r.success else '-'

        html_template += f"""
                <tr>
                    <td>{r.task_id}</td>
                    <td>{format_time(r.start_time)}</td>
                    <td>{format_time(r.ttfb_time) if r.ttfb_time else '-'}</td>
                    <td>{format_time(r.end_time) if r.end_time else '-'}</td>
                    <td>{ttfb_val}</td>
                    <td>{time_val}</td>
                    <td>{duration_val}</td>
                    <td>{rtf_val}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{wav_link}</td>
                </tr>"""

    html_template += f"""
            </tbody>
        </table>
    </div>

    <div class="footer">
        报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
        数据来源: TTS 压测工具
    </div>
</body>
</html>
"""

    # 保存 HTML 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)


def format_time(time_str: Optional[str]) -> str:
    """格式化时间显示"""
    if not time_str:
        return '-'
    try:
        dt = datetime.fromisoformat(time_str)
        return dt.strftime('%H:%M:%S.%f')[:-3]
    except Exception:
        return time_str


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
        results, batch_dir = await run_concurrent_tests(
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

        # 生成 HTML 报告
        test_params = {
            'url': args.url,
            'concurrency': args.concurrency,
            'batches': args.batches,
            'text': args.text,
            'prompt_text': args.prompt_text,
            'audio_file': str(audio_file)
        }

        # HTML 报告保存在批次目录中，这样相对路径才能正确工作
        html_file = batch_dir / f"{batch_id}_report.html"
        generate_html_report(results, html_file, test_params)
        print(f"  HTML 报告已生成: {html_file}")

        # 打印摘要
        print_summary(results)

    # 保存所有结果
    if args.batches > 1:
        all_results_file = output_dir / f"all_results_{int(time.time())}.json"
        save_results(all_results, all_results_file)
        print(f"\n💾 所有结果已保存到: {all_results_file}")


if __name__ == "__main__":
    asyncio.run(main())