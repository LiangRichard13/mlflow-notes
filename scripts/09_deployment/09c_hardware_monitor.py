"""
阶段 9 示例 3: 硬件资源监控（CPU / 内存 / 磁盘 / 网络）
========================================================

目标：
  - 用 psutil 后台采样 CPU/内存/磁盘/网络指标
  - 用 mlflow.log_metric 把指标记成 step 序列
  - 配合"模拟负载"展示真实 CPU 变化
  - UI 中看硬件 metric 曲线

⚠️ MLflow 不是专门的硬件监控工具（没有阈值告警、历史回溯）
   这个示例展示如何把硬件指标纳入 MLflow 工作流。
   生产环境应该用 Prometheus + Grafana 做专业监控。

运行：
  conda activate mlflow
  python scripts/09_deployment/09c_hardware_monitor.py

  # 然后另开终端：
  mlflow ui --port 5000
  # 看 experiment '09_hardware_monitor' → 选最新 Run → Metrics
  # 应该看到 cpu_percent / mem_percent / disk_io / net_io 等曲线
"""

import mlflow
import psutil
import threading
import time
import random
import os
from datetime import datetime


# ============================================================
# 1. 模拟负载函数（让 CPU 真实波动）
# ============================================================
def cpu_burst_work(duration: float = 1.0):
    """密集 CPU 工作，制造可观测的负载"""
    end = time.time() + duration
    while time.time() < end:
        # 计算一些浮点数
        sum(i * i for i in range(10000))


def memory_work(size_mb: int = 50):
    """占用 size_mb MB 内存"""
    return bytearray(size_mb * 1024 * 1024)


# ============================================================
# 2. 硬件指标采集器（后台线程）
# ============================================================
def collect_metrics(stop_event: threading.Event, interval: float = 1.0):
    """每隔 interval 秒采一次硬件指标"""
    metrics_queue = []

    # 第一次调用 cpu_percent 作为基线（必须有这一步，否则第一次返回 0.0）
    psutil.cpu_percent(interval=None)

    while not stop_event.is_set():
        ts = time.time()
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()

        metrics_queue.append({
            "timestamp": ts,
            "cpu_percent": cpu_pct,
            "mem_percent": mem.percent,
            "mem_used_gb": mem.used / 1024**3,
            "mem_available_gb": mem.available / 1024**3,
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / 1024**3,
            "disk_read_mb": disk_io.read_bytes / 1024**2,
            "disk_write_mb": disk_io.write_bytes / 1024**2,
            "net_sent_mb": net_io.bytes_sent / 1024**2,
            "net_recv_mb": net_io.bytes_recv / 1024**2,
        })

        time.sleep(interval)

    return metrics_queue


# ============================================================
# 3. 主流程
# ============================================================
def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("09_hardware_monitor")
    mlflow.openai.autolog(disable=True)   # 关闭 autolog（这次不用 LLM）

    print("=" * 60)
    print("🖥️  硬件资源监控演示")
    print("=" * 60)
    print(f"  CPU 核心数: {psutil.cpu_count()}")
    print(f"  总内存: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"  启动采样 + 模拟负载...")

    # 启动后台采样线程
    stop_event = threading.Event()
    samples = []
    sample_lock = threading.Lock()

    def collect_with_lock():
        psutil.cpu_percent(interval=None)
        while not stop_event.is_set():
            ts = time.time()
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            with sample_lock:
                samples.append({
                    "timestamp": ts,
                    "cpu_percent": cpu_pct,
                    "mem_percent": mem.percent,
                    "mem_used_gb": mem.used / 1024**3,
                    "mem_available_gb": mem.available / 1024**3,
                    "disk_percent": disk.percent,
                    "disk_read_mb": disk_io.read_bytes / 1024**2,
                    "disk_write_mb": disk_io.write_bytes / 1024**2,
                    "net_sent_mb": net_io.bytes_sent / 1024**2,
                    "net_recv_mb": net_io.bytes_recv / 1024**2,
                })
            time.sleep(1.0)

    sampler = threading.Thread(target=collect_with_lock, daemon=True)
    sampler.start()

    # 跑模拟负载 + 记录所有指标
    with mlflow.start_run(run_name="hardware-monitor-demo") as run:
        # 记录硬件配置（一次）
        mlflow.log_param("cpu_count", psutil.cpu_count())
        mlflow.log_param("cpu_freq_mhz", psutil.cpu_freq().current)
        mlflow.log_param("mem_total_gb", psutil.virtual_memory().total / 1024**3)

        mlflow.set_tag("monitoring_type", "hardware_resources")
        mlflow.set_tag("interval_seconds", 1)

        # 跑 30 秒：交替"空闲 → 高 CPU → 高内存 → 空闲"
        print(f"\n  跑 30 秒负载模拟（每 5 秒切一次）：")

        schedule = [
            (0, 5,   "idle",       None),
            (5, 10,  "cpu_burst",  cpu_burst_work),
            (10, 15, "memory",     lambda: memory_work(200)),
            (15, 20, "cpu_burst",  cpu_burst_work),
            (20, 25, "memory",     lambda: memory_work(500)),
            (25, 30, "idle",       None),
        ]

        start = time.time()
        next_log_step = 0

        while time.time() - start < 30:
            elapsed = time.time() - start
            # 找当前阶段
            current = "idle"
            for t_start, t_end, name, _ in schedule:
                if t_start <= elapsed < t_end:
                    current = name
                    break

            # 每秒 log 一次
            time.sleep(1.0)
            with sample_lock:
                if samples:
                    s = samples[-1]
            step = int(elapsed)

            mlflow.log_metric("cpu_percent", s["cpu_percent"], step=step)
            mlflow.log_metric("mem_percent", s["mem_percent"], step=step)
            mlflow.log_metric("mem_used_gb", s["mem_used_gb"], step=step)
            mlflow.log_metric("disk_percent", s["disk_percent"], step=step)
            mlflow.log_metric("disk_read_mb", s["disk_read_mb"], step=step)
            mlflow.log_metric("disk_write_mb", s["disk_write_mb"], step=step)
            mlflow.log_metric("net_sent_mb", s["net_sent_mb"], step=step)
            mlflow.log_metric("net_recv_mb", s["net_recv_mb"], step=step)

            # 同时跑当前阶段的负载
            for t_start, t_end, name, fn in schedule:
                if t_start <= elapsed < t_end and fn is not None:
                    if not getattr(collect_with_lock, f'_{name}_running', False):
                        setattr(collect_with_lock, f'_{name}_running', True)
                        threading.Thread(target=fn, kwargs={"duration": 4}, daemon=True).start()

            print(f"    [{step:>2}s] phase={current:<10} cpu={s['cpu_percent']:>5.1f}% mem={s['mem_percent']:>5.1f}% mem_used={s['mem_used_gb']:.2f}GB")

        # 停止采样
        stop_event.set()
        sampler.join(timeout=2)

        # 最终总结
        if samples:
            avg_cpu = sum(s["cpu_percent"] for s in samples) / len(samples)
            max_cpu = max(s["cpu_percent"] for s in samples)
            avg_mem = sum(s["mem_percent"] for s in samples) / len(samples)
            max_mem = max(s["mem_percent"] for s in samples)

            mlflow.log_metrics({
                "summary_avg_cpu": avg_cpu,
                "summary_max_cpu": max_cpu,
                "summary_avg_mem": avg_mem,
                "summary_max_mem": max_mem,
            })

            print(f"\n" + "=" * 60)
            print(f"📊 采样汇总（{len(samples)} 条）:")
            print(f"  CPU: avg={avg_cpu:.1f}% max={max_cpu:.1f}%")
            print(f"  Mem: avg={avg_mem:.1f}% max={max_mem:.1f}%")
            print("=" * 60)

    print(f"\n✓ Run: {run.info.run_id[:8]}")
    print("\n" + "=" * 60)
    print("📊 在 UI 看：")
    print("=" * 60)
    print("""
    1. mlflow ui --port 5000
    2. 选 experiment '09_hardware_monitor'
    3. 点开最新 Run
    4. 看 Metrics 标签，能看到以下曲线（按 step）：
       - cpu_percent（应该看到 5-10s 期间飙到 ~100%）
       - mem_percent（应该看到 10-15s 和 20-25s 期间明显上升）
       - disk_read_mb / disk_write_mb / net_sent_mb / net_recv_mb
    5. 还可以看 Tags 里的 monitoring_type

    💡 实际生产用法：
       - 把这个脚本改成一个 daemon，长期后台采样
       - 或在训练脚本里嵌入采样（用 callback）
       - 设阈值告警（Mem > 90% 触发 webhook）
    """)


if __name__ == "__main__":
    main()