"""
阶段 1 示例 1: MLflow 基础追踪 - Hello World
=============================================

目标：理解 4 个最核心概念
  - Experiment（实验）：一组相关 Run 的容器
  - Run（运行）：单次实验执行，记录参数/指标/产物
  - Param / Metric / Artifact：每次 Run 里记录的内容
  - Tracking UI：可视化查看所有 Run

运行：
  conda activate mlflow
  python scripts/01_basics/01_hello_mlflow.py

然后另开终端：
  conda activate mlflow
  mlflow ui --port 5000
  # 浏览器打开 http://localhost:5000
"""

import mlflow
from mlflow import log_param, log_metric, log_artifact, set_experiment
import random
import tempfile
import os


def main():
    # ============ 1. 设置实验 ============
    # Experiment 是一组相关 Run 的容器（类似"项目"）
    # 如果实验不存在会自动创建
    exp_name = "01_basics_demo"
    set_experiment(exp_name)
    print(f"实验名称: {exp_name}")
    print(f"实验 ID:  {mlflow.get_experiment_by_name(exp_name).experiment_id}")
    print(f"跟踪 URI: {mlflow.get_tracking_uri()}")
    print("-" * 60)

    # ============ 2. 跑 3 个 Run 做对比 ============
    # 每个 Run 模拟一次"实验"——比如调不同的超参训练模型
    for run_idx in range(3):
        # 模拟不同超参数
        learning_rate = random.choice([0.001, 0.01, 0.1])
        batch_size = random.choice([16, 32, 64])

        # start_run() 创建一个新的 Run，可用 with 自动关闭
        with mlflow.start_run(run_name=f"trial-{run_idx+1}") as run:
            print(f"\n[Run {run_idx+1}] run_id = {run.info.run_id}")

            # --- 记录参数（超参数、配置等） ---
            # Param 是字符串类型，UI 上会显示为表格列
            log_param("learning_rate", learning_rate)
            log_param("batch_size", batch_size)
            log_param("optimizer", "adam")

            # --- 训练过程（模拟） ---
            # 训练多轮，每轮记录一个 metric
            num_epochs = 10
            for epoch in range(num_epochs):
                # 模拟 loss 下降、accuracy 上升
                loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
                acc = 0.5 + epoch * 0.05 + random.uniform(-0.02, 0.02)
                acc = min(acc, 0.99)  # 上限

                # 记录 metric（每一步都会被记录，UI 可画曲线）
                log_metric("loss", loss, step=epoch)
                log_metric("accuracy", acc, step=epoch)

            # --- 记录最终结果 ---
            # 也可以记录最终值（不指定 step），UI 会单独高亮
            log_metric("final_loss", loss)
            log_metric("final_accuracy", acc)

            # --- 记录 Tag（任意标签）---
            # Tag 用于标记状态、备注，UI 用来过滤/搜索
            mlflow.set_tag("model_type", "demo")
            mlflow.set_tag("status", "completed")
            mlflow.set_tag("notes", f"lr={learning_rate}, bs={batch_size}")

            # --- 记录 Artifact（文件产物）---
            # 任何文件都能存：图表、模型、配置、日志等
            with tempfile.TemporaryDirectory() as tmpdir:
                # 写一个简单的配置文件作为 artifact
                config_path = os.path.join(tmpdir, "config.txt")
                with open(config_path, "w") as f:
                    f.write(f"learning_rate: {learning_rate}\n")
                    f.write(f"batch_size: {batch_size}\n")
                    f.write(f"final_loss: {loss:.4f}\n")
                    f.write(f"final_accuracy: {acc:.4f}\n")
                log_artifact(config_path, artifact_path="configs")

                # 写一个文本 artifact
                summary_path = os.path.join(tmpdir, "summary.md")
                with open(summary_path, "w") as f:
                    f.write(f"# Trial {run_idx+1}\n\n")
                    f.write(f"- Final loss: {loss:.4f}\n")
                    f.write(f"- Final accuracy: {acc:.4f}\n")
                log_artifact(summary_path, artifact_path="summaries")

            print(f"  完成! final_loss={loss:.4f}, final_accuracy={acc:.4f}")

    print("\n" + "=" * 60)
    print("所有 Run 已记录！下一步：")
    print("  1. 另开终端: mlflow ui --port 5000")
    print("  2. 浏览器访问 http://localhost:5000")
    print("  3. 在 UI 中选择 '01_basics_demo' 实验，查看 3 个 Run 的对比")
    print("=" * 60)


if __name__ == "__main__":
    main()
