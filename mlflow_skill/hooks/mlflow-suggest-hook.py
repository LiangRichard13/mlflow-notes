#!/usr/bin/env python3
"""UserPromptSubmit hook: detects MLflow usage and suggests relevant skill."""
import json
import sys
import re

def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    prompt = data.get("prompt", "").lower()

    suggestions = []

    if any(k in prompt for k in ["trace", "tracing", "autolog", "span", "instrument"]):
        suggestions.append("💡 Use the `instrumenting-with-mlflow-tracing` skill to add MLflow tracing.")

    if any(k in prompt for k in ["evaluat", "scorer", "judge", "dataset", "assess", "improve quality"]):
        suggestions.append("💡 Use the `agent-evaluation` skill to evaluate your agent with MLflow.")

    if any(k in prompt for k in ["trace id", "debug trace", "why did", "what went wrong", "analyze trace"]):
        suggestions.append("💡 Use the `analyze-mlflow-trace` skill to debug this trace.")

    if any(k in prompt for k in ["session", "conversation", "chat history", "multi-turn"]):
        suggestions.append("💡 Use the `analyze-mlflow-chat-session` skill to analyze chat sessions.")

    if any(k in prompt for k in ["search traces", "find traces", "filter traces", "get trace"]):
        suggestions.append("💡 Use the `retrieving-mlflow-traces` skill to search/filter traces.")

    if any(k in prompt for k in ["metrics", "token usage", "latency", "cost", "usage trend"]):
        suggestions.append("💡 Use the `querying-mlflow-metrics` skill to fetch aggregated metrics.")

    if any(k in prompt for k in ["get started", "set up mlflow", "onboard", "quickstart"]):
        suggestions.append("💡 Use the `mlflow-onboarding` skill to get started with MLflow.")

    if any(k in prompt for k in ["mlflow docs", "mlflow api", "how to use mlflow"]):
        suggestions.append("💡 Use the `searching-mlflow-docs` skill to search MLflow documentation.")

    # Classical ML lifecycle (intentionally avoids bare "evaluate"/"metrics" to prevent
    # collision with agent-evaluation and querying-mlflow-metrics). Uses model-context
    # phrases so a GenAI-style "evaluate my agent" won't trigger this.
    if any(k in prompt for k in [
        "sklearn", "scikit-learn", "xgboost", "lightgbm", "catboost",
        "pytorch train", "tensorflow train", "keras train",
        "train model", "training run",
        "log a model", "log my model", "log model",
        "register model", "registered model", "set alias", "champion", "challenger",
        "evaluate my model", "models.evaluate", "confusion matrix",
        "metric threshold", "model metrics",
        "serve my model", "deploy model", "build-docker", "containerize", "model predict",
        "dataset lineage", "log_input",
        "compare runs", "pick best model",
        "hyperparameter", "optuna", "grid search", "gridsearchcv",
    ]):
        suggestions.append("💡 Use the `classical-ml` skill for the classical ML model lifecycle (tracking, registry, evaluation, deployment, monitoring, optimization).")

    if suggestions:
        print("\n".join(suggestions))

if __name__ == "__main__":
    main()
