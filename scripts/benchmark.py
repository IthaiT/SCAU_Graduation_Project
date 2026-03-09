"""多轮训练+评估基准测试 (三分类版): 重复 N 次，统计 Accuracy/F1 均值±标准差。"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim.lr_scheduler import CosineAnnealingLR  # noqa: E402

from src.data.dataset import get_dataloaders  # noqa: E402
from src.engine.trainer import train_model  # noqa: E402
from src.models.networks import (  # noqa: E402
    LSTMModel,
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# ═══════════════════════════════════════════════════════════════════
# 可调参数
# ═══════════════════════════════════════════════════════════════════
NUM_RUNS = 10
SEQ_LEN = 30
NUM_CLASSES = 3
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_HEADS = 4
EPOCHS = 50
PATIENCE = 10
LR = 5e-4

MODEL_NAMES = ["LSTM", "Transformer", "LSTM_Transformer", "Parallel_LSTM_Transformer"]
MODEL_LABELS = {
    "LSTM": "LSTM", "Transformer": "Transformer",
    "LSTM_Transformer": "LSTM-Transformer",
    "Parallel_LSTM_Transformer": "Parallel-LSTM-Trans",
}
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ── 辅助 ──────────────────────────────────────────────────────────
def _build_model(name: str, input_dim: int) -> nn.Module:
    if name == "LSTM":
        return LSTMModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
    if name == "Transformer":
        return TransformerModel(input_dim=input_dim, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES)
    if name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES)
    return LSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES)


def _compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(true, pred)), 4),
        "Macro-F1": round(float(f1_score(true, pred, average="macro")), 4),
        "Weighted-F1": round(float(f1_score(true, pred, average="weighted")), 4),
    }


@torch.no_grad()
def _evaluate_model(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model(X)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


# ── 单轮: 训练 + 评估 ────────────────────────────────────────────
def run_once(
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    num_features: int,
) -> dict[str, dict[str, float]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models: dict[str, nn.Module] = {
        "LSTM": LSTMModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES),
        "Transformer": TransformerModel(input_dim=num_features, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES),
        "LSTM_Transformer": LSTMTransformerModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES),
        "Parallel_LSTM_Transformer": ParallelLSTMTransformerModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES),
    }

    for model_name, model in models.items():
        logger.info("训练: {} ({:,} params)", model_name, sum(p.numel() for p in model.parameters()))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, device=device,
            epochs=EPOCHS, patience=PATIENCE,
            save_path=MODELS_DIR / f"{model_name}_best.pth",
            scheduler=scheduler,
        )

    all_metrics: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES:
        model = _build_model(name, num_features)
        weight_path = MODELS_DIR / f"{name}_best.pth"
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)
        preds, trues = _evaluate_model(model, test_loader, device)
        all_metrics[name] = _compute_metrics(trues, preds)

    return all_metrics


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: {} | 总轮次: {}", device, NUM_RUNS)

    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)
    train_loader, val_loader, test_loader = get_dataloaders(
        df_values=df.values, columns=columns,
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    )

    all_runs: list[dict[str, dict[str, float]]] = []
    lines: list[str] = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Benchmark 开始时间: {timestamp}")
    lines.append(f"重复次数: {NUM_RUNS}  |  设备: {device}")
    lines.append(f"超参数: SEQ_LEN={SEQ_LEN}, EPOCHS={EPOCHS}, PATIENCE={PATIENCE}, BATCH_SIZE={BATCH_SIZE}")
    lines.append("")

    hdr = f"{'Model':<24s} {'Accuracy':>10s} {'Macro-F1':>10s} {'Weighted-F1':>12s}"
    sep = "-" * 60

    for run_idx in range(1, NUM_RUNS + 1):
        logger.info("=" * 60)
        logger.info(">>> 第 {}/{} 轮", run_idx, NUM_RUNS)

        metrics = run_once(device, train_loader, val_loader, test_loader, num_features)
        all_runs.append(metrics)

        lines.append(f"===== 第 {run_idx}/{NUM_RUNS} 轮 =====")
        lines.append(f"{sep}\n{hdr}\n{sep}")
        for name in MODEL_NAMES:
            m = metrics[name]
            lines.append(f"{MODEL_LABELS[name]:<24s} {m['Accuracy']:>10.4f} {m['Macro-F1']:>10.4f} {m['Weighted-F1']:>12.4f}")
            logger.info("  {} → Acc={:.4f}  F1={:.4f}", MODEL_LABELS[name], m["Accuracy"], m["Macro-F1"])
        lines.append(sep)
        lines.append("")

    # ── 计算均值 & 标准差 ─────────────────────────────────────────
    lines.append("=" * 60)
    lines.append(f"  {NUM_RUNS} 轮平均 ± 标准差")
    lines.append("=" * 60)
    lines.append(f"{'Model':<24s} {'Accuracy':>16s} {'Macro-F1':>16s} {'Weighted-F1':>18s}")
    lines.append("-" * 78)

    for name in MODEL_NAMES:
        label = MODEL_LABELS[name]
        vals = {k: [run[name][k] for run in all_runs] for k in ["Accuracy", "Macro-F1", "Weighted-F1"]}
        means = {k: np.mean(v) for k, v in vals.items()}
        stds = {k: np.std(v) for k, v in vals.items()}

        row = (
            f"{label:<24s}"
            f" {means['Accuracy']:>7.4f}±{stds['Accuracy']:<7.4f}"
            f" {means['Macro-F1']:>7.4f}±{stds['Macro-F1']:<7.4f}"
            f" {means['Weighted-F1']:>7.4f}±{stds['Weighted-F1']:<7.4f}"
        )
        lines.append(row)
        logger.info("{:<24s} Acc={:.4f}±{:.4f}  F1={:.4f}±{:.4f}",
                     label, means["Accuracy"], stds["Accuracy"],
                     means["Macro-F1"], stds["Macro-F1"])

    lines.append("=" * 78)
    lines.append(f"\nBenchmark 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "benchmark_results.txt"
    output_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("完整结果已保存 → {}", output_file)

    json_path = RESULTS_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(all_runs, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON 结果已保存 → {}", json_path)


if __name__ == "__main__":
    main()
