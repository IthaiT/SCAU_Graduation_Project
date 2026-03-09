"""多轮训练+评估基准测试: 重复 N 次训练-评估流程，记录每轮指标并统计均值±标准差。"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score

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
NUM_RUNS = 10          # ← 在这里修改重复次数
SEQ_LEN = 30
PRED_LEN = 1
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_HEADS = 4
EPOCHS = 50
PATIENCE = 10
LR = 5e-4

MODEL_NAMES = ["LSTM", "Transformer", "LSTM_Transformer", "Parallel_LSTM_Transformer"]
MODEL_LABELS = {"LSTM": "LSTM", "Transformer": "Transformer", "LSTM_Transformer": "LSTM-Transformer", "Parallel_LSTM_Transformer": "Parallel-LSTM-Trans"}
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "benchmark_results.txt"


# ── 辅助 ──────────────────────────────────────────────────────────
def _build_model(name: str, input_dim: int) -> nn.Module:
    if name == "LSTM":
        return LSTMModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN)
    if name == "Transformer":
        return TransformerModel(input_dim=input_dim, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN)
    if name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN)
    return LSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN)


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((true - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100)
    r2 = float(r2_score(true, pred))
    true_diff = np.diff(true)
    pred_diff = np.diff(pred)
    da = float(np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100)
    return {
        "MSE": round(mse, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "MAPE": round(mape, 4), "R2": round(r2, 4), "DA": round(da, 2),
    }


@torch.no_grad()
def _evaluate_model(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred, _ = model(X)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


# ── 单轮: 训练 + 评估 ────────────────────────────────────────────
def run_once(
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    scaler_target,
    num_features: int,
) -> dict[str, dict[str, float]]:
    """执行一次完整的 训练→评估，返回 {model_name: {MSE, RMSE, MAE, MAPE, R2, DA}}。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models: dict[str, nn.Module] = {
        "LSTM": LSTMModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN),
        "Transformer": TransformerModel(input_dim=num_features, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
        "LSTM_Transformer": LSTMTransformerModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
        "Parallel_LSTM_Transformer": ParallelLSTMTransformerModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
    }

    # 训练
    for model_name, model in models.items():
        logger.info("训练: {} ({:,} params)", model_name, sum(p.numel() for p in model.parameters()))
        criterion = nn.HuberLoss(delta=1.0)
        # 统一训练配方，确保公平对比（架构是唯一变量）
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, device=device,
            epochs=EPOCHS, patience=PATIENCE,
            save_path=MODELS_DIR / f"{model_name}_best.pth",
            scheduler=scheduler,
        )

    # 评估
    all_metrics: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES:
        model = _build_model(name, num_features)
        weight_path = MODELS_DIR / f"{name}_best.pth"
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)

        preds_norm, trues_norm = _evaluate_model(model, test_loader, device)
        pred_r = scaler_target.inverse_transform(preds_norm).ravel()
        true_r = scaler_target.inverse_transform(trues_norm).ravel()
        all_metrics[name] = compute_metrics(true_r, pred_r)

    return all_metrics


# ── 格式化输出 ────────────────────────────────────────────────────
def _fmt_table_header() -> str:
    hdr = f"{'Model':<20s} {'MSE':>10s} {'RMSE':>10s} {'MAE':>10s} {'R²':>10s} {'MAPE%':>10s} {'DA%':>10s}"
    sep = "-" * 84
    return f"{sep}\n{hdr}\n{sep}"


def _fmt_row(label: str, m: dict[str, float]) -> str:
    return f"{label:<20s} {m['MSE']:>10.2f} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} {m['R2']:>10.4f} {m['MAPE']:>10.2f} {m['DA']:>10.2f}"


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: {} | 总轮次: {}", device, NUM_RUNS)

    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)
    train_loader, val_loader, test_loader, scaler_target = get_dataloaders(
        df_values=df.values, columns=columns,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE,
    )

    # 存储所有轮次的指标
    all_runs: list[dict[str, dict[str, float]]] = []
    lines: list[str] = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"Benchmark 开始时间: {timestamp}")
    lines.append(f"重复次数: {NUM_RUNS}  |  设备: {device}")
    lines.append(f"超参数: SEQ_LEN={SEQ_LEN}, EPOCHS={EPOCHS}, PATIENCE={PATIENCE}, BATCH_SIZE={BATCH_SIZE}, HIDDEN_DIM={HIDDEN_DIM}")
    lines.append("")

    for run_idx in range(1, NUM_RUNS + 1):
        logger.info("=" * 60)
        logger.info(">>> 第 {}/{} 轮", run_idx, NUM_RUNS)
        logger.info("=" * 60)

        metrics = run_once(device, train_loader, val_loader, test_loader, scaler_target, num_features)
        all_runs.append(metrics)

        # 写入本轮结果
        lines.append(f"===== 第 {run_idx}/{NUM_RUNS} 轮 =====")
        lines.append(_fmt_table_header())
        for name in MODEL_NAMES:
            label = MODEL_LABELS[name]
            m = metrics[name]
            lines.append(_fmt_row(label, m))
            logger.info("  {} → MSE={:.2f}  RMSE={:.2f}  MAE={:.2f}  R²={:.4f}  MAPE={:.2f}%  DA={:.2f}%",
                         label, m["MSE"], m["RMSE"], m["MAE"], m["R2"], m["MAPE"], m["DA"])
        lines.append("-" * 84)
        lines.append("")

    # ── 计算均值 & 标准差 ─────────────────────────────────────────
    lines.append("=" * 60)
    lines.append(f"  {NUM_RUNS} 轮平均 ± 标准差")
    lines.append("=" * 60)
    lines.append(f"{'Model':<20s} {'MSE':>16s} {'RMSE':>16s} {'MAE':>16s} {'R²':>16s} {'MAPE%':>16s} {'DA%':>16s}")
    lines.append("-" * 120)

    logger.info("=" * 60)
    logger.info("  {} 轮统计汇总", NUM_RUNS)
    logger.info("=" * 60)

    for name in MODEL_NAMES:
        label = MODEL_LABELS[name]
        vals = {k: [run[name][k] for run in all_runs] for k in ["MSE", "RMSE", "MAE", "MAPE", "R2", "DA"]}
        means = {k: np.mean(v) for k, v in vals.items()}
        stds = {k: np.std(v) for k, v in vals.items()}

        row = (
            f"{label:<20s}"
            f" {means['MSE']:>7.2f}±{stds['MSE']:<7.2f}"
            f" {means['RMSE']:>7.2f}±{stds['RMSE']:<7.2f}"
            f" {means['MAE']:>7.2f}±{stds['MAE']:<7.2f}"
            f" {means['R2']:>7.4f}±{stds['R2']:<7.4f}"
            f" {means['MAPE']:>7.2f}±{stds['MAPE']:<7.2f}"
            f" {means['DA']:>7.2f}±{stds['DA']:<7.2f}"
        )
        lines.append(row)
        logger.info("{:<20s} MSE={:.2f}±{:.2f}  RMSE={:.2f}±{:.2f}  MAE={:.2f}±{:.2f}  R²={:.4f}±{:.4f}  MAPE={:.2f}%±{:.2f}  DA={:.2f}%±{:.2f}",
                     label, means["MSE"], stds["MSE"], means["RMSE"], stds["RMSE"],
                     means["MAE"], stds["MAE"], means["R2"], stds["R2"],
                     means["MAPE"], stds["MAPE"], means["DA"], stds["DA"])

    lines.append("=" * 120)
    lines.append(f"\nBenchmark 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 写入文件 ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("完整结果已保存 → {}", OUTPUT_FILE)

    # 同时保存 JSON 方便后续分析
    json_path = RESULTS_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(all_runs, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON 结果已保存 → {}", json_path)


if __name__ == "__main__":
    main()
