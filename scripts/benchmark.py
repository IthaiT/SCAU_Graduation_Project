"""V6 四模型多轮训练+评估基准测试: 重复 N 次训练-评估流程，统计均值±标准差。

适配 CSI 300 长周期数据:
- 输入: 17 维技术指标特征 (与 train.py 一致)
- 模型: LSTM / Transformer / Serial LSTM-Trans / Parallel LSTM-Trans
- 策略: MSELoss, Adam(lr=0.001), seq_len=60, epochs=100, patience=20。
"""
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

# ── 超参数 (严格与 train.py 保持一致) ─────────────────────────────
NUM_RUNS = 10
SEQ_LEN = 60
PRED_LEN = 1
BATCH_SIZE = 32
LR = 0.001

LSTM_HIDDEN = 60
NUM_HEADS = 5
HYBRID_HIDDEN = 64
HYBRID_HEADS = 4

EPOCHS = 100
PATIENCE = 20
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10

MODEL_NAMES =["LSTM", "Transformer", "LSTM_Transformer", "Parallel_LSTM_Transformer"]
MODEL_LABELS = {
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "LSTM_Transformer": "Serial LSTM-Trans",
    "Parallel_LSTM_Transformer": "Parallel LSTM-Trans",
}

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "benchmark_results.txt"


# ── 工具函数 ──────────────────────────────────────────────────────
def _build_model(name: str, input_dim: int) -> nn.Module:
    """遵循极简工厂模式，去除冗余模型。"""
    if name == "LSTM":
        return LSTMModel(
            input_dim=input_dim, hidden_dim=LSTM_HIDDEN,
            num_layers=2, pred_len=PRED_LEN, dropout=0.1,
        )
    if name == "Transformer":
        return TransformerModel(
            input_dim=input_dim, d_model=LSTM_HIDDEN,
            num_heads=NUM_HEADS, num_layers=2,
            ffn_dim=128, pred_len=PRED_LEN, dropout=0.15,
        )
    if name == "LSTM_Transformer":
        return LSTMTransformerModel(
            input_dim=input_dim, hidden_dim=HYBRID_HIDDEN,
            num_lstm_layers=2, num_heads=HYBRID_HEADS,
            num_transformer_layers=2, ffn_dim=256,
            pred_len=PRED_LEN, dropout=0.2,
        )
    if name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(
            input_dim=input_dim, hidden_dim=HYBRID_HIDDEN,
            num_lstm_layers=2, num_heads=HYBRID_HEADS,
            num_transformer_layers=2, ffn_dim=256,
            pred_len=PRED_LEN, dropout=0.2,
        )
    raise ValueError(f"Unknown model: {name}")


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((true - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100)
    r2 = float(r2_score(true, pred))
    da = float(np.mean(np.sign(np.diff(true)) == np.sign(np.diff(pred))) * 100)
    return {
        "MSE": round(mse, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "MAPE": round(mape, 4), "R2": round(r2, 4), "DA": round(da, 2),
    }


@torch.no_grad()
def _evaluate_model(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    preds, trues = [],[]
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred, _ = model(X)  # 忽略 extra/attention 返回值
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


# ── 单轮: 训练 + 评估 (高内聚流水线) ──────────────────────────────
def run_once(
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    scaler_target,
    num_features: int,
) -> dict[str, dict[str, float]]:
    """单轮完整生命周期：创建 -> 训练 -> 加载 -> 评估，避免状态泄露。"""
    
    # 使用独立的临时目录，防止压测覆盖标准训练权重
    tmp_dir = MODELS_DIR / "benchmark_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics: dict[str, dict[str, float]] = {}

    for name in MODEL_NAMES:
        model = _build_model(name, num_features)
        save_path = tmp_dir / f"{name}_best.pth"
        logger.info("  -> 训练: {} ({:,} params)", name, sum(p.numel() for p in model.parameters()))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # 1. 训练
        train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, device=device,
            epochs=EPOCHS, patience=PATIENCE, save_path=save_path,
            scheduler=None, max_grad_norm=1.0,
        )

        # 2. 评估 (强制重新加载最佳权重，保证逻辑严密)
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.to(device)
        
        preds_norm, trues_norm = _evaluate_model(model, test_loader, device)
        pred_r = scaler_target.inverse_transform(preds_norm).ravel()
        true_r = scaler_target.inverse_transform(trues_norm).ravel()
        
        all_metrics[name] = compute_metrics(true_r, pred_r)

    return all_metrics


# ── 格式化输出 ────────────────────────────────────────────────────
def _fmt_table_header() -> str:
    return (
        f"{'-'*84}\n"
        f"{'Model':<20s} {'MSE':>10s} {'RMSE':>10s} {'MAE':>10s} {'R²':>10s} {'MAPE%':>10s} {'DA%':>10s}\n"
        f"{'-'*84}"
    )


def _fmt_row(label: str, m: dict[str, float]) -> str:
    return f"{label:<20s} {m['MSE']:>10.2f} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} {m['R2']:>10.4f} {m['MAPE']:>10.2f} {m['DA']:>10.2f}"


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: {} | 总轮次: {}", device, NUM_RUNS)

    # 修复数据源：严格使用 17 维特征，与 train.py 对齐
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    if not csv_path.exists():
        logger.error("数据文件未找到: {}", csv_path)
        sys.exit(1)
        
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)  # = 17

    train_loader, val_loader, test_loader, scaler_target = get_dataloaders(
        df_values=df.values, columns=columns,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE,
        target_col="close", train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
    )

    all_runs: list[dict[str, dict[str, float]]] = []
    lines: list[str] =[
        f"Benchmark 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"重复次数: {NUM_RUNS}  |  设备: {device}  |  特征数: {num_features}",
        f"超参数: SEQ_LEN={SEQ_LEN}, EPOCHS={EPOCHS}, PATIENCE={PATIENCE}, BATCH_SIZE={BATCH_SIZE}, LR={LR}",
        ""
    ]

    for run_idx in range(1, NUM_RUNS + 1):
        logger.info("=" * 60)
        logger.info(">>> 第 {}/{} 轮", run_idx, NUM_RUNS)
        logger.info("=" * 60)

        metrics = run_once(device, train_loader, val_loader, test_loader, scaler_target, num_features)
        all_runs.append(metrics)

        # 写入并打印本轮结果
        lines.append(f"===== 第 {run_idx}/{NUM_RUNS} 轮 =====")
        lines.append(_fmt_table_header())
        for name in MODEL_NAMES:
            label = MODEL_LABELS[name]
            m = metrics[name]
            lines.append(_fmt_row(label, m))
            logger.info(
                "  {} → MSE={:.2f}  RMSE={:.2f}  MAE={:.2f}  R²={:.4f}  MAPE={:.2f}%  DA={:.2f}%",
                label, m["MSE"], m["RMSE"], m["MAE"], m["R2"], m["MAPE"], m["DA"]
            )
        lines.append("-" * 84 + "\n")

    # ── 计算均值 & 标准差 ─────────────────────────────────────────
    lines.extend([
        "=" * 120,
        f"  {NUM_RUNS} 轮平均 ± 标准差",
        "=" * 120,
        f"{'Model':<20s} {'MSE':>16s} {'RMSE':>16s} {'MAE':>16s} {'R²':>16s} {'MAPE%':>16s} {'DA%':>16s}",
        "-" * 120
    ])

    logger.info("=" * 80)
    logger.info("  {} 轮统计汇总", NUM_RUNS)
    logger.info("=" * 80)

    for name in MODEL_NAMES:
        label = MODEL_LABELS[name]
        vals = {k: [run[name][k] for run in all_runs] for k in["MSE", "RMSE", "MAE", "MAPE", "R2", "DA"]}
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
        logger.info(
            "{:<20s} MSE={:.2f}±{:.2f}  RMSE={:.2f}±{:.2f}  MAE={:.2f}±{:.2f}  "
            "R²={:.4f}±{:.4f}  MAPE={:.2f}%±{:.2f}  DA={:.2f}%±{:.2f}",
            label, means["MSE"], stds["MSE"], means["RMSE"], stds["RMSE"],
            means["MAE"], stds["MAE"], means["R2"], stds["R2"],
            means["MAPE"], stds["MAPE"], means["DA"], stds["DA"]
        )

    lines.append("=" * 120)
    lines.append(f"\nBenchmark 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 写入文件 ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    
    json_path = RESULTS_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(all_runs, indent=2, ensure_ascii=False), encoding="utf-8")
    
    logger.info("完整报告已保存 → {}", OUTPUT_FILE)
    logger.info("JSON 数据已保存 → {}", json_path)


if __name__ == "__main__":
    main()