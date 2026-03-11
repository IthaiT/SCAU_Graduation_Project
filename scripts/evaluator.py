"""V6.1 学术级评估流水线: 基于最优超参数配置的权重加载与指标计算。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger

from src.data.dataset import get_dataloaders
from src.models.networks import (
    LSTMModel,
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# ── 全局绘图风格 ──────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.sans-serif":["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})
sns.set_theme(style="whitegrid", font="Microsoft YaHei", palette="muted")

_COLORS: dict[str, str] = {
    "LSTM": "#4C72B0",
    "Transformer": "#DD8452",
    "LSTM_Transformer": "#C44E52",
    "Parallel_LSTM_Transformer": "#8172B3",
}
_LABELS: dict[str, str] = {
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "LSTM_Transformer": "Serial LSTM-Trans",
    "Parallel_LSTM_Transformer": "Parallel LSTM-Trans",
}

# ── 基础任务配置 ──────────────────────────────────────────────────
SEQ_LEN = 60
PRED_LEN = 1
EVAL_BATCH_SIZE = 64  # 推理统一用64提升速度
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAMES = ["LSTM", "Transformer", "LSTM_Transformer", "Parallel_LSTM_Transformer"]

# ── 核心: 映射训练时的最优架构参数 ────────────────────────────────
BEST_ARCH_CONFIGS: dict[str, dict[str, Any]] = {
    "LSTM": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.437},
    "Transformer": {"d_model": 64, "num_heads": 8, "num_layers": 2, "ffn_dim": 64, "dropout": 0.131},
    "LSTM_Transformer": {"hidden_dim": 64, "num_lstm_layers": 1, "num_transformer_layers": 2, "num_heads": 4, "ffn_dim": 128, "dropout": 0.384},
    "Parallel_LSTM_Transformer": {"hidden_dim": 32, "num_lstm_layers": 1, "num_transformer_layers": 1, "num_heads": 4, "ffn_dim": 256, "dropout": 0.184},
}


# ── 工具函数 ──────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _build_model(model_name: str, input_dim: int) -> nn.Module:
    kwargs = BEST_ARCH_CONFIGS[model_name]
    if model_name == "LSTM":
        return LSTMModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "Transformer":
        return TransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "LSTM_Transformer":
        return LSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, seq_len=SEQ_LEN, **kwargs)
    elif model_name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    raise ValueError(f"未知模型: {model_name}")


@torch.no_grad()
def _inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]:
    model.eval()
    preds, trues = [],[]
    attn_w_last: NDArray[np.float32] | None = None
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred, extra = model(X)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
        if extra is not None:
            attn_w_last = extra[0].cpu().numpy()
    return np.concatenate(preds), np.concatenate(trues), attn_w_last


# ── 指标计算与作图函数维持原样 ────────────────────────────────────
def compute_metrics(true: NDArray, pred: NDArray) -> dict[str, float]:
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

def plot_predictions(true: NDArray, preds_dict: dict[str, NDArray], save_path: Path, last_n: int = 100) -> None:
    tail = true[-last_n:]
    x = np.arange(len(tail))
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.plot(x, tail, color="black", linewidth=2.0, label="真实值", zorder=5)
    styles =["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, (name, pred) in enumerate(preds_dict.items()):
        ax.plot(x, pred[-last_n:], linestyle=styles[i % len(styles)],
                color=_COLORS.get(name), linewidth=1.4,
                label=_LABELS.get(name, name), alpha=0.85)
    ax.set_xlabel(f"时间步 (最后 {last_n} 天)")
    ax.set_ylabel("沪深300指数点位")
    ax.set_title("测试集预测值与真实值对比", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, save_path)

def plot_loss_curves(histories: dict[str, dict[str, list[float]]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for idx, (key, title) in enumerate([("train_loss", "训练损失"), ("val_loss", "验证损失")]):
        ax: plt.Axes = axes[idx]
        for name, hist in histories.items():
            epochs = range(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], label=_LABELS.get(name, name),
                    color=_COLORS.get(name), linewidth=1.6, alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True, fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.suptitle("四模型训练过程损失对比 (HPO后)", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, save_path)

def plot_attention_heatmap(attn: NDArray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(attn, ax=ax, cmap="YlOrRd", square=True,
                xticklabels=5, yticklabels=5, linewidths=0.15, linecolor="white",
                cbar_kws={"shrink": 0.82, "label": "权重"})
    ax.set_xlabel("Key 时间步")
    ax.set_ylabel("Query 时间步")
    ax.set_title("Serial LSTM-Trans Cross-Attention 权重", fontsize=13, fontweight="bold")
    _save(fig, save_path)

def plot_error_distribution(true: NDArray, preds_dict: dict[str, NDArray], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, pred in preds_dict.items():
        errors = pred - true
        ax.hist(errors, bins=50, alpha=0.45, label=_LABELS.get(name, name),
                color=_COLORS.get(name), edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("预测误差 (预测值 − 真实值)")
    ax.set_ylabel("频数")
    ax.set_title("四模型预测误差分布", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, save_path)


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("评估流水线启动 | 设备: {}", device)

    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)

    # 推理阶段的 dataloader 只需要初始化一次
    _, _, test_loader, scaler_target = get_dataloaders(
        df_values=df.values, columns=columns,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=EVAL_BATCH_SIZE,
        target_col="close", train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
    )

    histories: dict[str, dict[str, list[float]]] = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}_history.json"
        if path.exists():
            histories[name] = json.loads(path.read_text())
    if histories:
        plot_loss_curves(histories, RESULTS_DIR / "loss_curves.png")

    all_metrics: dict[str, dict[str, float]] = {}
    preds_real: dict[str, NDArray] = {}
    true_real: NDArray | None = None
    attn_for_heatmap: NDArray | None = None

    for name in MODEL_NAMES:
        model = _build_model(name, num_features)
        weight_path = MODELS_DIR / f"{name}_best.pth"
        if not weight_path.exists():
            logger.warning("跳过 {}: 权重文件不存在", name)
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)

        preds_norm, trues_norm, attn_w = _inference(model, test_loader, device)

        pred_r = scaler_target.inverse_transform(preds_norm).ravel()
        true_r = scaler_target.inverse_transform(trues_norm).ravel()

        preds_real[name] = pred_r
        if true_real is None:
            true_real = true_r

        m = compute_metrics(true_r, pred_r)
        all_metrics[name] = m
        
        if name == "LSTM_Transformer" and attn_w is not None:
            attn_for_heatmap = attn_w

    if not all_metrics:
        logger.error("未找到任何模型的预测结果，脚本退出。")
        return

    assert true_real is not None
    plot_predictions(true_real, preds_real, RESULTS_DIR / "predictions.png")
    if attn_for_heatmap is not None:
        plot_attention_heatmap(attn_for_heatmap, RESULTS_DIR / "attention_heatmap.png")
    plot_error_distribution(true_real, preds_real, RESULTS_DIR / "error_distribution.png")

    logger.info("=" * 80)
    logger.info("{:<22s} {:>10s} {:>8s} {:>8s} {:>8s} {:>8s} {:>6s}",
                "Model", "MSE", "RMSE", "MAE", "R²", "MAPE%", "DA%")
    logger.info("-" * 80)
    for name, m in all_metrics.items():
        logger.info("{:<22s} {:>10.2f} {:>8.2f} {:>8.2f} {:>8.4f} {:>8.2f} {:>6.2f}",
                     _LABELS.get(name, name), m["MSE"], m["RMSE"], m["MAE"],
                     m["R2"], m["MAPE"], m["DA"])
    logger.info("=" * 80)


if __name__ == "__main__":
    main()