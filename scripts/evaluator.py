"""学术级评估流水线: 加载权重 → 推理 → 指标计算 → 四张论文级图表。

生成图表:
    1. 三模型预测值与真实值拟合曲线
    2. 训练/验证 Loss 下降曲线
    3. Attention 热力图 (LSTM-Transformer 最后一层)
    4. 预测误差分布直方图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from loguru import logger  # noqa: E402

from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.networks import (  # noqa: E402
    LSTMModel,
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# ── 全局绘图风格 ──────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
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
    "LSTM_Transformer": "#55A868",
    "Parallel_LSTM_Transformer": "#C44E52",
}
_LABELS: dict[str, str] = {
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "LSTM_Transformer": "LSTM-Transformer",
    "Parallel_LSTM_Transformer": "Parallel-LSTM-Transformer",
}

# ── 超参数 (与 train.py 一致) ────────────────────────────────────
SEQ_LEN = 30
PRED_LEN = 1
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_HEADS = 4

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAMES = ["LSTM", "Transformer", "LSTM_Transformer", "Parallel_LSTM_Transformer"]


# ── 工具函数 ──────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _build_model(name: str, input_dim: int) -> nn.Module:
    if name == "LSTM":
        return LSTMModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN)
    if name == "Transformer":
        return TransformerModel(input_dim=input_dim, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN)
    if name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN)
    return LSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN)


@torch.no_grad()
def _inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, NDArray[np.float32] | None]]:
    """返回 (preds, trues, meta)。meta 包含 attn_w / gate_lstm / gate_trans。"""
    model.eval()
    preds, trues = [], []
    meta: dict[str, NDArray[np.float32] | None] = {"attn_w": None, "gate_lstm": None, "gate_trans": None}
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred, extra = model(X)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
        # extra 可能是 Tensor (attn_w)、dict (含 gate 信息) 或 None
        if isinstance(extra, dict):
            meta["attn_w"] = extra["attn_w"][0].cpu().numpy()
            meta["gate_lstm"] = extra["gate_lstm"].cpu().numpy()
            meta["gate_trans"] = extra["gate_trans"].cpu().numpy()
        elif extra is not None:
            meta["attn_w"] = extra[0].cpu().numpy()
    return np.concatenate(preds), np.concatenate(trues), meta


# ── 指标计算 ──────────────────────────────────────────────────────
def compute_metrics(true: NDArray, pred: NDArray) -> dict[str, float]:
    """计算 MSE, RMSE, MAE, MAPE, R², DA。"""
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


# ── 图1: 预测值 vs 真实值 ────────────────────────────────────────
def plot_predictions(
    true: NDArray,
    preds_dict: dict[str, NDArray],
    save_path: Path,
    last_n: int = 100,
) -> None:
    tail = true[-last_n:]
    x = np.arange(len(tail))
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.plot(x, tail, color="black", linewidth=2.0, label="真实值", zorder=5)
    styles = ["-", "--", "-.", ":"]
    for i, (name, pred) in enumerate(preds_dict.items()):
        ax.plot(x, pred[-last_n:], linestyle=styles[i % 3],
                color=_COLORS.get(name), linewidth=1.4,
                label=_LABELS.get(name, name), alpha=0.85)
    ax.set_xlabel("时间步 (最后 {} 天)".format(last_n))
    ax.set_ylabel("沪深300指数点位")
    ax.set_title("测试集预测值与真实值对比", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, save_path)


# ── 图2: Loss 曲线 ───────────────────────────────────────────────
def plot_loss_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
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
    fig.suptitle("四模型训练过程损失对比", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, save_path)


# ── 图3: Attention 热力图 ────────────────────────────────────────
def plot_attention_heatmap(attn: NDArray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(attn, ax=ax, cmap="YlOrRd", square=True,
                xticklabels=5, yticklabels=5, linewidths=0.15, linecolor="white",
                cbar_kws={"shrink": 0.82, "label": "权重"})
    ax.set_xlabel("Key 时间步")
    ax.set_ylabel("Query 时间步")
    ax.set_title("LSTM-Transformer 自注意力权重 (最后一层)", fontsize=13, fontweight="bold")
    _save(fig, save_path)


# ── 图4: 误差分布直方图 ──────────────────────────────────────────
def plot_error_distribution(
    true: NDArray,
    preds_dict: dict[str, NDArray],
    save_path: Path,
) -> None:
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


# ── 图5: 动态门控权重分布 ────────────────────────────────────────
def plot_gate_weights(
    gate_lstm: NDArray, gate_trans: NDArray, save_path: Path,
) -> None:
    """可视化 Parallel 模型的门控权重分布，展示双塔动态加权。"""
    mean_lstm = gate_lstm.mean(axis=0)  # (hidden_dim,)
    mean_trans = gate_trans.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # 左图: 每个隐藏维度上两塔的平均门控值
    ax: plt.Axes = axes[0]
    dims = np.arange(len(mean_lstm))
    ax.bar(dims - 0.2, mean_lstm, 0.4, label="LSTM 塔", color="#4C72B0", alpha=0.85)
    ax.bar(dims + 0.2, mean_trans, 0.4, label="Transformer 塔", color="#DD8452", alpha=0.85)
    ax.set_xlabel("隐藏维度索引")
    ax.set_ylabel("平均门控权重 (Sigmoid 输出)")
    ax.set_title("各隐藏维度的门控权重", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 右图: 两塔门控值的整体分布直方图
    ax2: plt.Axes = axes[1]
    ax2.hist(gate_lstm.ravel(), bins=50, alpha=0.6, label="LSTM 塔", color="#4C72B0")
    ax2.hist(gate_trans.ravel(), bins=50, alpha=0.6, label="Transformer 塔", color="#DD8452")
    ax2.set_xlabel("门控权重值")
    ax2.set_ylabel("频数")
    ax2.set_title("门控权重整体分布", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fancybox=True, fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Parallel-LSTM-Transformer 动态门控权重分析", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, save_path)


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("评估设备: {}", device)

    # 数据
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)

    _, _, test_loader, scaler_target = get_dataloaders(
        df_values=df.values, columns=columns,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=BATCH_SIZE,
    )

    # ❶ Loss 曲线
    histories: dict[str, dict[str, list[float]]] = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}_history.json"
        histories[name] = json.loads(path.read_text())
    plot_loss_curves(histories, RESULTS_DIR / "loss_curves.png")
    logger.info("✓ 图2: Loss 曲线已保存")

    # ❷ 逐模型推理
    all_metrics: dict[str, dict[str, float]] = {}
    preds_real: dict[str, NDArray] = {}
    true_real: NDArray | None = None
    attn_for_heatmap: NDArray | None = None
    gate_data: dict[str, NDArray | None] = {"gate_lstm": None, "gate_trans": None}

    for name in MODEL_NAMES:
        model = _build_model(name, num_features)
        weight_path = MODELS_DIR / f"{name}_best.pth"
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)

        preds_norm, trues_norm, meta = _inference(model, test_loader, device)

        pred_r = scaler_target.inverse_transform(preds_norm).ravel()
        true_r = scaler_target.inverse_transform(trues_norm).ravel()

        preds_real[name] = pred_r
        if true_real is None:
            true_real = true_r

        m = compute_metrics(true_r, pred_r)
        all_metrics[name] = m
        logger.info("{} → MSE={:.2f} RMSE={:.2f} MAE={:.2f} R²={:.4f} MAPE={:.2f}% DA={:.2f}%",
                     name, m["MSE"], m["RMSE"], m["MAE"], m["R2"], m["MAPE"], m["DA"])

        if name in ("LSTM_Transformer", "Parallel_LSTM_Transformer") and meta["attn_w"] is not None:
            attn_for_heatmap = meta["attn_w"]
        if name == "Parallel_LSTM_Transformer" and meta["gate_lstm"] is not None:
            gate_data["gate_lstm"] = meta["gate_lstm"]
            gate_data["gate_trans"] = meta["gate_trans"]

    assert true_real is not None

    # ❸ 图1: 预测曲线
    plot_predictions(true_real, preds_real, RESULTS_DIR / "predictions.png")
    logger.info("✓ 图1: 预测曲线已保存")

    # ❹ 图3: Attention 热力图
    if attn_for_heatmap is not None:
        plot_attention_heatmap(attn_for_heatmap, RESULTS_DIR / "attention_heatmap.png")
        logger.info("✓ 图3: Attention 热力图已保存")

    # ❺ 图4: 误差分布
    plot_error_distribution(true_real, preds_real, RESULTS_DIR / "error_distribution.png")
    logger.info("✓ 图4: 误差分布直方图已保存")

    # ❻ 门控权重可视化
    if gate_data["gate_lstm"] is not None:
        plot_gate_weights(gate_data["gate_lstm"], gate_data["gate_trans"],
                          RESULTS_DIR / "gate_weights.png")
        logger.info("✓ 图5: 门控权重分布已保存")

    # ❼ 汇总
    logger.info("=" * 80)
    logger.info("{:<22s} {:>10s} {:>8s} {:>8s} {:>8s} {:>8s} {:>6s}",
                "Model", "MSE", "RMSE", "MAE", "R²", "MAPE%", "DA%")
    logger.info("-" * 80)
    for name, m in all_metrics.items():
        logger.info("{:<22s} {:>10.2f} {:>8.2f} {:>8.2f} {:>8.4f} {:>8.2f} {:>6.2f}",
                     _LABELS.get(name, name), m["MSE"], m["RMSE"], m["MAE"],
                     m["R2"], m["MAPE"], m["DA"])
    logger.info("=" * 80)
    logger.info("所有图表已保存至 {}", RESULTS_DIR)


if __name__ == "__main__":
    main()
