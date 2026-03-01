"""可视化工具箱: 论文级中文图表绘制。

所有函数统一 (data, save_path) 签名，高 DPI 保存，自动创建目录。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray

# ── 全局中文 & 样式 ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})
sns.set_theme(style="whitegrid", font="Microsoft YaHei", palette="muted")

# 三模型统一配色 (学术冷色系)
_MODEL_COLORS: dict[str, str] = {
    "LSTM": "#4C72B0",
    "Transformer": "#DD8452",
    "LSTM_Transformer": "#55A868",
}
_MODEL_LABELS: dict[str, str] = {
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "LSTM_Transformer": "LSTM-Transformer",
}


def _save(fig: plt.Figure, path: str | Path) -> None:
    """保存并关闭图表。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)


# ── 1. 损失曲线对比 ──────────────────────────────────────────────
def plot_loss_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: str | Path = "results/loss_curves.png",
) -> None:
    """1×2 子图: 左 Train Loss, 右 Val Loss, 三模型对比。

    Args:
        histories: {model_name: {"train_loss": [...], "val_loss": [...]}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for panel_idx, (loss_key, title) in enumerate(
        [("train_loss", "训练损失"), ("val_loss", "验证损失")]
    ):
        ax: plt.Axes = axes[panel_idx]
        for name, hist in histories.items():
            epochs = range(1, len(hist[loss_key]) + 1)
            ax.plot(
                epochs,
                hist[loss_key],
                label=_MODEL_LABELS.get(name, name),
                color=_MODEL_COLORS.get(name, None),
                linewidth=1.6,
                alpha=0.9,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    fig.suptitle("三模型训练过程损失对比", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, save_path)


# ── 2. 指标柱状图 ────────────────────────────────────────────────
def plot_metrics_bar(
    metrics: dict[str, dict[str, float]],
    save_path: str | Path = "results/metrics_bar.png",
) -> None:
    """分组柱状图对比 RMSE / MAE / MAPE。

    Args:
        metrics: {model_name: {"RMSE": float, "MAE": float, "MAPE": float}}
    """
    metric_names = ["RMSE", "MAE", "MAPE"]
    model_names = list(metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    x = np.arange(n_metrics)
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for i, name in enumerate(model_names):
        vals = [metrics[name][m] for m in metric_names]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=_MODEL_LABELS.get(name, name),
            color=_MODEL_COLORS.get(name, None),
            edgecolor="white",
            linewidth=0.5,
        )
        # 柱顶标注数值
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("指标值")
    ax.set_title("三模型测试集指标对比", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.grid(axis="y", alpha=0.4)

    _save(fig, save_path)


# ── 3. 预测曲线 ──────────────────────────────────────────────────
def plot_predictions(
    true_values: NDArray[np.float64],
    preds_dict: dict[str, NDArray[np.float64]],
    save_path: str | Path = "results/predictions.png",
    last_n: int = 100,
) -> None:
    """截取最后 last_n 天, 真实值黑色粗线 + 三模型预测线。

    Args:
        true_values: (T,) 真实收盘价 (已反归一化)。
        preds_dict: {model_name: (T,) 预测值}。
    """
    true_tail = true_values[-last_n:]
    x = np.arange(len(true_tail))

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

    ax.plot(x, true_tail, color="black", linewidth=2.0, label="真实值", zorder=5)

    line_styles = ["-", "--", "-."]
    for i, (name, pred) in enumerate(preds_dict.items()):
        pred_tail = pred[-last_n:]
        ax.plot(
            x,
            pred_tail,
            linestyle=line_styles[i % len(line_styles)],
            color=_MODEL_COLORS.get(name, None),
            linewidth=1.4,
            label=_MODEL_LABELS.get(name, name),
            alpha=0.85,
        )

    ax.set_xlabel("时间步 (最后 {} 天)".format(last_n))
    ax.set_ylabel("沪深300指数点位")
    ax.set_title("测试集预测值与真实值对比", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    _save(fig, save_path)


# ── 4. 注意力热力图 ──────────────────────────────────────────────
def plot_attention_heatmap(
    attn_matrix: NDArray[np.float64],
    save_path: str | Path = "results/attention_heatmap.png",
) -> None:
    """绘制 (seq_len, seq_len) 注意力权重热力图。

    Args:
        attn_matrix: 2D 数组, 形如 (30, 30)。
    """
    seq_len = attn_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    sns.heatmap(
        attn_matrix,
        ax=ax,
        cmap="YlOrRd",
        square=True,
        xticklabels=5,
        yticklabels=5,
        linewidths=0.15,
        linecolor="white",
        cbar_kws={"shrink": 0.82, "label": "权重"},
    )
    ax.set_xlabel("Key 时间步")
    ax.set_ylabel("Query 时间步")
    ax.set_title("LSTM-Transformer 自注意力权重", fontsize=13, fontweight="bold")

    _save(fig, save_path)
