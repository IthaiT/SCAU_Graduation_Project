"""学术级评估流水线: 加载权重 → 推理 → 分类指标 → 论文级图表。

生成图表:
    1. 四模型混淆矩阵 (2×2 子图)
    2. Accuracy / Macro-F1 对比柱状图
    3. 训练 Loss & Accuracy 曲线 (2×2 子图)
    4. Attention 热力图 (LSTM-Transformer 最后一层)
    5. 门控权重分布 (Parallel 模型)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

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
_CLASS_NAMES = ["跌 (Drop)", "平 (Flat)", "涨 (Rise)"]

# ── 超参数 (与 train.py 一致) ────────────────────────────────────
SEQ_LEN = 30
NUM_CLASSES = 3
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
        return LSTMModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
    if name == "Transformer":
        return TransformerModel(input_dim=input_dim, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES)
    if name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES)
    return LSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES)


@torch.no_grad()
def _inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, NDArray[np.float32] | None]]:
    """返回 (pred_labels, true_labels, meta)。"""
    model.eval()
    all_preds, all_trues = [], []
    meta: dict[str, NDArray[np.float32] | None] = {"attn_w": None, "gate_lstm": None, "gate_trans": None}
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, extra = model(X)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_trues.append(y.cpu().numpy())
        if isinstance(extra, dict):
            meta["attn_w"] = extra["attn_w"][0].cpu().numpy()
            meta["gate_lstm"] = extra["gate_lstm"].cpu().numpy()
            meta["gate_trans"] = extra["gate_trans"].cpu().numpy()
        elif extra is not None:
            meta["attn_w"] = extra[0].cpu().numpy()
    return np.concatenate(all_preds), np.concatenate(all_trues), meta


# ── 指标计算 ──────────────────────────────────────────────────────
def compute_metrics(true: NDArray, pred: NDArray) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(true, pred)), 4),
        "Macro-F1": round(float(f1_score(true, pred, average="macro")), 4),
        "Weighted-F1": round(float(f1_score(true, pred, average="weighted")), 4),
    }


# ── 图1: 混淆矩阵 (2×2) ─────────────────────────────────────────
def plot_confusion_matrices(
    true: NDArray,
    preds_dict: dict[str, NDArray],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (name, pred) in zip(axes.ravel(), preds_dict.items()):
        cm = confusion_matrix(true, pred, labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=_CLASS_NAMES, yticklabels=_CLASS_NAMES,
                    linewidths=0.5, linecolor="white")
        ax.set_title(_LABELS.get(name, name), fontsize=12, fontweight="bold")
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")
    fig.suptitle("四模型混淆矩阵对比", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, save_path)


# ── 图2: Accuracy & F1 柱状图 ────────────────────────────────────
def plot_metrics_bar(
    metrics_dict: dict[str, dict[str, float]],
    save_path: Path,
) -> None:
    names = list(metrics_dict.keys())
    labels = [_LABELS.get(n, n) for n in names]
    acc = [metrics_dict[n]["Accuracy"] for n in names]
    f1 = [metrics_dict[n]["Macro-F1"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars1 = ax.bar(x - width / 2, acc, width, label="Accuracy", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1, width, label="Macro-F1", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("四模型分类性能对比", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3, axis="y")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    _save(fig, save_path)


# ── 图3: Loss & Accuracy 训练曲线 (2×2) ──────────────────────────
def plot_loss_accuracy_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    panels = [
        ("train_loss", "训练损失", "CrossEntropy Loss"),
        ("val_loss", "验证损失", "CrossEntropy Loss"),
        ("train_acc", "训练准确率", "Accuracy"),
        ("val_acc", "验证准确率", "Accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.ravel(), panels):
        for name, hist in histories.items():
            if key not in hist:
                continue
            epochs = range(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], label=_LABELS.get(name, name),
                    color=_COLORS.get(name), linewidth=1.6, alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(frameon=True, fancybox=True, fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("四模型训练过程对比", fontsize=14, fontweight="bold", y=1.02)
    _save(fig, save_path)


# ── 图4: Attention 热力图 ────────────────────────────────────────
def plot_attention_heatmap(attn: NDArray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(attn, ax=ax, cmap="YlOrRd", square=True,
                xticklabels=5, yticklabels=5, linewidths=0.15, linecolor="white",
                cbar_kws={"shrink": 0.82, "label": "权重"})
    ax.set_xlabel("Key 时间步")
    ax.set_ylabel("Query 时间步")
    ax.set_title("LSTM-Transformer 自注意力权重 (最后一层)", fontsize=13, fontweight="bold")
    _save(fig, save_path)


# ── 图5: 动态门控权重分布 ────────────────────────────────────────
def plot_gate_weights(
    gate_lstm: NDArray, gate_trans: NDArray, save_path: Path,
) -> None:
    mean_lstm = gate_lstm.mean(axis=0)
    mean_trans = gate_trans.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax: plt.Axes = axes[0]
    dims = np.arange(len(mean_lstm))
    ax.bar(dims - 0.2, mean_lstm, 0.4, label="LSTM 塔", color="#4C72B0", alpha=0.85)
    ax.bar(dims + 0.2, mean_trans, 0.4, label="Transformer 塔", color="#DD8452", alpha=0.85)
    ax.set_xlabel("隐藏维度索引")
    ax.set_ylabel("平均门控权重 (Sigmoid 输出)")
    ax.set_title("各隐藏维度的门控权重", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, fontsize=9)
    ax.grid(alpha=0.3, axis="y")

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

    _, _, test_loader = get_dataloaders(
        df_values=df.values, columns=columns,
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    )

    # ❶ Loss & Accuracy 曲线
    histories: dict[str, dict[str, list[float]]] = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}_history.json"
        histories[name] = json.loads(path.read_text())
    plot_loss_accuracy_curves(histories, RESULTS_DIR / "loss_accuracy_curves.png")
    logger.info("✓ 图3: Loss & Accuracy 曲线已保存")

    # ❷ 逐模型推理
    all_metrics: dict[str, dict[str, float]] = {}
    preds_dict: dict[str, NDArray] = {}
    true_labels: NDArray | None = None
    attn_for_heatmap: NDArray | None = None
    gate_data: dict[str, NDArray | None] = {"gate_lstm": None, "gate_trans": None}

    for name in MODEL_NAMES:
        model = _build_model(name, num_features)
        weight_path = MODELS_DIR / f"{name}_best.pth"
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)

        preds, trues, meta = _inference(model, test_loader, device)
        preds_dict[name] = preds
        if true_labels is None:
            true_labels = trues

        m = compute_metrics(trues, preds)
        all_metrics[name] = m
        logger.info("{} → Accuracy={:.4f}  Macro-F1={:.4f}  Weighted-F1={:.4f}",
                     name, m["Accuracy"], m["Macro-F1"], m["Weighted-F1"])

        if name in ("LSTM_Transformer", "Parallel_LSTM_Transformer") and meta["attn_w"] is not None:
            attn_for_heatmap = meta["attn_w"]
        if name == "Parallel_LSTM_Transformer" and meta["gate_lstm"] is not None:
            gate_data["gate_lstm"] = meta["gate_lstm"]
            gate_data["gate_trans"] = meta["gate_trans"]

    assert true_labels is not None

    # ❸ 图1: 混淆矩阵
    plot_confusion_matrices(true_labels, preds_dict, RESULTS_DIR / "confusion_matrices.png")
    logger.info("✓ 图1: 混淆矩阵已保存")

    # ❹ 图2: Accuracy & F1 柱状图
    plot_metrics_bar(all_metrics, RESULTS_DIR / "metrics_bar.png")
    logger.info("✓ 图2: 分类性能柱状图已保存")

    # ❺ 图4: Attention 热力图
    if attn_for_heatmap is not None:
        plot_attention_heatmap(attn_for_heatmap, RESULTS_DIR / "attention_heatmap.png")
        logger.info("✓ 图4: Attention 热力图已保存")

    # ❻ 门控权重可视化
    if gate_data["gate_lstm"] is not None:
        plot_gate_weights(gate_data["gate_lstm"], gate_data["gate_trans"],
                          RESULTS_DIR / "gate_weights.png")
        logger.info("✓ 图5: 门控权重分布已保存")

    # ❼ Classification Report
    report_lines: list[str] = []
    for name in MODEL_NAMES:
        report_lines.append("=" * 60)
        report_lines.append(f"  {_LABELS.get(name, name)}")
        report_lines.append("=" * 60)
        report_lines.append(classification_report(
            true_labels, preds_dict[name], target_names=_CLASS_NAMES,
        ))
    report_path = RESULTS_DIR / "classification_reports.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("✓ 分类报告已保存 → {}", report_path)

    # ❽ 汇总
    logger.info("=" * 70)
    logger.info("{:<28s} {:>10s} {:>10s} {:>12s}", "Model", "Accuracy", "Macro-F1", "Weighted-F1")
    logger.info("-" * 70)
    for name, m in all_metrics.items():
        logger.info("{:<28s} {:>10.4f} {:>10.4f} {:>12.4f}",
                     _LABELS.get(name, name), m["Accuracy"], m["Macro-F1"], m["Weighted-F1"])
    logger.info("=" * 70)
    logger.info("所有图表已保存至 {}", RESULTS_DIR)


if __name__ == "__main__":
    main()
