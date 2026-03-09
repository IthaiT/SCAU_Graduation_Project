"""Academic-grade evaluation pipeline for binary classification.

Generates publication-ready plots (IEEE/Nature aesthetic, 300+ DPI):
    1. 2×2 Binary Confusion Matrices
    2. ROC Curves with AUC (all models, single chart)
    3. Precision-Recall Curves (all models, single chart)
    4. Training Loss & Accuracy learning curves (2×2)
    5. Attention heatmap + Gate weight distributions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
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

# ── Unified plot style (IEEE/Nature aesthetic) ────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
})
sns.set_theme(style="whitegrid", palette="colorblind", rc=matplotlib.rcParams)

_PALETTE: dict[str, str] = {
    "LSTM": "#0173B2",
    "Transformer": "#DE8F05",
    "LSTM_Transformer": "#029E73",
    "Parallel_LSTM_Transformer": "#D55E00",
}
_LABELS: dict[str, str] = {
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "LSTM_Transformer": "LSTM-Transformer",
    "Parallel_LSTM_Transformer": "Parallel-LSTM-Transformer",
}
_CLASS_NAMES = ["Drop (0)", "Rise (1)"]

SEQ_LEN = 30
NUM_CLASSES = 2
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_HEADS = 4

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAMES = ["LSTM", "Transformer", "LSTM_Transformer", "Parallel_LSTM_Transformer"]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _build_model(name: str, input_dim: int) -> nn.Module:
    common = dict(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
    if name == "LSTM":
        return LSTMModel(**common)
    if name == "Transformer":
        return TransformerModel(input_dim=input_dim, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES)
    if name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(**common, num_heads=NUM_HEADS)
    return LSTMTransformerModel(**common, num_heads=NUM_HEADS)


@torch.no_grad()
def _inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[NDArray, NDArray, NDArray, dict[str, NDArray | None]]:
    """Returns (pred_labels, true_labels, prob_positive, meta)."""
    model.eval()
    all_preds, all_trues, all_probs = [], [], []
    meta: dict[str, NDArray | None] = {"attn_w": None, "gate_lstm": None, "gate_trans": None}

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, extra = model(X)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs[:, 1].cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_trues.append(y.cpu().numpy())

        if isinstance(extra, dict):
            meta["attn_w"] = extra["attn_w"][0].cpu().numpy()
            meta["gate_lstm"] = extra["gate_lstm"].cpu().numpy()
            meta["gate_trans"] = extra["gate_trans"].cpu().numpy()
        elif extra is not None:
            meta["attn_w"] = extra[0].cpu().numpy()

    return np.concatenate(all_preds), np.concatenate(all_trues), np.concatenate(all_probs), meta


def compute_metrics(true: NDArray, pred: NDArray) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(true, pred)), 4),
        "Macro-F1": round(float(f1_score(true, pred, average="macro")), 4),
        "Weighted-F1": round(float(f1_score(true, pred, average="weighted")), 4),
    }


# ── Plot 1: 2×2 Confusion Matrices ──────────────────────────────
def plot_confusion_matrices(
    true: NDArray, preds_dict: dict[str, NDArray], save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, (name, pred) in zip(axes.ravel(), preds_dict.items()):
        cm = confusion_matrix(true, pred, labels=[0, 1])
        cm_pct = cm.astype(float) / cm.sum() * 100
        annot = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                          for row_v, row_p in zip(cm, cm_pct)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                    xticklabels=_CLASS_NAMES, yticklabels=_CLASS_NAMES,
                    linewidths=0.8, linecolor="white", cbar_kws={"shrink": 0.75})
        ax.set_title(_LABELS[name], fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Binary Confusion Matrices", fontsize=14, fontweight="bold")
    _save(fig, save_path)


# ── Plot 2: ROC Curves ──────────────────────────────────────────
def plot_roc_curves(
    true: NDArray,
    probs_dict: dict[str, NDArray],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=_PALETTE[name], lw=2.0, alpha=0.9,
                label=f"{_LABELS[name]} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    _save(fig, save_path)


# ── Plot 3: Precision-Recall Curves ─────────────────────────────
def plot_pr_curves(
    true: NDArray,
    probs_dict: dict[str, NDArray],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    baseline = true.sum() / len(true)

    for name, probs in probs_dict.items():
        precision, recall, _ = precision_recall_curve(true, probs)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=_PALETTE[name], lw=2.0, alpha=0.9,
                label=f"{_LABELS[name]} (AP = {pr_auc:.4f})")

    ax.axhline(y=baseline, color="k", ls="--", lw=1.0, alpha=0.5,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fancybox=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    _save(fig, save_path)


# ── Plot 4: Learning Curves ─────────────────────────────────────
def plot_learning_curves(
    histories: dict[str, dict[str, list[float]]],
    save_path: Path,
) -> None:
    panels = [
        ("train_loss", "Training Loss", "CrossEntropy Loss"),
        ("val_loss", "Validation Loss", "CrossEntropy Loss"),
        ("train_acc", "Training Accuracy", "Accuracy"),
        ("val_acc", "Validation Accuracy", "Accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.ravel(), panels):
        for name, hist in histories.items():
            if key not in hist:
                continue
            epochs = range(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], label=_LABELS.get(name, name),
                    color=_PALETTE.get(name), lw=1.8, alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, fancybox=True)
        ax.grid(alpha=0.3)
    fig.suptitle("Training & Validation Learning Curves", fontsize=14, fontweight="bold")
    _save(fig, save_path)


# ── Plot 5a: Attention Heatmap ───────────────────────────────────
def plot_attention_heatmap(attn: NDArray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(attn, ax=ax, cmap="YlOrRd", square=True,
                xticklabels=5, yticklabels=5, linewidths=0.1, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "Weight"})
    ax.set_xlabel("Key Timestep")
    ax.set_ylabel("Query Timestep")
    ax.set_title("Self-Attention Weights (Last Layer)", fontweight="bold")
    _save(fig, save_path)


# ── Plot 5b: Gate Weight Distributions ───────────────────────────
def plot_gate_weights(
    gate_lstm: NDArray, gate_trans: NDArray, save_path: Path,
) -> None:
    mean_lstm = gate_lstm.mean(axis=0)
    mean_trans = gate_trans.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    dims = np.arange(len(mean_lstm))
    width = 0.4
    axes[0].bar(dims - width / 2, mean_lstm, width, label="LSTM Tower", color=_PALETTE["LSTM"], alpha=0.85)
    axes[0].bar(dims + width / 2, mean_trans, width, label="Transformer Tower", color=_PALETTE["Transformer"], alpha=0.85)
    axes[0].set_xlabel("Hidden Dimension Index")
    axes[0].set_ylabel("Mean Gate Weight")
    axes[0].set_title("Per-Dimension Gate Allocation", fontweight="bold")
    axes[0].legend(frameon=True, fancybox=True)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].hist(gate_lstm.ravel(), bins=50, alpha=0.65, label="LSTM Tower", color=_PALETTE["LSTM"])
    axes[1].hist(gate_trans.ravel(), bins=50, alpha=0.65, label="Transformer Tower", color=_PALETTE["Transformer"])
    axes[1].set_xlabel("Gate Weight Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Gate Weight Distribution", fontweight="bold")
    axes[1].legend(frameon=True, fancybox=True)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Parallel-LSTM-Transformer: Complementary Gate Analysis",
                 fontsize=13, fontweight="bold")
    _save(fig, save_path)


# ── Main pipeline ────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)

    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    num_features = len(df.columns)

    _, _, test_loader = get_dataloaders(df=df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    # Load training histories
    histories: dict[str, dict[str, list[float]]] = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}_history.json"
        if path.exists():
            histories[name] = json.loads(path.read_text())
    if histories:
        plot_learning_curves(histories, RESULTS_DIR / "learning_curves.png")
        logger.info("Plot 4: Learning curves saved")

    # Per-model inference
    all_metrics: dict[str, dict[str, float]] = {}
    preds_dict: dict[str, NDArray] = {}
    probs_dict: dict[str, NDArray] = {}
    true_labels: NDArray | None = None
    attn_for_heatmap: NDArray | None = None
    gate_data: dict[str, NDArray | None] = {"gate_lstm": None, "gate_trans": None}

    for name in MODEL_NAMES:
        model = _build_model(name, num_features)
        weight_path = MODELS_DIR / f"{name}_best.pth"
        if not weight_path.exists():
            logger.warning("Skipping {} — weights not found", name)
            continue
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)

        preds, trues, probs, meta = _inference(model, test_loader, device)
        preds_dict[name] = preds
        probs_dict[name] = probs
        if true_labels is None:
            true_labels = trues

        m = compute_metrics(trues, preds)
        all_metrics[name] = m
        logger.info("{} → Acc={:.4f}  Macro-F1={:.4f}  W-F1={:.4f}",
                     _LABELS[name], m["Accuracy"], m["Macro-F1"], m["Weighted-F1"])

        if name in ("LSTM_Transformer", "Parallel_LSTM_Transformer") and meta["attn_w"] is not None:
            attn_for_heatmap = meta["attn_w"]
        if name == "Parallel_LSTM_Transformer" and meta["gate_lstm"] is not None:
            gate_data["gate_lstm"] = meta["gate_lstm"]
            gate_data["gate_trans"] = meta["gate_trans"]

    assert true_labels is not None

    # Plot 1: Confusion matrices
    plot_confusion_matrices(true_labels, preds_dict, RESULTS_DIR / "confusion_matrices.png")
    logger.info("Plot 1: Confusion matrices saved")

    # Plot 2: ROC curves
    plot_roc_curves(true_labels, probs_dict, RESULTS_DIR / "roc_curves.png")
    logger.info("Plot 2: ROC curves saved")

    # Plot 3: PR curves
    plot_pr_curves(true_labels, probs_dict, RESULTS_DIR / "pr_curves.png")
    logger.info("Plot 3: PR curves saved")

    # Plot 5a: Attention heatmap
    if attn_for_heatmap is not None:
        plot_attention_heatmap(attn_for_heatmap, RESULTS_DIR / "attention_heatmap.png")
        logger.info("Plot 5a: Attention heatmap saved")

    # Plot 5b: Gate weights
    if gate_data["gate_lstm"] is not None:
        plot_gate_weights(gate_data["gate_lstm"], gate_data["gate_trans"],
                          RESULTS_DIR / "gate_weights.png")
        logger.info("Plot 5b: Gate weight distribution saved")

    # Classification report
    report_lines: list[str] = []
    for name in MODEL_NAMES:
        if name not in preds_dict:
            continue
        report_lines.append("=" * 60)
        report_lines.append(f"  {_LABELS[name]}")
        report_lines.append("=" * 60)
        report_lines.append(classification_report(
            true_labels, preds_dict[name], target_names=_CLASS_NAMES,
        ))
    report_path = RESULTS_DIR / "classification_reports.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Classification report saved → {}", report_path)

    # Summary table
    logger.info("=" * 70)
    logger.info("{:<28s} {:>10s} {:>10s} {:>12s}", "Model", "Accuracy", "Macro-F1", "Weighted-F1")
    logger.info("-" * 70)
    for name, m in all_metrics.items():
        logger.info("{:<28s} {:>10.4f} {:>10.4f} {:>12.4f}",
                     _LABELS[name], m["Accuracy"], m["Macro-F1"], m["Weighted-F1"])
    logger.info("=" * 70)
    logger.info("All plots saved to {}", RESULTS_DIR)


if __name__ == "__main__":
    main()
