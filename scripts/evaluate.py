"""M6 评估流水线: 加载权重 → 测试集推理 → 反归一化 → 指标计算 → 图表生成。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.networks import (  # noqa: E402
    LSTMModel,
    LSTMTransformerModel,
    TransformerModel,
)
from src.utils.visualize import (  # noqa: E402
    plot_attention_heatmap,
    plot_loss_curves,
    plot_metrics_bar,
    plot_predictions,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# ── 超参数 (与 train.py 保持一致) ────────────────────────────────
SEQ_LEN = 30
PRED_LEN = 1
BATCH_SIZE = 64
HIDDEN_DIM = 64
NUM_HEADS = 4

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAMES = ["LSTM", "Transformer", "LSTM_Transformer"]


# ── 工具函数 ──────────────────────────────────────────────────────
def _build_model(name: str, input_dim: int) -> nn.Module:
    """按名称构建模型实例。"""
    factory: dict[str, nn.Module] = {
        "LSTM": LSTMModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN),
        "Transformer": TransformerModel(input_dim=input_dim, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
        "LSTM_Transformer": LSTMTransformerModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
    }
    return factory[name]


def _load_histories() -> dict[str, dict[str, list[float]]]:
    """读取三个模型的训练历史 JSON。"""
    histories: dict[str, dict[str, list[float]]] = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}_history.json"
        histories[name] = json.loads(path.read_text())
        logger.info("已读取历史: {} ({} epochs)", name, len(histories[name]["train_loss"]))
    return histories


@torch.no_grad()
def _inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]:
    """测试集前向推理, 返回 (all_preds, all_trues, last_batch_attn)。

    all_preds / all_trues 形状: (N, PRED_LEN), 仍为归一化空间。
    last_batch_attn: 最后一个 batch 的第一条样本的 attn_weights (seq_len, seq_len) 或 None。
    """
    model.eval()
    preds_list: list[np.ndarray] = []
    trues_list: list[np.ndarray] = []
    attn_sample: NDArray[np.float32] | None = None

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred, attn = model(X)
        preds_list.append(pred.cpu().numpy())
        trues_list.append(y.cpu().numpy())
        if attn is not None:
            attn_sample = attn[0].cpu().numpy()  # 取第一条样本 (seq_len, seq_len)

    all_preds = np.concatenate(preds_list, axis=0)
    all_trues = np.concatenate(trues_list, axis=0)
    return all_preds, all_trues, attn_sample


def _compute_metrics(
    true: NDArray[np.float64],
    pred: NDArray[np.float64],
) -> dict[str, float]:
    """计算 RMSE, MAE, MAPE (基于反归一化后的真实点位)。"""
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae = float(np.mean(np.abs(true - pred)))
    mape = float(np.mean(np.abs((true - pred) / (true + 1e-8))) * 100)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("评估设备: {}", device)

    # ❶ 构建 DataLoader (复用训练时同一切分)
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)

    _, _, test_loader, scaler_target = get_dataloaders(
        df_values=df.values,
        columns=columns,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE,
    )

    # ❷ 损失曲线
    histories = _load_histories()
    plot_loss_curves(histories, RESULTS_DIR / "loss_curves.png")
    logger.info("✓ 损失曲线已保存")

    # ❸ 逐模型推理 + 反归一化 + 指标计算
    all_metrics: dict[str, dict[str, float]] = {}
    preds_real: dict[str, NDArray[np.float64]] = {}
    true_real: NDArray[np.float64] | None = None
    attn_for_heatmap: NDArray[np.float32] | None = None

    for name in MODEL_NAMES:
        # 实例化 + 加载权重
        model = _build_model(name, num_features)
        weight_path = MODELS_DIR / f"{name}_best.pth"
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)
        logger.info("已加载: {} ← {}", name, weight_path.name)

        # 前向推理
        preds_norm, trues_norm, attn = _inference(model, test_loader, device)

        # 反归一化 → 真实沪深300点位
        pred_real = scaler_target.inverse_transform(preds_norm)
        true_vals = scaler_target.inverse_transform(trues_norm)

        # 展平为 1D
        pred_real = pred_real.ravel()
        true_vals = true_vals.ravel()

        preds_real[name] = pred_real
        if true_real is None:
            true_real = true_vals

        # 指标
        m = _compute_metrics(true_vals, pred_real)
        all_metrics[name] = m
        logger.info("{} → RMSE={:.2f}  MAE={:.2f}  MAPE={:.2f}%", name, m["RMSE"], m["MAE"], m["MAPE"])

        # 保留 LSTM_Transformer 的注意力权重
        if name == "LSTM_Transformer" and attn is not None:
            attn_for_heatmap = attn

    # ❹ 指标柱状图
    plot_metrics_bar(all_metrics, RESULTS_DIR / "metrics_bar.png")
    logger.info("✓ 指标柱状图已保存")

    # ❺ 预测曲线
    assert true_real is not None
    plot_predictions(true_real, preds_real, RESULTS_DIR / "predictions.png")
    logger.info("✓ 预测曲线已保存")

    # ❻ 注意力热力图
    if attn_for_heatmap is not None:
        plot_attention_heatmap(attn_for_heatmap, RESULTS_DIR / "attention_heatmap.png")
        logger.info("✓ 注意力热力图已保存")

    # ❼ 汇总输出
    logger.info("=" * 50)
    logger.info("评估完成, 所有图表已保存至 {}", RESULTS_DIR)
    for name, m in all_metrics.items():
        logger.info("  {:<20s} RMSE={:<10.2f} MAE={:<10.2f} MAPE={:.2f}%", name, m["RMSE"], m["MAE"], m["MAPE"])


if __name__ == "__main__":
    main()
