"""V5-beta 三模型对比训练: LSTM / Transformer / LSTM-mTrans-MLP。

基于论文 "LSTM–Transformer-Based Robust Hybrid Deep Learning Model
for Financial Time Series Forecasting" (Sci 2025, 7, 7) 架构,
适配 CSI 300 长周期数据:
- 输入: 仅收盘价 (单特征, input_dim=1)
- 序列长度: 60
- Loss: MSE
- Optimizer: Adam(lr=0.001)
- Batch size: 32 (适配 20 年高噪声数据)
- Normalization: MinMaxScaler [0,1]
"""
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.data.dataset import get_dataloaders  # noqa: E402
from src.engine.trainer import train_model  # noqa: E402
from src.models.networks import (  # noqa: E402
    LSTMModel,
    LSTMmTransMLPModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# ── 超参数 ────────────────────────────────────────────────────────
SEQ_LEN = 60
PRED_LEN = 1
BATCH_SIZE = 32
LR = 0.001
LSTM_HIDDEN = 60
NUM_HEADS = 5
HEAD_DIM = 120

# LSTM / Transformer baseline
EPOCHS = 30
PATIENCE = 10

# LSTM-mTrans-MLP 需要更多轮次
# (d_model=1 上的 LayerNorm 退化, ReZero 初始化让 mTrans 从零学起)
MTRANS_EPOCHS = 100
MTRANS_PATIENCE = 20

# 数据切分
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: {}", device)

    # ── 数据: 仅使用收盘价 (单特征) ──
    csv_path = PROJECT_ROOT / "data" / "csi300_raw.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df_close = df[["close"]].copy()
    columns = df_close.columns.tolist()
    num_features = len(columns)  # = 1

    train_loader, val_loader, test_loader, scaler_target = get_dataloaders(
        df_values=df_close.values,
        columns=columns,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE,
        target_col="close",
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    # ── 三模型实例化 ──
    models: dict[str, nn.Module] = {
        "LSTM": LSTMModel(
            input_dim=num_features, hidden_dim=LSTM_HIDDEN,
            num_layers=2, pred_len=PRED_LEN, dropout=0.1,
        ),
        "Transformer": TransformerModel(
            input_dim=num_features, d_model=LSTM_HIDDEN,
            num_heads=NUM_HEADS, num_layers=2,
            ffn_dim=128, pred_len=PRED_LEN, dropout=0.15,
        ),
        "LSTM_mTrans_MLP": LSTMmTransMLPModel(
            input_dim=num_features, lstm_hidden=LSTM_HIDDEN,
            num_lstm_layers=2, num_heads=NUM_HEADS,
            head_dim=HEAD_DIM, ffn_mid=5, pred_len=PRED_LEN,
            lstm_dropout=0.1, trans_dropout=0.15, mlp_dropout=0.1,
        ),
    }

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        logger.info("=" * 40)
        logger.info("开始训练: {} ({:,} params)", model_name, sum(p.numel() for p in model.parameters()))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        ep = MTRANS_EPOCHS if model_name == "LSTM_mTrans_MLP" else EPOCHS
        pat = MTRANS_PATIENCE if model_name == "LSTM_mTrans_MLP" else PATIENCE

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=ep,
            patience=pat,
            save_path=out_dir / f"{model_name}_best.pth",
            scheduler=None,
            max_grad_norm=1.0,
        )

        # 持久化训练历史
        history_path = out_dir / f"{model_name}_history.json"
        history_path.write_text(json.dumps(history, indent=2))
        logger.info("历史已保存 → {}", history_path)


if __name__ == "__main__":
    main()
