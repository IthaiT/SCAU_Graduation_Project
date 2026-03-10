"""V6 四模型对比训练: LSTM / Transformer / Serial LSTM-Trans / Parallel LSTM-Trans。

适配 CSI 300 长周期数据:
- 输入: 17 维技术指标特征 (input_dim=17)
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
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# ── 超参数 (四模型统一训练策略，仅保留架构差异) ──────────────
SEQ_LEN = 60
PRED_LEN = 1
BATCH_SIZE = 32
LR = 0.001

# 模型架构参数
LSTM_HIDDEN = 60      # LSTM / Transformer
NUM_HEADS = 5         # Transformer
HYBRID_HIDDEN = 64    # Serial / Parallel
HYBRID_HEADS = 4      # Serial / Parallel

# 统一训练配置 (公平对比)
EPOCHS = 100
PATIENCE = 20

# 数据切分
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: {}", device)

    # ── 数据: 17 维技术指标特征 ──
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)  # = 17

    train_loader, val_loader, test_loader, scaler_target = get_dataloaders(
        df_values=df.values,
        columns=columns,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE,
        target_col="close",
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    # ── 四模型实例化 ──
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
        "LSTM_Transformer": LSTMTransformerModel(
            input_dim=num_features, hidden_dim=HYBRID_HIDDEN,
            num_lstm_layers=2, num_heads=HYBRID_HEADS,
            num_transformer_layers=2, ffn_dim=256,
            pred_len=PRED_LEN, dropout=0.2,
        ),
        "Parallel_LSTM_Transformer": ParallelLSTMTransformerModel(
            input_dim=num_features, hidden_dim=HYBRID_HIDDEN,
            num_lstm_layers=2, num_heads=HYBRID_HEADS,
            num_transformer_layers=2, ffn_dim=256,
            pred_len=PRED_LEN, dropout=0.2,
        ),
    }

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        logger.info("=" * 40)
        logger.info("开始训练: {} ({:,} params)", model_name, sum(p.numel() for p in model.parameters()))

        # 统一训练策略: MSELoss + Adam(lr=1e-3) + 无调度器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=EPOCHS,
            patience=PATIENCE,
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
