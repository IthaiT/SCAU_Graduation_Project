"""三模型对比训练: LSTM / Transformer / LSTM+Transformer。"""
import json
import sys
from pathlib import Path

from loguru import logger

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

# ── 超参数 ────────────────────────────────────────────────────────
SEQ_LEN = 30
PRED_LEN = 1
BATCH_SIZE = 32
HIDDEN_DIM = 64
NUM_HEADS = 4
EPOCHS = 50
PATIENCE = 10
LR = 5e-4


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: {}", device)

    # 数据
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)

    train_loader, val_loader, test_loader, scaler_target = get_dataloaders(  # test_loader / scaler_target 留给 M6 评估阶段使用
        df_values=df.values,
        columns=columns,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE,
    )

    # 四模型实例化
    models: dict[str, nn.Module] = {
        "LSTM": LSTMModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN),
        "Transformer": TransformerModel(input_dim=num_features, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
        "LSTM_Transformer": LSTMTransformerModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
        "Parallel_LSTM_Transformer": ParallelLSTMTransformerModel(input_dim=num_features, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
    }

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        logger.info("=" * 40)
        logger.info("开始训练: {} ({:,} params)", model_name, sum(p.numel() for p in model.parameters()))

        criterion = nn.HuberLoss(delta=1.0)  # V4: 鲁棒损失，抑制异常值极端梯度

        # V5: 统一训练配方，确保公平对比（架构是唯一变量）
        lr = LR
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        logger.info("  LR={:.1e}, Scheduler={}", lr, type(scheduler).__name__)

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
            scheduler=scheduler,
        )

        # 持久化训练历史
        history_path = out_dir / f"{model_name}_history.json"
        history_path.write_text(json.dumps(history, indent=2))
        logger.info("历史已保存 → {}", history_path)


if __name__ == "__main__":
    main()
