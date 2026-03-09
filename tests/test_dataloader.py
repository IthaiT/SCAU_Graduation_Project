"""V5-beta: 验证 DataLoader 张量流转正确性 (仅收盘价, seq_len=60)。"""
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.dataset import get_dataloaders  # noqa: E402

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# 超参数 (与 train.py 一致)
SEQ_LEN = 60
PRED_LEN = 1
BATCH_SIZE = 32
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10


def main() -> None:
    # 仅使用收盘价 (单特征)
    csv_path = PROJECT_ROOT / "data" / "csi300_raw.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df_close = df[["close"]].copy()
    logger.info("读取: {} {} → 仅 close 列", csv_path.name, df.shape)

    columns = df_close.columns.tolist()
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

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        X, y = next(iter(loader))
        logger.info("{:>5s} batch — X: {}  y: {}", name, tuple(X.shape), tuple(y.shape))
        assert X.shape[-1] == 1, f"输入特征维度应为 1 (收盘价), 实际: {X.shape[-1]}"

    # 验证 inverse_transform 可用
    sample_y = y[:3].numpy()
    restored = scaler_target.inverse_transform(sample_y)
    logger.info("inverse_transform 验证 (前3): {}", restored.flatten().tolist())


if __name__ == "__main__":
    main()
