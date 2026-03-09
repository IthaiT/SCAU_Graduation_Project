"""验证 DataLoader 张量流转正确性 (三分类)。"""
import sys
from pathlib import Path

import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.dataset import get_dataloaders  # noqa: E402

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

SEQ_LEN = 30
BATCH_SIZE = 64


def main() -> None:
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    logger.info("读取: {} {}", csv_path.name, df.shape)

    columns = df.columns.tolist()
    train_loader, val_loader, test_loader = get_dataloaders(
        df_values=df.values,
        columns=columns,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        X, y = next(iter(loader))
        logger.info("{:>5s} batch — X: {}  y: {} dtype={}", name, tuple(X.shape), tuple(y.shape), y.dtype)
        assert X.dim() == 3 and X.shape[1] == SEQ_LEN
        assert y.dim() == 1 and y.dtype == torch.long
        assert y.min() >= 0 and y.max() <= 2

    logger.info("全部断言通过 ✓")


if __name__ == "__main__":
    main()
