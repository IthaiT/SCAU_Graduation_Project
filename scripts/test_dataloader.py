"""验证 DataLoader 张量流转正确性。"""
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.dataset import get_dataloaders  # noqa: E402

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

SEQ_LEN = 30
PRED_LEN = 1
BATCH_SIZE = 64


def main() -> None:
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    logger.info("读取: {} {}", csv_path.name, df.shape)

    columns = df.columns.tolist()
    train_loader, val_loader, test_loader, scaler_target = get_dataloaders(
        df_values=df.values,
        columns=columns,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE,
    )

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        X, y = next(iter(loader))
        logger.info("{:>5s} batch — X: {}  y: {}", name, tuple(X.shape), tuple(y.shape))

    # 验证 inverse_transform 可用
    sample_y = y[:3].numpy()
    restored = scaler_target.inverse_transform(sample_y)
    logger.info("inverse_transform 验证 (前3): {}", restored.flatten().tolist())


if __name__ == "__main__":
    main()
