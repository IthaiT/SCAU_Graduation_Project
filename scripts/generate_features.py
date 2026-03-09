"""读取原始数据 → 特征工程 → CEEMDAN 分解 → 输出 CSV。"""
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.features import apply_ceemdan, build_features  # noqa: E402

# loguru 配置: 去掉默认 stderr handler，重新添加自定义格式
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")


def main() -> None:
    raw_path = PROJECT_ROOT / "data" / "csi300_raw.csv"
    out_path = PROJECT_ROOT / "data" / "csi300_features.csv"

    df = pd.read_csv(raw_path, index_col="date", parse_dates=True)
    logger.info("读取原始数据: {}", raw_path)

    df = build_features(df)

    # CEEMDAN 多尺度分解: 将 close 分解为多个 IMF 分量 + 残差
    df = apply_ceemdan(df, column="close")

    df.to_csv(out_path)
    logger.info("已保存 → {} ({} 行)", out_path, len(df))
    logger.info("预览:\n{}", df.head())


if __name__ == "__main__":
    main()
