# pip install pandas_ta
"""技术指标特征工程模块。"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from loguru import logger


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 OHLCV DataFrame 上追加技术指标并清除 NaN 行。

    Args:
        df: 必须包含 open/high/low/close/volume 列，DatetimeIndex。

    Returns:
        追加 MA5/MA20/RSI14/MACD/BBands 后、dropna 清洗完毕的 DataFrame。
    """
    shape_before = df.shape
    logger.info("特征扩充前: {}", shape_before)

    # pandas_ta 批量追加，列名自动全小写
    df = df.copy()
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)

    # 统一列名为全小写
    df.columns = [c.lower() for c in df.columns]

    df = df.dropna()

    logger.info("特征扩充后: {} (丢弃 {} 行 NaN)", df.shape, shape_before[0] - len(df))
    return df
