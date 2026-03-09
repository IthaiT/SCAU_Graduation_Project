"""Backward-looking technical indicator feature engineering.

All features are strictly causal — no future information leakage.
CEEMDAN removed: global decomposition on the full series is inherently leaky.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from loguru import logger


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append pandas-TA indicators to OHLCV DataFrame. All backward-looking."""
    n_before = len(df)
    logger.info("特征扩充前: {}", df.shape)

    df = df.copy()

    # Trend
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True)
    df.ta.macd(append=True)

    # Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)
    df.ta.willr(length=14, append=True)
    df.ta.roc(length=10, append=True)

    # Volatility
    df.ta.bbands(append=True)
    df.ta.atr(length=14, append=True)

    # Volume
    df.ta.obv(append=True)

    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()

    logger.info("特征扩充后: {} (丢弃 {} 行 NaN)", df.shape, n_before - len(df))
    return df


def rolling_zscore(df: pd.DataFrame, window: int = 60, min_periods: int = 20) -> pd.DataFrame:
    """Strictly causal rolling z-score: z_t = (x_t - mu_{t-w:t}) / sigma_{t-w:t}.

    Uses expanding window during warm-up (< window samples) to avoid excessive NaN loss.
    Columns with zero variance within a window get z=0 (safe default).
    """
    roll = df.rolling(window=window, min_periods=min_periods)
    mu = roll.mean()
    sigma = roll.std(ddof=1)
    sigma = sigma.replace(0, 1.0)  # avoid division by zero
    z = (df - mu) / sigma
    return z.dropna()
