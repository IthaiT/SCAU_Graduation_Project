# pip install akshare yfinance pandas
"""沪深300 日线数据获取与清洗模块。

数据源优先级: akshare/新浪 → akshare/东方财富 → yfinance。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

_KEEP_COLS: list[str] = ["open", "high", "low", "close", "volume", "amount"]

# ── akshare symbol 与 yfinance ticker 的映射 ─────────────────────
_YF_TICKER_MAP: dict[str, str] = {
    "sh000300": "000300.SS",
    "sh000001": "000001.SS",
    "sz399006": "399006.SZ",
}


def _to_date_range(start: str, end: str) -> tuple[str, str]:
    """YYYYMMDD → YYYY-MM-DD。"""
    fmt = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:]}"  # noqa: E731
    return fmt(start), fmt(end)


def _fetch_akshare_sina(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """akshare 新浪源 — 无需日期参数，拉全量后截取。"""
    import akshare as ak

    raw = ak.stock_zh_index_daily(symbol=symbol)
    df = (
        raw.assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .sort_index()
    )
    # 新浪源无 amount，用 close * volume 近似
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    start, end = _to_date_range(start_date, end_date)
    return df.loc[start:end, _KEEP_COLS]


def _fetch_akshare_em(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """akshare 东方财富源。"""
    import akshare as ak

    raw = ak.stock_zh_index_daily_em(
        symbol=symbol, start_date=start_date, end_date=end_date,
    )
    return (
        raw.loc[:, ["date"] + _KEEP_COLS]
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .sort_index()
    )


def _fetch_yfinance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Yahoo Finance 源。"""
    import yfinance as yf

    ticker = _YF_TICKER_MAP.get(symbol, symbol)
    start, end = _to_date_range(start_date, end_date)

    raw: pd.DataFrame = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"yfinance 返回空数据: {ticker}")

    # yfinance MultiIndex columns → 压平
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    col_map = {"Open": "open", "High": "high", "Low": "low",
               "Close": "close", "Volume": "volume"}
    df = raw.rename(columns=col_map)
    df.index.name = "date"

    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]

    return df.loc[:, _KEEP_COLS]


def fetch_and_clean_data(
    symbol: str = "sh000300",
    start_date: str = "20150101",
    end_date: str = "20240101",
) -> pd.DataFrame:
    """获取指数日线并返回清洗后的 DataFrame。

    按 新浪 → 东方财富 → yfinance 顺序尝试，任一成功即返回。

    Args:
        symbol: 带市场前缀的指数代码 (sh/sz/csi)，默认沪深300。
        start_date: 起始日期 YYYYMMDD。
        end_date: 结束日期 YYYYMMDD。

    Returns:
        DatetimeIndex('date') 索引、全小写列名的 DataFrame。
    """
    fetchers = [
        ("akshare/sina", _fetch_akshare_sina),
        ("akshare/em", _fetch_akshare_em),
        ("yfinance", _fetch_yfinance),
    ]

    last_err: Exception | None = None
    for name, fn in fetchers:
        try:
            logger.info("[{}] 拉取 {} [{}, {}] ...", name, symbol, start_date, end_date)
            df = fn(symbol, start_date, end_date)
            if df.empty:
                raise ValueError("返回空数据")
            break
        except Exception as e:
            logger.warning("[{}] 失败: {}", name, e)
            last_err = e
    else:
        raise RuntimeError(f"所有数据源均失败，最后错误: {last_err}") from last_err

    # 前向填充: 金融时序处理停牌/缺失的行业惯例
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        logger.warning("发现 {} 个缺失值，执行 ffill", n_missing)
    df = df.ffill()

    logger.info("清洗完成 [{}]: {} 行 × {} 列", name, *df.shape)
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """持久化到 CSV，自动建目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("已保存 → {} ({} 行)", path, len(df))
    return path
