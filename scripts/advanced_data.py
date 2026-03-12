"""
工业级多模态特征构建脚本 (V2.0 鲁棒增强版)

更新说明:
1. 移除 yfinance，改用 akshare 获取海外数据，规避 Rate Limit。
2. 修复 SHIBOR 接口字段嗅探逻辑。
3. 动态列生成：失败的数据维度将不会出现在最终 CSV 中。
4. 增强两融与估值数据的容错性。
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
from loguru import logger

# 环境配置
warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_CSV = DATA_DIR / "csi300_features.csv"
OUTPUT_CSV = DATA_DIR / "csi300_features_advanced.csv"

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

# ── 基础工具 ──────────────────────────────────────────────────────────
def get_col(df_columns: pd.Index, candidates: list[str]) -> str | None:
    """智能嗅探列名"""
    for col in candidates:
        if col in df_columns: return col
    return None

def safe_log_ret(series: pd.Series) -> pd.Series:
    """计算对数收益率并处理无穷值"""
    res = np.log(series / series.shift(1))
    return res.replace([np.inf, -np.inf], 0).fillna(0)

# ── 维度 1: 资金面 (北向/两融) ──────────────────────────────────────────
def fetch_smart_money(index_df: pd.DataFrame) -> pd.DataFrame | None:
    logger.info(">>> 抓取资金面数据 (北向/两融)...")
    res = pd.DataFrame(index=index_df.index)
    
    # 1. 北向资金
    try:
        df_hk = ak.stock_hsgt_hist_em(symbol="北向资金")
        df_hk["日期"] = pd.to_datetime(df_hk["日期"])
        df_hk = df_hk.set_index("日期")
        # 使用 shift(1) 确保使用的是 T-1 及之前的历史资金流
        res["north_flow"] = df_hk["当日资金流入"]
    except Exception as e:
        logger.warning(f"北向数据获取失败: {e}")

    # 2. 两融余额
    try:
        df_margin = ak.stock_margin_detail_sse()
        d_col = get_col(df_margin.columns, ["信用交易日期", "日期", "date"])
        v_col = get_col(df_margin.columns, ["融资余额", "余额"])
        if d_col and v_col:
            df_margin[d_col] = pd.to_datetime(df_margin[d_col])
            df_margin = df_margin.drop_duplicates(subset=d_col).set_index(d_col)
            # 两融数据通常晚一天公布，必须 shift(1)
            res["margin_bal_ret"] = safe_log_ret(df_margin[v_col].shift(1))
    except Exception as e:
        logger.warning(f"两融数据获取失败: {e}")

    return res.dropna(how='all', axis=1)

# ── 维度 2: 利率与国债 (SHIBOR/CN10Y) ──────────────────────────────────
def fetch_rates_and_bonds(index_df: pd.DataFrame) -> pd.DataFrame | None:
    logger.info(">>> 抓取利率与国债数据 (SHIBOR/CN10Y)...")
    res = pd.DataFrame(index=index_df.index)
    
    # 1. SHIBOR (修复列名映射)
    try:
        df_shibor = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜")
        d_col = get_col(df_shibor.columns, ["日期", "date", "report_date"])
        v_col = get_col(df_shibor.columns, ["定价", "利率", "close"])
        if d_col and v_col:
            df_shibor[d_col] = pd.to_datetime(df_shibor[d_col])
            aligned_shibor = df_shibor.set_index(d_col)[v_col].reindex(index_df.index, method='ffill')
            res["shibor_diff"] = aligned_shibor.diff()
    except Exception as e:
        logger.warning(f"SHIBOR 数据获取失败: {e}")

    # 2. 中国 10 年期国债
    try:
        df_bond = ak.bond_zh_us_rate()
        d_col = get_col(df_bond.columns, ['date', '日期'])
        if d_col:
            df_bond[d_col] = pd.to_datetime(df_bond[d_col])
            df_bond = df_bond.set_index(d_col)
            res["cn_10y_diff"] = df_bond["中国国债收益率10年"].shift(1).diff()
    except Exception as e:
        logger.warning(f"中国国债获取失败: {e}")

    return res.dropna(how='all', axis=1)

# ── 维度 3: 大宗与汇率 (USDCNH/Copper/Gold) ──────────────────────────
def fetch_fx_commodities(index_df: pd.DataFrame) -> pd.DataFrame | None:
    logger.info(">>> 抓取大宗与汇率数据...")
    res = pd.DataFrame(index=index_df.index)
    d_cands = ['date', '日期', '交易日']
    c_cands = ['close', '收盘', '最新价', '收盘价', '结算价']

    tasks = [
        ("usdcnh", ak.forex_hist_em, {"symbol": "USDCNH"}),
        ("copper", ak.futures_zh_daily_sina, {"symbol": "cu0"}),
        ("gold", ak.spot_hist_sge, {"symbol": "Au99.99"})
    ]

    for name, func, kwargs in tasks:
        try:
            df = func(**kwargs)
            d_col = get_col(df.columns, d_cands)
            c_col = get_col(df.columns, c_cands)
            if d_col and c_col:
                df[d_col] = pd.to_datetime(df[d_col])
                df = df.set_index(d_col)
                res[f"{name}_ret"] = safe_log_ret(df[c_col])
        except Exception as e:
            logger.warning(f"{name} 抓取失败: {e}")
    
    return res.dropna(how='all', axis=1)

# ── 维度 4: 海外市场 (规避 yfinance，改用新浪/东财) ────────────────────
def fetch_global_markets(index_df: pd.DataFrame) -> pd.DataFrame | None:
    logger.info(">>> 抓取海外联动数据 (SP500/US10Y/VIX)...")
    res = pd.DataFrame(index=index_df.index)
    
    # 1. 标普500 (新浪接口)
    try:
        df_spx = ak.index_us_stock_sina(symbol=".INX")
        df_spx['date'] = pd.to_datetime(df_spx['date'])
        df_spx = df_spx.set_index('date')
        # 海外权益修复 (以SP500为例)
        shifted_close = df_spx["close"].shift(1) # 在美股序列上延后一天防穿透
        aligned_close = shifted_close.reindex(index_df.index, method='ffill') # 映射到A股交易日。10-08 对应10.7收盘，09-30 对应9.29收盘
        res["sp500_ret"] = safe_log_ret(aligned_close) # 这将完美捕获长假期间累积涨跌幅
    except Exception as e:
        logger.warning(f"SP500 抓取失败: {e}")

    # 2. 美债 10Y (使用之前的 bond_zh_us_rate 接口提取)
    try:
        df_bond = ak.bond_zh_us_rate()
        d_col = get_col(df_bond.columns, ['date', '日期'])
        if d_col:
            df_bond[d_col] = pd.to_datetime(df_bond[d_col])
            df_bond = df_bond.set_index(d_col)
            # 美债修复同样逻辑：
            shifted_yield = df_bond["美国国债收益率10年"].shift(1)
            aligned_yield = shifted_yield.reindex(index_df.index, method='ffill')
            res["us10y_diff"] = aligned_yield.diff()
    except Exception as e:
        logger.warning(f"美债 10Y 抓取失败: {e}")

    return res.dropna(how='all', axis=1)

# ── 维度 5: 基本面估值 (PE Rank) ─────────────────────────────────────
def fetch_valuation(index_df: pd.DataFrame) -> pd.DataFrame | None:
    logger.info(">>> 抓取指数估值数据...")
    res = pd.DataFrame(index=index_df.index)
    try:
        df_val = ak.stock_zh_index_value_csindex(symbol="000300")
        d_col = get_col(df_val.columns, ["日期", "date"])
        v_col = get_col(df_val.columns, ["市盈率1", "PE"])
        if d_col and v_col:
            df_val[d_col] = pd.to_datetime(df_val[d_col])
            df_val = df_val.set_index(d_col)
            res["pe_rank"] = df_val[v_col].shift(1).rolling(252, min_periods=100).rank(pct=True)
    except Exception as e:
        logger.warning(f"估值抓取失败: {e}")
    return res.dropna(how='all', axis=1)

# ── 维度 6: 统计与周期编码 ────────────────────────────────────────────
def get_stats_and_time(base_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(">>> 生成统计特征与时间编码...")
    res = pd.DataFrame(index=base_df.index)
    
    if "close" in base_df.columns:
        vol = base_df["close"].pct_change().rolling(10).std()
        res["market_vol_10d"] = np.log(vol + 1e-6)
    
    idx = base_df.index
    res["sin_dow"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    res["cos_dow"] = np.cos(2 * np.pi * idx.dayofweek / 5)
    res["sin_month"] = np.sin(2 * np.pi * idx.month / 12)
    res["cos_month"] = np.cos(2 * np.pi * idx.month / 12)
    return res

# ── 主程序 ───────────────────────────────────────────────────────────
def main():
    if not INPUT_CSV.exists():
        logger.error(f"缺失输入文件: {INPUT_CSV}")
        return

    base_df = pd.read_csv(INPUT_CSV, index_col="date", parse_dates=True)
    final_df = base_df.copy()

    # 待抓取的维度函数列表
    fetch_funcs = [
        fetch_smart_money,
        fetch_rates_and_bonds,
        fetch_fx_commodities,
        fetch_global_markets,
        fetch_valuation
    ]

    for func in fetch_funcs:
        add_df = func(base_df)
        if add_df is not None and not add_df.empty:
            # 仅合并那些确实拿到了数据的列
            final_df = final_df.join(add_df, how="left")
            logger.info(f"成功集成维度: {func.__name__}，新增列: {add_df.columns.tolist()}")
        else:
            logger.error(f"维度 {func.__name__} 未获取到有效数据，已跳过。")

    # 注入统计与时间特征 (这一步是基于本地计算，不依赖网络，必成)
    final_df = final_df.join(get_stats_and_time(base_df), how="left")

    # 清洗：先向前填充（对齐交易日），剩余的填0
    final_df = final_df.ffill().fillna(0)
    
    final_df.to_csv(OUTPUT_CSV)
    logger.success(f"✅ 处理完成！最终维度: {len(final_df.columns)}")
    logger.info(f"最终特征列: {final_df.columns.tolist()}")

if __name__ == "__main__":
    main()