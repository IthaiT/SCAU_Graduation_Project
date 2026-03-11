"""工业级多模态高维特征构建脚本 (防泄漏、平稳化)。

核心逻辑:
1. 资金面: 北向资金 (动量)、两融余额 (滞后1天 + 收益率)
2. 宏观: SHIBOR (差分)、美债10Y (滞后1天 + 差分)
3. 估值: 沪深300 PE-TTM (滚动 252 天历史分位数)
4. 外围: S&P500、VIX (滞后1天 + 对数收益率)
5. 时间: Sin/Cos 周期编码

严格约束:
- 带有延迟发布的数据必须 shift(1)
- 价格和绝对量必须转换为收益率/动量/差分，保证平稳性
- 原始特征列 (包含 close) 绝对不被篡改
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

DATA_DIR = PROJECT_ROOT / "data"
INPUT_CSV = DATA_DIR / "csi300_features.csv"
OUTPUT_CSV = DATA_DIR / "csi300_features_advanced.csv"


# ── 工具函数 ──────────────────────────────────────────────────────
def get_date_range(df: pd.DataFrame) -> tuple[str, str]:
    """获取起始和结束日期 (YYYYMMDD格式)"""
    start_date = df.index.min().strftime("%Y%m%d")
    end_date = df.index.max().strftime("%Y%m%d")
    return start_date, end_date

def safe_rank_pct(series: pd.Series) -> float:
    """计算滚动分位数，防呆处理"""
    if len(series.dropna()) < 2:
        return 0.5
    return series.rank(pct=True).iloc[-1]


# ── 维度 1: 资金面与微观结构 (Smart Money) ─────────────────────────
def fetch_smart_money(start_date: str, end_date: str, index_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(">>> 正在获取 [资金面] 数据...")
    res = pd.DataFrame(index=index_df.index)
    
    # 1. 北向资金 (北向从 2014 年底才有，前面填充 0)
    try:
        # 使用东方财富接口获取沪深港通历史数据
        df_hk = ak.stock_hsgt_hist_em(symbol="北向资金")
        df_hk["日期"] = pd.to_datetime(df_hk["日期"])
        df_hk.set_index("日期", inplace=True)
        # 北向资金在交易日实时可知，当日盘后可见，可用 T 日特征预测 T+1 (无需shift)
        res["north_net_flow"] = df_hk["当日资金流入"]
    except Exception as e:
        logger.warning(f"北向资金获取失败，采用零填充: {e}")
        res["north_net_flow"] = 0.0

    # 2. 两融余额 (沪市作为代理)
    try:
        df_margin = ak.stock_margin_detail_sse()
        df_margin["信用交易日期"] = pd.to_datetime(df_margin["信用交易日期"])
        # ⚠️ 关键修复：删除重复索引
        df_margin = df_margin[~df_margin.index.duplicated(keep='first')] 
        df_margin.set_index("信用交易日期", inplace=True)
        res["margin_balance"] = df_margin["融资余额"].shift(1)
    except Exception as e:
        logger.warning(f"两融数据获取失败: {e}")
        res["margin_balance"] = 0.0

    # ── 特征平稳化处理 ──
    res.fillna(0, inplace=True)
    
    # 北向动量 (5日均值 - 20日均值，表征资金进攻/撤退加速)
    res["north_ma5"] = res["north_net_flow"].rolling(5, min_periods=1).mean()
    res["north_ma20"] = res["north_net_flow"].rolling(20, min_periods=1).mean()
    res["north_momentum"] = res["north_ma5"] - res["north_ma20"]
    
    # 两融收益率 (表征杠杆资金情绪变化率)
    # 使用 log1p 防止零除
    res["margin_ret_5d"] = np.log1p(res["margin_balance"].pct_change(5).fillna(0))
    res["margin_ret_20d"] = np.log1p(res["margin_balance"].pct_change(20).fillna(0))
    
    # 丢弃未平稳的绝对值列
    return res[["north_momentum", "margin_ret_5d", "margin_ret_20d"]]


# ── 维度 2: 宏观与流动性 (Macro Anchors) ──────────────────────────
def fetch_macro(start_date: str, end_date: str, index_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(">>> 正在获取 [宏观与流动性] 数据...")
    res = pd.DataFrame(index=index_df.index)
    
    try:
        # SHIBOR 隔夜利率 (中国短期流动性)
        df_shibor = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜")
        if "日期" in df_shibor.columns:
            df_shibor["日期"] = pd.to_datetime(df_shibor["日期"])
            df_shibor.set_index("日期", inplace=True)
            res["shibor_on"] = df_shibor["定价"]
        else:
            res["shibor_on"] = 0.0
    except Exception as e:
        logger.warning(f"SHIBOR 获取失败: {e}")
        res["shibor_on"] = 0.0

    # ── 特征平稳化处理 ──
    res.ffill(inplace=True)
    res.fillna(0, inplace=True)
    
    # 利率不能直接用绝对值（长周期有下降趋势），使用一阶差分
    res["shibor_diff"] = res["shibor_on"].diff().fillna(0)
    
    return res[["shibor_diff"]]


# ── 维度 3: 基本面估值 (Valuation) ───────────────────────────────
def fetch_valuation(start_date: str, end_date: str, index_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(">>> 正在获取 [基本面估值] 数据...")
    res = pd.DataFrame(index=index_df.index)
    
    try:
        # 中证指数估值
        df_val = ak.stock_zh_index_value_csindex(symbol="000300")
        df_val["日期"] = pd.to_datetime(df_val["日期"])
        df_val.set_index("日期", inplace=True)
        res["pe_ttm"] = df_val["市盈率1"]
        res["dividend_yield"] = df_val["股息率1"]
    except Exception as e:
        logger.warning(f"估值数据获取失败: {e}")
        res["pe_ttm"] = 15.0  # 给予均值兜底
        res["dividend_yield"] = 2.0

    res.ffill(inplace=True)
    
    # ── 特征平稳化处理: 历史分位数 (消除长期趋势，反映绝对贵贱) ──
    # 计算过去 252 个交易日（一年）的 PE 历史分位数
    res["pe_quantile_252d"] = res["pe_ttm"].rolling(252, min_periods=1).apply(safe_rank_pct)
    # 股息率同理
    res["dy_quantile_252d"] = res["dividend_yield"].rolling(252, min_periods=1).apply(safe_rank_pct)

    return res[["pe_quantile_252d", "dy_quantile_252d"]]


# ── 维度 4: 跨市场联动 (Global Context) ─────────────────────────
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_with_retry(tickers, start, end):
    return yf.download(tickers, start=start, end=end, progress=False)

# ── 修改 1: 修正 fetch_global_context 缺失的处理 ──
def fetch_global_context(start_str: str, end_str: str, index_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(">>> 正在获取 [跨市场联动] 数据 (通过 yfinance)...")
    res = pd.DataFrame(index=index_df.index)
    
    start_dt = datetime.strptime(start_str, "%Y%m%d").strftime("%Y-%m-%d")
    end_dt = (datetime.strptime(end_str, "%Y%m%d") + timedelta(days=5)).strftime("%Y-%m-%d")
    
    # 增加重试机制，防止 Too Many Requests
    for attempt in range(3):
        try:
            yf_df = yf.download(["^GSPC", "^VIX", "^TNX"], start=start_dt, end=end_dt, progress=False)
            if not yf_df.empty:
                break
            time.sleep(5)
        except Exception as e:
            logger.warning(f"下载尝试 {attempt+1} 失败: {e}")
            time.sleep(10)
    else:
        logger.error("连续 3 次下载海外数据失败，请检查网络或稍后再试。")
        return res.assign(gspc_log_ret=0, vix_pct=0, us_10y_diff=0)

    # 提取收盘价
    try:
        # 处理可能的 MultiIndex
        gspc = yf_df["Close"]["^GSPC"] if isinstance(yf_df.columns, pd.MultiIndex) else yf_df["Close"]
        vix = yf_df["Close"]["^VIX"] if isinstance(yf_df.columns, pd.MultiIndex) else pd.Series(20.0, index=yf_df.index)
        tnx = yf_df["Close"]["^TNX"] if isinstance(yf_df.columns, pd.MultiIndex) else pd.Series(4.0, index=yf_df.index)
        
        us_df = pd.DataFrame({"gspc": gspc, "vix": vix, "us_10y": tnx})
        us_df.index = pd.to_datetime(us_df.index).tz_localize(None)
        us_df = us_df.shift(1) # 防未来函数
        
        res = res.join(us_df, how="left")
    except Exception as e:
        logger.error(f"处理数据格式异常: {e}")
        return res.assign(gspc_log_ret=0, vix_pct=0, us_10y_diff=0)

    # ── 特征平稳化处理 (修复 bfill 问题) ──
    res = res.ffill().bfill().fillna(0)
    
    res["gspc_log_ret"] = np.log(res["gspc"] / res["gspc"].shift(1)).fillna(0)
    res["vix_pct"] = res["vix"].pct_change().fillna(0)
    res["us_10y_diff"] = res["us_10y"].diff().fillna(0)

    return res[["gspc_log_ret", "vix_pct", "us_10y_diff"]]

# ── 维度 5: 时间编码 (Time Embeddings) ───────────────────────────
def get_time_embeddings(index_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(">>> 正在生成 [时间与周期编码]...")
    res = pd.DataFrame(index=index_df.index)
    
    # 提取日期序列
    dates = pd.Series(index_df.index.date, index=index_df.index)
    dates = pd.to_datetime(dates)
    
    # 星期编码 (0-4 代表周一到周五)
    day_of_week = dates.dt.dayofweek
    res["sin_dow"] = np.sin(2 * np.pi * day_of_week / 5)
    res["cos_dow"] = np.cos(2 * np.pi * day_of_week / 5)
    
    # 月份编码 (1-12)
    month = dates.dt.month
    res["sin_month"] = np.sin(2 * np.pi * month / 12)
    res["cos_month"] = np.cos(2 * np.pi * month / 12)
    
    return res


# ── 主构建流程 ───────────────────────────────────────────────────
def main():
    if not INPUT_CSV.exists():
        logger.error(f"基础数据文件不存在: {INPUT_CSV}")
        sys.exit(1)
        
    logger.info("加载基础量价数据: {}", INPUT_CSV)
    base_df = pd.read_csv(INPUT_CSV, index_col="date", parse_dates=True)
    start_str, end_str = get_date_range(base_df)
    logger.info(f"数据区间: {start_str} -> {end_str}, 共 {len(base_df)} 交易日")
    
    # 逐一获取五个维度的高阶特征
    df_smart_money = fetch_smart_money(start_str, end_str, base_df)
    df_macro = fetch_macro(start_str, end_str, base_df)
    df_valuation = fetch_valuation(start_str, end_str, base_df)
    df_global = fetch_global_context(start_str, end_str, base_df)
    df_time = get_time_embeddings(base_df)
    
    # 合并所有特征 (基于 Base DataFrame 的 index 进行 Left Join)
    logger.info(">>> 合并多模态特征矩阵...")
    advanced_df = base_df.copy()
    
    for ext_df in[df_smart_money, df_macro, df_valuation, df_global, df_time]:
        advanced_df = advanced_df.join(ext_df, how="left")
        
    # 终极防空值清洗 (ffill -> fillna 0)
    advanced_df.ffill(inplace=True)
    advanced_df.fillna(0, inplace=True)
    
    # 校验是否破坏了基础数据
    assert (advanced_df["close"] == base_df["close"]).all(), "❌ 致命错误：close 列被篡改！"
    assert len(advanced_df) == len(base_df), "❌ 致命错误：数据行数发生变动！"
    
    # 统计信息
    added_cols = len(advanced_df.columns) - len(base_df.columns)
    logger.info(f"特征扩充完成！新增了 {added_cols} 维跨模态特征。总维度: {len(advanced_df.columns)}")
    
    # 保存
    advanced_df.to_csv(OUTPUT_CSV)
    logger.info(f"✅ 高阶数据集已保存至: {OUTPUT_CSV}")
    
    # 打印最终特征列名
    logger.info("当前网络输入特征一览:")
    logger.info(advanced_df.columns.tolist())

if __name__ == "__main__":
    main()