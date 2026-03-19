"""
LSTM-Transformer 混合深度学习架构金融时序预测可视化系统
毕业设计 - 金融时间序列量价预测对比平台
Author: SCAU 计算机本科毕业设计
Date: 2026-03
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import akshare as ak
import os
import torch
import torch.nn as nn
from contextlib import contextmanager
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 导入自定义模型定义
import sys
sys.path.append(".")
from src.models.networks import LSTMModel, TransformerModel, LSTMTransformerModel
from src.data.fetch import fetch_and_clean_data

# ============================================================================
# 超参数配置字典（必须与训练时一致，从scripts/train.py复制）
# ============================================================================
BEST_CONFIGS = {
    "LSTM": {
        "model_args": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.21597852403871323},
    },
    "Transformer": {
        "model_args": {"d_model": 128, "num_heads": 2, "num_layers": 2, "ffn_dim": 512, "dropout": 0.2646631095765896},
    },
    "LSTM_Transformer": {
        "model_args": {"hidden_dim": 128, "num_lstm_layers": 1, "num_transformer_layers": 2, "num_heads": 4, "ffn_dim": 256, "dropout": 0.40810640767923295},
    },
}

# ============================================================================
# 固化特征列顺序（与训练时一致）
# ============================================================================
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume','amount',
    'sma_5', 'sma_20', 'rsi_14',
    'macd_12_26_9', 'macdh_12_26_9', 'macds_12_26_9',
    'bbl_5_2.0_2.0', 'bbm_5_2.0_2.0', 'bbu_5_2.0_2.0', 'bbb_5_2.0_2.0', 'bbp_5_2.0_2.0',
    'north_momentum', 
    'sin_dow', 'cos_dow', 'sin_month', 'cos_month'
]

SEQ_LEN = 60
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10
PROJECT_ROOT = Path(__file__).resolve().parent

@contextmanager
def _no_proxy():
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    backup = {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}
    for k in keys:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in backup.items():
            os.environ[k] = v


# ============================================================================
# 数据获取与预处理模块
# ============================================================================
def fetch_stock_data(symbol: str, years: int = 3) -> pd.DataFrame:
    """
    使用 akshare 获取指定标的的日线历史数据
    
    Args:
        symbol: 标的代码，如 "sh000300" (沪深300), "sh000016" (上证50)
        years: 获取最近多少年的数据，默认3年
    
    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume, amount
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365*years)).strftime("%Y%m%d")
    
    try:
        with _no_proxy():
            df = fetch_and_clean_data(symbol=symbol, start_date=start_date, end_date=end_date)
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        result = df[required_cols].astype(float).ffill().bfill()
        if result.isnull().any().any():
            raise ValueError("数据中存在无法处理的缺失值")
        st.success(f"✅ 数据获取成功: {symbol} ({len(result)} 行)")
        return result
    except Exception as e:
        st.error(f"❌ 数据获取失败: {e}")
        raise

# ============================================================================
# 特征工程模块（集成用户提供的代码）
# ============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """技术指标特征扩充"""
    df = df.copy()
    # 使用 pandas_ta 添加技术指标
    import pandas_ta as ta
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.columns = [c.lower() for c in df.columns]
    return df.dropna()

def get_time_embeddings(index_df: pd.DataFrame) -> pd.DataFrame:
    """时间周期性编码"""
    res = pd.DataFrame(index=index_df.index)
    dates = pd.to_datetime(pd.Series(index_df.index.date, index=index_df.index))
    day_of_week = dates.dt.dayofweek
    res["sin_dow"] = np.sin(2 * np.pi * day_of_week / 5)
    res["cos_dow"] = np.cos(2 * np.pi * day_of_week / 5)
    month = dates.dt.month
    res["sin_month"] = np.sin(2 * np.pi * month / 12)
    res["cos_month"] = np.cos(2 * np.pi * month / 12)
    return res

def fetch_smart_money(start_date: str, end_date: str, index_df: pd.DataFrame) -> pd.DataFrame:
    """北向资金流向特征"""
    res = pd.DataFrame(index=index_df.index)
    try:
        df_hk = ak.stock_hsgt_hist_em(symbol="北向资金")
        df_hk["日期"] = pd.to_datetime(df_hk["日期"])
        df_hk.set_index("日期", inplace=True)
        res["north_net_flow"] = df_hk["当日资金流入"]
    except Exception:
        res["north_net_flow"] = 0.0
    res.fillna(0, inplace=True)
    res["north_ma5"] = res["north_net_flow"].rolling(5, min_periods=1).mean()
    res["north_ma20"] = res["north_net_flow"].rolling(20, min_periods=1).mean()
    res["north_momentum"] = res["north_ma5"] - res["north_ma20"]
    return res

def process_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """全量特征处理流水线"""
    df = build_features(df)
    time_feats = get_time_embeddings(df)
    smart_money = fetch_smart_money(None, None, df)
    final_df = pd.concat([df, time_feats, smart_money], axis=1).dropna()
    
    # 确保特征列顺序与定义的FEATURE_COLS一致
    missing_cols = [col for col in FEATURE_COLS if col not in final_df.columns]
    
    if missing_cols:
        raise ValueError(f"特征工程结果缺失关键列: {missing_cols}")
    
    # 按顺序重排列
    final_df = final_df[FEATURE_COLS]
    return final_df

def load_aligned_feature_data() -> pd.DataFrame:
    data_path = PROJECT_ROOT / "data" / "final_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"未找到训练对齐数据集: {data_path}")
    df = pd.read_csv(data_path, index_col="date", parse_dates=True)
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"final_data.csv 缺失特征列: {missing_cols}")
    return df[FEATURE_COLS].copy()

# ============================================================================
# 模型加载与推理模块
# ============================================================================
def create_model(model_name: str):
    """根据配置创建模型实例，输入维度动态设置为FEATURE_COLS长度"""
    config = BEST_CONFIGS[model_name]["model_args"]
    
    # 使用固化特征列的长度作为输入维度
    INPUT_DIM = len(FEATURE_COLS)  # 应为22
    
    if model_name == "LSTM":
        model = LSTMModel(
            input_dim=INPUT_DIM,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            pred_len=1
        )
    elif model_name == "Transformer":
        model = TransformerModel(
            input_dim=INPUT_DIM,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            ffn_dim=config["ffn_dim"],
            dropout=config["dropout"],
            pred_len=1
        )
    elif model_name == "LSTM_Transformer":
        model = LSTMTransformerModel(
            input_dim=INPUT_DIM,
            hidden_dim=config["hidden_dim"],
            num_lstm_layers=config["num_lstm_layers"],
            num_heads=config["num_heads"],
            num_transformer_layers=config["num_transformer_layers"],
            ffn_dim=config["ffn_dim"],
            dropout=config["dropout"],
            pred_len=1,
            seq_len=60  # 需要传递seq_len参数
        )
    else:
        raise ValueError(f"未知模型: {model_name}")
    
    return model

def load_model_weights(model, model_name: str):
    """加载预训练权重"""
    weight_path = f"./models/{model_name}_best.pth"
    try:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        st.sidebar.info(f"✅ {model_name} 权重加载成功")
        return True
    except FileNotFoundError:
        st.sidebar.warning(f"⚠️ {model_name} 预训练权重未找到，使用随机初始化")
        return False
    except Exception as e:
        st.sidebar.error(f"❌ {model_name} 权重加载失败: {e}")
        return False

def prepare_scalers(full_df: pd.DataFrame):
    """
    创建特征和目标的MinMaxScaler
    - scaler_all: 对所有特征（包含close）进行归一化，用于模型输入
    - scaler_target: 仅对close列归一化，用于结果反归一化
    """
    split_idx = int(len(full_df) * TRAIN_RATIO)
    train_data = full_df.iloc[:split_idx]
    
    # 对所有特征（包括close）进行归一化
    scaler_all = MinMaxScaler()
    scaler_all.fit(train_data)  # 包含close在内的所有列
    
    # 单独对目标列（close）进行归一化，用于反归一化
    scaler_target = MinMaxScaler()
    scaler_target.fit(train_data[['close']])
    
    return scaler_all, scaler_target

def create_inference_context(full_df: pd.DataFrame):
    n = len(full_df)
    t1 = int(n * TRAIN_RATIO)
    t2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_df = full_df.iloc[:t1].copy()
    val_df = full_df.iloc[t1:t2].copy()
    test_df = full_df.iloc[t2:].copy()
    scaler_x = MinMaxScaler()
    scaler_x.fit(train_df[FEATURE_COLS])
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_df[['close']])
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "all": full_df.copy(),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y
    }

def sliding_window_inference(model, scaler_x, scaler_y, test_data, seq_len=60):
    """滑动窗口推理，使用固化的特征列顺序"""
    test_data = test_data.copy()
    if len(test_data) <= seq_len:
        return []
    # 确保test_data包含所有需要的特征列
    missing_cols = [col for col in FEATURE_COLS if col not in test_data.columns]
    if missing_cols:
        raise ValueError(f"推理数据缺失特征列: {missing_cols}")
    
    # 使用固化的特征列顺序
    feature_cols = FEATURE_COLS
    predictions = []
    with torch.no_grad():
        for i in range(len(test_data) - seq_len):
            # 获取窗口数据
            window = test_data.iloc[i:i+seq_len]
            
            # 缩放特征（scaler_x已经拟合了所有列，包含close）
            X_scaled = scaler_x.transform(window[feature_cols])
            
            # 转换为张量
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)  # (1, seq_len, input_dim)
            
            # 推理
            pred_scaled, _ = model(X_tensor)
            pred = scaler_y.inverse_transform(pred_scaled.numpy().reshape(-1, 1))
            
            predictions.append(pred[0, 0])
    
    return predictions

# ============================================================================
# 评估指标计算
# ============================================================================
def calculate_metrics(y_true, y_pred):
    """计算多种评估指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    safe_denominator = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / safe_denominator)) * 100
    
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "MSE": round(mse, 4)
    }

# ============================================================================
# 可视化工具函数
# ============================================================================
def plot_candlestick_with_indicators(df: pd.DataFrame, title: str = "K线图与技术指标"):
    """绘制专业的K线图带技术指标"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("价格与移动平均线", "成交量", "RSI")
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # 移动平均线
    if 'sma_5' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_5'], name="MA5", line=dict(color='orange', width=1)),
            row=1, col=1
        )
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name="MA20", line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # 布林带
    if 'bbm_5_2.0_2.0' in df.columns and 'bbu_5_2.0_2.0' in df.columns and 'bbl_5_2.0_2.0' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bbu_5_2.0_2.0'], name="BB Upper", line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bbm_5_2.0_2.0'], name="BB Middle", line=dict(color='gray', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bbl_5_2.0_2.0'], name="BB Lower", line=dict(color='gray', width=1, dash='dash'), fill='tonexty'),
            row=1, col=1
        )
    
    # 成交量
    colors = ['red' if df['close'].iloc[i] > df['open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name="成交量", marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'rsi_14' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi_14'], name="RSI", line=dict(color='purple', width=1)),
            row=3, col=1
        )
        # 添加超买超卖线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="价格",
        template="plotly_dark",
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def plot_predictions_comparison(df_true, predictions_dict, lookback_days):
    """绘制真实价格与多模型预测对比图"""
    fig = go.Figure()
    
    # 真实价格曲线 - 使用最后lookback_days天的数据
    true_prices = df_true['close'].iloc[-lookback_days:].values
    dates = df_true.index[-lookback_days:].tolist()
    
    fig.add_trace(go.Scatter(
        x=dates, y=true_prices,
        mode='lines',
        name='真实收盘价',
        line=dict(color='white', width=3)
    ))
    
    # 各模型预测曲线
    colors = {'LSTM': '#FF6B6B', 'Transformer': '#FFA726', 'LSTM_Transformer': '#45B7D1'}
    for model_name, preds in predictions_dict.items():
        if len(preds) > 0:
            # 预测点数量应等于lookback_days，与真实日期完全对齐
            if len(preds) == len(dates):
                pred_dates = dates
            else:
                # 如果预测点数量不同，使用最后len(preds)个日期
                pred_dates = df_true.index[-len(preds):].tolist()
            fig.add_trace(go.Scatter(
                x=pred_dates, y=preds,
                mode='lines',
                name=f'{model_name}预测',
                line=dict(color=colors.get(model_name, '#95A5A6'), width=2, dash='dash')
            ))
    
    fig.update_layout(
        title=f"模型性能对比 (最近{lookback_days}天)",
        xaxis_title="日期",
        yaxis_title="价格",
        template="plotly_dark",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

# ============================================================================
# Streamlit 应用主界面
# ============================================================================
def main():
    # 页面配置
    st.set_page_config(
        page_title="金融时序预测系统",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 标题与描述
    st.title("LSTM-Transformer 金融时序预测对比系统")
    st.markdown("**毕业设计项目** - 展示混合架构在金融时间序列预测中的性能优势")
    st.markdown("---")
    
    # 侧边栏配置
    st.sidebar.header("系统配置")
    
    # 模块一：数据选择
    st.sidebar.subheader("数据配置")
    symbol_options = {
        "沪深300 (sh000300)": "sh000300",
        "上证50 (sh000016)": "sh000016", 
        "上证指数 (sh000001)": "sh000001",
        "创业板指 (sz399006)": "sz399006",
        "中证500 (sh000905)": "sh000905"
    }
    selected_symbol_name = st.sidebar.selectbox(
        "选择预测标的",
        list(symbol_options.keys()),
        index=0
    )
    symbol = symbol_options[selected_symbol_name]
    
    data_scope_options = {
        "test（默认）": "test",
        "val": "val",
        "train": "train",
        "all": "all"
    }
    selected_scope_name = st.sidebar.selectbox(
        "推理数据范围",
        list(data_scope_options.keys()),
        index=0,
        help="默认使用test集，确保与训练评估口径一致"
    )
    eval_scope = data_scope_options[selected_scope_name]

    lookback_days = st.sidebar.slider(
        "回测天数", 
        min_value=30, 
        max_value=365, 
        value=100,
        help="选择最近多少天的数据进行模型对比"
    )
    
    # 模块二：模型选择
    st.sidebar.subheader("模型选择")
    available_models = ["LSTM", "Transformer", "LSTM_Transformer"]
    selected_models = []
    for model in available_models:
        if st.sidebar.checkbox(model, value=(model == "LSTM_Transformer")):
            selected_models.append(model)
    
    if not selected_models:
        st.sidebar.warning("请至少选择一个模型进行对比")
        selected_models = ["LSTM_Transformer"]
    
    # 初始化会话状态
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'feature_data' not in st.session_state:
        st.session_state.feature_data = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = {}
    
    # 数据获取按钮
    st.sidebar.subheader("数据操作")
    if st.sidebar.button("加载训练对齐数据", type="primary"):
        with st.spinner("正在加载 final_data.csv ..."):
            feature_data = load_aligned_feature_data()
            st.session_state.raw_data = feature_data[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            st.session_state.feature_data = feature_data
            st.sidebar.info("训练对齐数据加载完成")

    if st.sidebar.button("获取最新数据并构建特征"):
        try:
            with st.spinner("正在获取历史数据..."):
                raw_data = fetch_stock_data(symbol)
                st.session_state.raw_data = raw_data
            with st.spinner("正在进行特征工程..."):
                feature_data = process_all_features(raw_data)
                st.session_state.feature_data = feature_data
        except Exception:
            st.sidebar.error("在线数据获取失败，可能是代理/网络限制；建议使用“加载训练对齐数据”")
            st.stop()
    
    # 主界面
    if st.session_state.raw_data is not None:
        raw_data = st.session_state.raw_data
        feature_data = st.session_state.feature_data
        
        # ====================================================================
        # 模块一：数据与特征展示
        # ====================================================================
        st.header("模块一：数据与特征展示")
        
        with st.expander("点击查看特征工程流水线与原始数据", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("专业K线图与技术指标")
                # 显示最近200天的K线图
                plot_data = feature_data.tail(200)
                fig = plot_candlestick_with_indicators(plot_data, f"{selected_symbol_name} K线图")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("特征数据预览")
                st.dataframe(feature_data.tail(10), use_container_width=True)
                
                st.metric("总数据量", f"{len(feature_data)} 行")
                st.metric("特征维度", f"{feature_data.shape[1]} 维")
                st.metric("时间范围", f"{feature_data.index[0].date()} 至 {feature_data.index[-1].date()}")
                
                # 特征相关性热图
                if st.checkbox("显示特征相关性"):
                    corr_matrix = feature_data.corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="特征相关性热图",
                        color_continuous_scale="RdBu"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        # ====================================================================
        # 模块二：模型对比
        # ====================================================================
        st.header("模块二：模型对比")
        
        if len(selected_models) > 0:
            context = create_inference_context(feature_data)
            eval_df = context[eval_scope]
            max_valid_days = max(0, len(eval_df) - SEQ_LEN)
            effective_days = min(lookback_days, max_valid_days)
            if effective_days <= 0:
                st.warning(f"{selected_scope_name} 数据不足，至少需要 {SEQ_LEN + 1} 行")
            else:
                test_data = eval_df.iloc[-(effective_days + SEQ_LEN):]
            
                predictions_dict = {}
                metrics_dict = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, model_name in enumerate(selected_models):
                    status_text.text(f"正在加载 {model_name} 模型...")
                    model = create_model(model_name)
                    weights_loaded = load_model_weights(model, model_name)
                    if not weights_loaded:
                        st.warning(f"{model_name} 跳过推理：未找到可用权重")
                        progress_bar.progress((idx + 1) / len(selected_models))
                        continue
                    model.eval()
                    
                    status_text.text(f"正在进行 {model_name} 推理...")
                    predictions = sliding_window_inference(
                        model, context["scaler_x"], context["scaler_y"], test_data, seq_len=SEQ_LEN
                    )
                    predictions_dict[model_name] = predictions
                    
                    if len(predictions) > 0:
                        true_values = test_data['close'].iloc[SEQ_LEN:].values
                        metrics = calculate_metrics(true_values, predictions)
                        metrics_dict[model_name] = metrics
                    
                    progress_bar.progress((idx + 1) / len(selected_models))
                
                status_text.text("推理完成！")
                progress_bar.empty()
                
                st.subheader("预测结果对比")
                fig_comparison = plot_predictions_comparison(
                    eval_df, predictions_dict, effective_days
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                st.subheader("模型性能评估")
                if metrics_dict:
                    metrics_df = pd.DataFrame(metrics_dict).T
                    st.dataframe(metrics_df.style.highlight_min(axis=0, color='#FF6B6B'), use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    for model_name, metrics in metrics_dict.items():
                        with col1 if model_name == "LSTM" else col2 if model_name == "Transformer" else col3:
                            st.metric(
                                label=f"{model_name} RMSE",
                                value=metrics["RMSE"],
                                delta=f"MAPE: {metrics['MAPE']}%"
                            )
                else:
                    st.warning("没有可用模型结果，请检查权重文件是否存在")
        
        # ====================================================================
        # 模块三：单步预测
        # ====================================================================
        st.header("模块三：单步预测")
        
        prediction_container = st.container()
        with prediction_container:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("明日价格预测")
                if st.button("预测下一个交易日收盘价", type="primary", use_container_width=True):
                    # 模拟日志流
                    log_placeholder = st.empty()
                    logs = [
                        "[INFO] 拉取最新数据...",
                        "[INFO] 数据预处理完成",
                        "[INFO] 构建特征张量...",
                        "[INFO] 加载预训练权重...",
                        "[INFO] 模型推理中...",
                        "[INFO] 推理完成！"
                    ]
                    
                    for i, log in enumerate(logs):
                        log_placeholder.text(log)
                        time.sleep(0.5)
                    
                    # 获取最新数据
                    context = create_inference_context(feature_data)
                    latest_data = feature_data.tail(SEQ_LEN)
                    
                    # 使用LSTM_Transformer模型进行预测
                    if "LSTM_Transformer" in selected_models:
                        model_name = "LSTM_Transformer"
                        model = create_model(model_name)
                        weights_loaded = load_model_weights(model, model_name)
                        if not weights_loaded:
                            st.error("LSTM_Transformer 权重不存在，无法执行单步预测")
                            st.stop()
                        model.eval()

                        X_scaled = context["scaler_x"].transform(latest_data[FEATURE_COLS])
                        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
                        
                        # 推理
                        with torch.no_grad():
                            pred_scaled, _ = model(X_tensor)
                            pred_price = context["scaler_y"].inverse_transform(
                                pred_scaled.numpy().reshape(-1, 1)
                            )[0, 0]
                        
                        # 计算涨跌幅
                        current_price = latest_data['close'].iloc[-1]
                        change = ((pred_price - current_price) / current_price) * 100
                        
                        # 显示结果
                        st.success("预测完成！")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "今日收盘价",
                                f"{current_price:.2f}",
                                delta="基准"
                            )
                        with col_b:
                            st.metric(
                                "明日预测价",
                                f"{pred_price:.2f}",
                                delta=f"{change:.2f}%",
                                delta_color="inverse" if change < 0 else "normal"
                            )
                        
                        # 趋势判断
                        if change > 1:
                            st.success(f"强烈看涨信号 (+{change:.2f}%)")
                        elif change > 0:
                            st.info(f"小幅看涨 (+{change:.2f}%)")
                        elif change > -1:
                            st.warning(f"震荡走势 ({change:.2f}%)")
                        else:
                            st.error(f"看跌信号 ({change:.2f}%)")
                    else:
                        st.warning("请先选择LSTM_Transformer模型")
            
            with col2:
                st.subheader("模型架构说明")
                st.markdown("""
                **LSTM-Transformer 混合架构优势**

                1. **时序特征提取**：LSTM层捕获长期依赖关系
                2. **全局注意力机制**：Transformer层学习序列全局依赖
                3. **渐进式学习**：ReZero残差连接确保稳定训练
                4. **多尺度特征**：结合局部与全局信息

                **技术参数**
                - 输入维度：22个技术指标 + 时间特征 + 资金流向
                - 序列长度：60个交易日
                - 输出：单步预测（明日收盘价）
                - 损失函数：MSE + 自定义正则化
                """)
    
    else:
        # 初始状态：显示欢迎信息和数据获取提示
        st.info("""
        ## 🎯 使用说明
        
        1. **侧边栏配置**：选择预测标的、推理范围和回测天数
        2. **选择对比模型**：勾选要参与对比的模型
        3. **优先加载训练对齐数据**：点击"加载训练对齐数据"
        4. **可选在线数据**：点击"获取最新数据并构建特征"（受网络/代理影响）
        5. **查看特征**：在模块一查看K线图和特征数据
        6. **模型对比**：在模块二查看各模型预测结果和性能指标
        7. **明日预测**：在模块三进行单步实时预测
        
        ### 支持的模型
        - **LSTM**：传统循环神经网络基线
        - **Transformer**：纯注意力机制基线  
        - **LSTM_Transformer**：提出的混合架构（主力模型）
        
        ### 特征维度
        系统自动生成22维特征，包括：
        - 原始OHLCV价格数据
        - 技术指标（MA, RSI, MACD, 布林带）
        - 时间周期性编码
        - 北向资金流向特征
        """)
        
        if st.button("加载训练对齐数据", type="primary"):
            with st.spinner("正在加载 final_data.csv ..."):
                feature_data = load_aligned_feature_data()
                st.session_state.raw_data = feature_data[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
                st.session_state.feature_data = feature_data
                st.rerun()

# ============================================================================
# 应用入口
# ============================================================================
if __name__ == "__main__":
    main()
