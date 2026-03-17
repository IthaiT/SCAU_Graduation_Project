import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append(".")

# 读取final_data.csv查看维度
df = pd.read_csv('data/final_data.csv', index_col='date', parse_dates=True)
print(f"final_data.csv 形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"特征数量 (不包括date): {len(df.columns)}")

# 测试我们的特征工程
from app import process_all_features

# 创建示例数据
dates = pd.date_range('2023-01-01', periods=100, freq='D')
raw_data = pd.DataFrame({
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 105,
    'low': np.random.randn(100).cumsum() + 95,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
}, index=dates)

feature_data = process_all_features(raw_data)
print(f"\n特征工程后形状: {feature_data.shape}")
print(f"特征列名: {feature_data.columns.tolist()}")

# 检查权重期望的维度
import torch
from src.models.networks import LSTMModel

# 创建与训练时相同配置的模型
model = LSTMModel(
    input_dim=22,  # final_data.csv的特征数
    hidden_dim=128,
    num_layers=1,
    dropout=0.21597852403871323,
    pred_len=1
)

print(f"\n模型第一层权重形状 (LSTM weight_ih_l0): {model.lstm.weight_ih_l0.shape}")
print(f"期望输入维度: {model.lstm.weight_ih_l0.shape[1]}")

# 加载权重检查
try:
    state_dict = torch.load('models/LSTM_best.pth', map_location='cpu')
    print(f"\nLSTM权重中 weight_ih_l0 形状: {state_dict['lstm.weight_ih_l0'].shape}")
except Exception as e:
    print(f"加载权重失败: {e}")