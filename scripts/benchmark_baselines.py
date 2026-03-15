"""传统 Baseline 多轮严格评估脚本 (ARIMA 1轮, XGBoost 10轮随机种子)。"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, r2_score
from loguru import logger

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders

# 基础配置 (必须与 DL 保持绝对一致)
SEQ_LEN = 60
PRED_LEN = 1
NUM_RUNS = 10
TARGET_COL = "close"

# ==========================================
# ⚠️ 请将这里替换为你刚才用 Optuna 跑出来的最佳参数！
# ==========================================
XGB_BEST_PARAMS = {
    'n_estimators': 288,
    'max_depth': 3,
    'learning_rate': 0.032953091167420456,
    'subsample': 0.7325183097883851,
    'colsample_bytree': 0.9088333354933051,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'device': 'cuda', # 如果支持GPU就开，不支持就删掉这行
}

ARIMA_BEST_PARAMS = {'p': 0, 'd': 1, 'q': 0}
# ==========================================

def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((true - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100)
    r2 = float(r2_score(true, pred))
    da = float(np.mean(np.sign(np.diff(true)) == np.sign(np.diff(pred))) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "DA": da}

def main():
    logger.info("准备对齐数据...")
    csv_path = PROJECT_ROOT / "data" / "final_data.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()

    train_loader, val_loader, test_loader, scaler_target = get_dataloaders(
        df_values=df.values, columns=columns, seq_len=SEQ_LEN, pred_len=PRED_LEN,
        batch_size=128, target_col=TARGET_COL, train_ratio=0.72, val_ratio=0.10,
    )

    # 抽取与 DL 完全一致的测试集
    N_test = len(test_loader.dataset)
    test_X_xgb = test_loader.dataset.X.numpy().reshape(N_test, -1)
    
    train_X_xgb = train_loader.dataset.X.numpy().reshape(len(train_loader.dataset), -1)
    train_y_xgb = train_loader.dataset.y.numpy().ravel()
    
    val_X_xgb = val_loader.dataset.X.numpy().reshape(len(val_loader.dataset), -1)
    val_y_xgb = val_loader.dataset.y.numpy().ravel()

    # 合并 train 和 val 用于 XGBoost 和 ARIMA 的最终拟合
    X_train_full = np.vstack([train_X_xgb, val_X_xgb])
    y_train_full = np.concatenate([train_y_xgb, val_y_xgb])

    target_idx = columns.index(TARGET_COL)
    full_scaled_target = scaler_target.transform(df.values[:, target_idx:target_idx+1]).ravel()
    
    test_y_norm = test_loader.dataset.y.numpy().ravel()
    true_real = scaler_target.inverse_transform(test_y_norm.reshape(-1, 1)).ravel()

    # ==========================================
    # 1. 评估 XGBoost (跑 10 次，每次改变 random_state)
    # ==========================================
    logger.info("开始评估 XGBoost (10 轮不同随机种子)...")
    xgb_metrics_list =[]
    for i in range(NUM_RUNS):
        # 核心：每次赋予不同的随机种子
        params = XGB_BEST_PARAMS.copy()
        params['random_state'] = i 
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_full, y_train_full, verbose=False)
        
        pred_norm = model.predict(test_X_xgb)
        pred_r = scaler_target.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
        
        xgb_metrics_list.append(compute_metrics(true_real, pred_r))
        logger.info(f"XGBoost 第 {i+1} 轮完成.")

    # ==========================================
    # 2. 评估 ARIMA (数学上完全确定，只跑 1 次！)
    # ==========================================
    logger.info("开始评估 ARIMA (数学唯一解，仅需 1 轮)...")
    p, d, q = ARIMA_BEST_PARAMS['p'], ARIMA_BEST_PARAMS['d'], ARIMA_BEST_PARAMS['q']
    
    # 因为 ARIMA_best 已经跑过，你可以用 apply。此处为严谨直接重新 fit 到 val 结束的数据
    train_val_seq = full_scaled_target[:-N_test]
    model_arima = ARIMA(train_val_seq, order=(p, d, q)).fit()
    
    res = model_arima.apply(full_scaled_target)
    arima_pred_norm = res.fittedvalues[-N_test:]
    arima_pred_r = scaler_target.inverse_transform(arima_pred_norm.reshape(-1, 1)).ravel()
    
    arima_metric = compute_metrics(true_real, arima_pred_r)
    
    # ==========================================
    # 3. 计算均值和方差并输出 Markdown
    # ==========================================
    print("\n" + "="*80)
    print("  Baseline 补充数据 (直接复制拼接到你的 DL 表格中)")
    print("="*80)
    
    # XGBoost 统计
    xgb_stats = {k: (np.mean([m[k] for m in xgb_metrics_list]), np.std([m[k] for m in xgb_metrics_list])) 
                 for k in xgb_metrics_list[0].keys()}
    
    print(f"| {'XGBoost':<17} "
          f"| {xgb_stats['MSE'][0]:>7.2f} ± {xgb_stats['MSE'][1]:<7.2f} "
          f"| {xgb_stats['RMSE'][0]:>5.2f} ± {xgb_stats['RMSE'][1]:<4.2f} "
          f"| {xgb_stats['MAE'][0]:>5.2f} ± {xgb_stats['MAE'][1]:<4.2f} "
          f"| {xgb_stats['R2'][0]:>6.4f} ± {xgb_stats['R2'][1]:<6.4f} "
          f"| {xgb_stats['MAPE'][0]:>4.2f} ± {xgb_stats['MAPE'][1]:<4.2f} |")

    # ARIMA 统计 (标准差固定为 0.00)
    print(f"| {'ARIMA':<17} "
          f"| {arima_metric['MSE']:>7.2f} ± 0.00    "
          f"| {arima_metric['RMSE']:>5.2f} ± 0.00 "
          f"| {arima_metric['MAE']:>5.2f} ± 0.00 "
          f"| {arima_metric['R2']:>6.4f} ± 0.0000 "
          f"| {arima_metric['MAPE']:>4.2f} ± 0.00 |")
    print("="*80)

if __name__ == "__main__":
    main()