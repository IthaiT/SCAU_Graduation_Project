"""传统机器学习与计量经济学 Baseline: 严格对齐 DL 窗口的搜索与训练引擎。"""
import sys
import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from loguru import logger

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 引入你严谨的 DataLoader 工厂
from src.data.dataset import get_dataloaders

SEQ_LEN = 60
PRED_LEN = 1
TARGET_COL = "close"
N_TRIALS = 150
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_strictly_aligned_data():
    """从你的 dataset.py 中提取严格对齐的数据，保证所有 Baseline 与 DL 模型 100% 同步。"""
    csv_path = PROJECT_ROOT / "data" / "final_data.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()

    train_loader, val_loader, test_loader, scaler_target = get_dataloaders(
        df_values=df.values, columns=columns, seq_len=SEQ_LEN, pred_len=PRED_LEN,
        batch_size=128, target_col=TARGET_COL, train_ratio=0.72, val_ratio=0.10,
    )

    # ==========================================
    # 1. 为 XGBoost 准备严格对齐的展平数据
    # dataset.X 形状为 (N, seq_len, num_features) -> 展平为 (N, seq_len * num_features)
    # dataset.y 形状为 (N, 1) -> 展平为 (N,)
    # ==========================================
    xgb_X_train = train_loader.dataset.X.numpy().reshape(len(train_loader.dataset), -1)
    xgb_y_train = train_loader.dataset.y.numpy().ravel()
    
    xgb_X_val = val_loader.dataset.X.numpy().reshape(len(val_loader.dataset), -1)
    xgb_y_val = val_loader.dataset.y.numpy().ravel()

    # ==========================================
    # 2. 为 ARIMA 准备连续的 1D 时序序列
    # ==========================================
    target_idx = columns.index(TARGET_COL)
    # 使用你 dataset.py 里的同一把 Scaler 进行归一化
    target_scaled = scaler_target.transform(df.values[:, target_idx:target_idx+1]).ravel()
    
    n = len(df.values)
    t1 = int(n * 0.72)
    t2 = int(n * (0.72 + 0.10))
    
    arima_train_seq = target_scaled[:t1]
    arima_train_val_seq = target_scaled[:t2] # 用于评估验证集时做 history
    
    return (xgb_X_train, xgb_y_train, xgb_X_val, xgb_y_val), \
           (arima_train_seq, arima_train_val_seq, xgb_y_val)


def optimize_xgb(xgb_data):
    X_train, y_train, X_val, y_val = xgb_data
    logger.info("="*60 + "\n🚀 开始 XGBoost 150次超参搜索 (严格对齐滑动窗口)...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'reg:squarederror',
            'tree_method': 'hist',   # 新增：告诉 XGBoost 使用直方图算法 (GPU支持的前提)
            'device': 'cuda',        # 新增：强制使用 GPU (旧版本写的是 'tree_method': 'gpu_hist')
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction='minimize', study_name="XGBoost_Baseline")
    study.optimize(objective, n_trials=N_TRIALS)
    
    logger.success(f"✅ XGBoost 最佳验证集 MSE: {study.best_value:.6f}")
    
    # 用最佳参数训练最终模型并保存
    best_model = xgb.XGBRegressor(**study.best_params, objective='reg:squarederror', n_jobs=-1, random_state=42)
    best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    
    results = best_model.evals_result()
    history = {"train_loss": [x**2 for x in results['validation_0']['rmse']], 
               "val_loss":[x**2 for x in results['validation_1']['rmse']]}
    
    best_model.save_model(MODELS_DIR / "XGBoost_best.json")
    (MODELS_DIR / "XGBoost_history.json").write_text(json.dumps(history))


def optimize_arima(arima_data):
    train_seq, train_val_seq, y_val_true = arima_data
    logger.info("="*60 + "\n🚀 开始 ARIMA 150次超参搜索 (滚动1步验证机制)...")
    
    def objective(trial):
        p = trial.suggest_int('p', 0, 5)
        d = trial.suggest_int('d', 0, 1) # 时序通常一阶差分即可平稳
        q = trial.suggest_int('q', 0, 5)
        try:
            # 在 train_seq 上拟合参数
            model = ARIMA(train_seq, order=(p, d, q)).fit()
            
            # 核心技巧：使用 apply 方法将训练好的参数作用于包含 val 的更长序列
            # 这会产生单步滚动预测，fittedvalues 的最后 len(y_val_true) 个值刚好与 DataLoader 里的 val_y 一一对应！
            res = model.apply(train_val_seq)
            arima_val_preds = res.fittedvalues[-len(y_val_true):]
            
            mse = mean_squared_error(y_val_true, arima_val_preds)
            return mse if not np.isnan(mse) else float('inf')
        except Exception:
            return float('inf')

    study = optuna.create_study(direction='minimize', study_name="ARIMA_Baseline")
    study.optimize(objective, n_trials=N_TRIALS)
    
    logger.success(f"✅ ARIMA 最佳参数: {study.best_params}, 最佳 MSE: {study.best_value:.6f}")
    
    # 拟合最终模型并保存
    best_model = ARIMA(train_val_seq, order=(study.best_params['p'], study.best_params['d'], study.best_params['q'])).fit()
    with open(MODELS_DIR / "ARIMA_best.pkl", "wb") as f:
        pickle.dump(best_model, f)


if __name__ == "__main__":
    xgb_data, arima_data = get_strictly_aligned_data()
    
    optimize_xgb(xgb_data)
    
    # ARIMA 在大图上搜索太慢，为了提速可以只取过去 2000 个点作为搜索基础。
    # 这里我们完整传入以保证严谨性，如果觉得慢，可将 n_trials 调为 50。
    optimize_arima(arima_data)
    
    logger.success("所有异构 Baseline 搜索及训练完毕，现在你可以运行 evaluator.py 进行终极对决了！")