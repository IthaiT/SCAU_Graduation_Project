"""V7.1 顶刊级评估流水线: 归一化 Loss 轴、局部放大分面视图、KDE 误差分析。"""
from __future__ import annotations

import json
import sys
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger

from src.data.dataset import get_dataloaders
from src.models.networks import LSTMModel, LSTMTransformerModel, TransformerModel

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")


# ── 全局科学绘图风格设置 ──────────────────────────────────────────
sns.set_theme(style="ticks", palette="muted") 
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif":["Microsoft YaHei", "SimHei", "Arial"], 
    "axes.unicode_minus": False, "mathtext.fontset": "stix", 
    "figure.dpi": 300, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10,
    "grid.alpha": 0.25, "grid.linestyle": "--", "axes.linewidth": 1.2
})

_COLORS = {
    "ARIMA": "#999999", "XGBoost": "#F39B7F", "LSTM": "#E64B35", 
    "Transformer": "#4DBBD5", "LSTM_Transformer": "#00A087",
}
_LABELS = {
    "ARIMA": "Baseline: ARIMA", 
    "XGBoost": "Baseline: XGBoost", 
    "LSTM": "Baseline: LSTM", 
    "Transformer": "Baseline: Transformer", 
    "LSTM_Transformer": "Proposed: LSTM-Trans",
}

# ── 配置与状态 ───────────────────────────────────────────────────
SEQ_LEN = 60
PRED_LEN = 1
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAMES =["ARIMA", "XGBoost", "LSTM", "Transformer", "LSTM_Transformer"]

# 深度学习历史最佳参数 (与 train/optimize 保持一致)
BEST_ARCH_CONFIGS: dict[str, dict[str, Any]] = {
    "LSTM": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.21597852403871323},
    "Transformer": {"d_model": 128, "num_heads": 2, "num_layers": 2, "ffn_dim": 4 * 128, "dropout": 0.2646631095765896},
    "LSTM_Transformer": {"hidden_dim": 128, "num_lstm_layers": 1, "num_transformer_layers": 2, "num_heads": 4, "ffn_dim": 2 * 128, "dropout": 0.40810640767923295},
}


# ── 工具函数 ─────────────────────────────────────────────────────
def smooth_curve(points: list[float], factor: float = 0.85) -> list[float]:
    s_points =[]
    for p in points:
        s_points.append(s_points[-1] * factor + p * (1 - factor) if s_points else p)
    return s_points

def compute_metrics(true: NDArray, pred: NDArray) -> dict[str, float]:
    mse = float(np.mean((true - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100)
    r2 = float(r2_score(true, pred))
    da = float(np.mean(np.sign(np.diff(true.ravel())) == np.sign(np.diff(pred.ravel()))) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "DA": da}

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ── 异构推理引擎 ────────────────────────────────────────────────
def inference_all_models(df: pd.DataFrame, columns: list[str], device: torch.device):
    num_features = len(columns)
    
    _, _, test_loader, scaler_target = get_dataloaders(
        df_values=df.values, columns=columns, seq_len=SEQ_LEN, pred_len=PRED_LEN, 
        batch_size=128, target_col="close", train_ratio=0.72, val_ratio=0.10,
    )
    
    N_test = len(test_loader.dataset)
    test_y_norm = test_loader.dataset.y.numpy().ravel()
    true_real = scaler_target.inverse_transform(test_y_norm.reshape(-1, 1)).ravel()
    
    preds_real_dict = {}
    
    for name in MODEL_NAMES:
        try:
            if name in BEST_ARCH_CONFIGS:
                k = BEST_ARCH_CONFIGS[name]
                if name == "LSTM": model = LSTMModel(input_dim=num_features, pred_len=PRED_LEN, **k)
                elif name == "Transformer": model = TransformerModel(input_dim=num_features, pred_len=PRED_LEN, **k)
                else: model = LSTMTransformerModel(input_dim=num_features, pred_len=PRED_LEN, seq_len=SEQ_LEN, **k)
                
                model.load_state_dict(torch.load(MODELS_DIR / f"{name}_best.pth", map_location=device, weights_only=True))
                model.to(device).eval()
                
                preds =[]
                with torch.no_grad():
                    for X, _ in test_loader:
                        out = model(X.to(device))
                        preds.append((out[0] if isinstance(out, tuple) else out).cpu().numpy())
                pred_norm = np.concatenate(preds).ravel()
                
            elif name == "XGBoost":
                model = xgb.XGBRegressor()
                model.load_model(MODELS_DIR / "XGBoost_best.json")
                xgb_X = test_loader.dataset.X.numpy().reshape(N_test, -1)
                pred_norm = model.predict(xgb_X)
                
            elif name == "ARIMA":
                with open(MODELS_DIR / "ARIMA_best.pkl", "rb") as f:
                    model = pickle.load(f)
                target_idx = columns.index("close")
                full_scaled_target = scaler_target.transform(df.values[:, target_idx:target_idx+1]).ravel()
                res = model.apply(full_scaled_target)
                pred_norm = res.fittedvalues[-N_test:]

            preds_real_dict[name] = scaler_target.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
            logger.info(f"{name} 推理成功。")
            
        except Exception as e:
            logger.error(f"跳过 {name}: 可能是模型权重文件尚未生成，异常信息: {e}")
            
    return true_real, preds_real_dict


# ── 学术级绘图套件 ────────────────────────────────────────────────

def plot_loss_curves(save_path: Path) -> None:
    """归一化 X 轴的 Loss 收敛曲线 (解决 XGBoost 与 DL 迭代次数不一致的问题)"""
    histories = {}
    for name in["XGBoost","LSTM", "Transformer", "LSTM_Transformer"]:
        path = MODELS_DIR / f"{name}_history.json"
        if path.exists(): histories[name] = json.loads(path.read_text())
            
    if not histories: return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    titles =[("train_loss", "训练集收敛轨迹 (Train Loss)"), ("val_loss", "验证集收敛轨迹 (Val Loss)")]
    
    for ax, (key, title) in zip(axes, titles):
        for name, hist in histories.items():
            raw_loss = hist[key]
            smoothed_loss = smooth_curve(raw_loss, factor=0.85)
            
            # 核心修改：将长短不一的 Epochs/Trees 统一映射到 0% ~ 100% 的训练进度
            progress = np.linspace(0, 100, len(raw_loss))
            
            color = _COLORS.get(name)
            label = _LABELS.get(name, name)
            
            ax.plot(progress, raw_loss, color=color, alpha=0.20, linewidth=1.0)
            ax.plot(progress, smoothed_loss, color=color, label=label, linewidth=2.0, alpha=0.9)
                    
        ax.set_xlabel("相对训练进度 (Relative Training Progress %)", fontweight="bold")
        ax.set_ylabel("MSE Loss (Log Scale)", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_yscale("log") 
        ax.legend(frameon=True, edgecolor="black")
        sns.despine(ax=ax)
        
    _save(fig, save_path)


def plot_predictions_faceted(true: NDArray, preds_dict: dict[str, NDArray], save_path: Path, last_n: int = 200) -> None:
    """分面视图: 局部截取 + 图例外置 (完美解决文字遮挡与细节看不清)"""
    n_models = len(preds_dict)
    
    # 核心修改：只截取最后 last_n 个时间步进行精细展示
    true_slice = true[-last_n:]
    x = np.arange(len(true_slice))
    
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(12, 2.2 * n_models), sharex=True, constrained_layout=True)
    if n_models == 1: axes = [axes]
    
    for ax, (name, pred) in zip(axes, preds_dict.items()):
        pred_slice = pred[-last_n:]
        
        ax.plot(x, true_slice, color="#333333", linewidth=2.0, label="真实点位 (Ground Truth)", zorder=5)
        ax.plot(x, pred_slice, color=_COLORS.get(name), linewidth=1.5, label=_LABELS.get(name, name), zorder=10)
        
        ax.set_ylabel("Price", fontweight="bold")
        
        # 核心修改：强制将图例放置在绘图区域的最右侧外部，杜绝遮挡
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, edgecolor="black")
        
        ax.grid(True, linestyle=":", alpha=0.6)
        sns.despine(ax=ax)
        
    axes[-1].set_xlabel(f"测试集末端时序步 (Last {last_n} Time Steps)", fontweight="bold")
    fig.suptitle(f"各模型预测值与真实值分面对比", fontsize=16, fontweight="bold")
    _save(fig, save_path)


def plot_error_distribution_kde(true: NDArray, preds_dict: dict[str, NDArray], save_path: Path) -> None:
    """KDE 误差分布证明图: 越窄越高且对称于 0，模型越强"""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    for name, pred in preds_dict.items():
        errors = pred - true
        sns.kdeplot(errors, ax=ax, label=_LABELS.get(name, name), 
                    color=_COLORS.get(name), fill=True, alpha=0.15, linewidth=2)
                    
    ax.axvline(x=0, color="#555555", linestyle="--", linewidth=1.5, alpha=0.8, label="Zero Error (零偏误)")
    
    ax.set_xlabel("预测误差残差 (Predicted - Actual)", fontweight="bold")
    ax.set_ylabel("概率密度分布 (Density)", fontweight="bold")
    ax.set_title("全基线误差核密度估计 (Error KDE) - 验证正态白噪声假设", fontweight="bold", pad=12)
    ax.legend(frameon=True, edgecolor="black")
    sns.despine(ax=ax)
    
    _save(fig, save_path)


def plot_scatter_fit_grid(true: NDArray, preds_dict: dict[str, NDArray], save_path: Path) -> None:
    """自适应的 2x3 R² 散点图网格"""
    n_models = len(preds_dict)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    axes = axes.flatten()
    
    min_val = min(np.min(true), min(np.min(p) for p in preds_dict.values())) * 0.98
    max_val = max(np.max(true), max(np.max(p) for p in preds_dict.values())) * 1.02
    
    for i, (name, pred) in enumerate(preds_dict.items()):
        ax = axes[i]
        ax.scatter(true, pred, alpha=0.5, color=_COLORS.get(name), edgecolors="white", linewidths=0.2, s=20)
        ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', lw=1.2, alpha=0.6)
        
        m, b = np.polyfit(true.flatten(), pred.flatten(), 1)
        ax.plot(true, m*true + b, color='#E64B35', lw=1.5, alpha=0.8)
        
        r2 = r2_score(true, pred)
        ax.text(0.05, 0.95, f"$R^2 = {r2:.4f}$", transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
        
        ax.set_title(_LABELS.get(name, name), fontweight="bold")
        ax.set_xlabel("Actual (真实值)", fontweight="bold")
        ax.set_ylabel("Predicted (预测值)", fontweight="bold")
        
        ax.set_xlim(min_val, max_val); ax.set_ylim(min_val, max_val)
        ax.grid(alpha=0.2)
        sns.despine(ax=ax)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle("多模型预测稳健性散点拟合分析", fontsize=16, fontweight="bold")
    _save(fig, save_path)


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("启动终极科研级评估流水线 (Linus Standard)")

    df = pd.read_csv(PROJECT_ROOT / "data" / "final_data.csv", index_col="date", parse_dates=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    true_real, preds_real = inference_all_models(df, df.columns.tolist(), device)

    if not preds_real: return

    all_metrics = {name: compute_metrics(true_real, pred) for name, pred in preds_real.items()}

    logger.info("正在渲染学术可视化图表 (PDF矢量标准)...")
    plot_loss_curves(RESULTS_DIR / "01_loss_curves_with_xgb.png")
    # 默认展示最后 200 个时间步，既有大跌大涨的波峰波谷，又非常清晰
    plot_predictions_faceted(true_real, preds_real, RESULTS_DIR / "02_predictions_faceted.png", last_n=200)
    plot_error_distribution_kde(true_real, preds_real, RESULTS_DIR / "03_error_distribution_kde.png")
    plot_scatter_fit_grid(true_real, preds_real, RESULTS_DIR / "04_scatter_fit_grid.png")

    logger.info("\n" + "=" * 80)
    logger.info("{:<22s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>6s}", "Model Architecture", "MSE", "RMSE", "MAE", "R²", "MAPE%", "DA%")
    logger.info("-" * 80)
    for name in MODEL_NAMES:
        if name in all_metrics:
            m = all_metrics[name]
            logger.info("{:<22s} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.4f} | {:>8.2f} | {:>6.2f}",
                         _LABELS.get(name, name), m["MSE"], m["RMSE"], m["MAE"], m["R2"], m["MAPE"], m["DA"])
    logger.info("=" * 80)
    
    logger.success(f"评估完美结束！所有的精美学术图表已存入: {RESULTS_DIR}")

if __name__ == "__main__":
    main()