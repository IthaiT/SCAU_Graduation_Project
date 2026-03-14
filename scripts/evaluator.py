"""V6.3 科研级评估流水线: 极高渲染质量、局部时序放大、核密度误差分布与高级自适应热力图。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from src.data.dataset import get_dataloaders
from src.models.networks import (
    LSTMModel,
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

# ── 核心工具：EMA 平滑算法 (学术绘图必备) ──────────────────────
def smooth_curve(points: list[float], factor: float = 0.85) -> list[float]:
    """指数移动平均 (EMA) 平滑，用于学术 Loss 曲线绘制"""
    smoothed_points =[]
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# ── 全局绘图风格 (Nature/Science 中文期刊排版规范) ────────────────
# 必须先调用 seaborn 的 set_theme，否则会覆盖自定义字体
sns.set_theme(style="ticks", palette="muted") 

# 然后再更新 matplotlib 的参数
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    # Windows 环境优先使用 Microsoft YaHei，回退使用 SimHei
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"], 
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix", # 让数学公式(如 R^2)的字体看起来像 Times New Roman
    "figure.dpi": 300,          # 提升渲染分辨率，300是大部分期刊的基础要求
    "savefig.dpi": 600,         # 论文级最终输出极高分辨率
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.titlesize": 13,       # 标题不宜过大
    "axes.labelsize": 11,       # 坐标轴标签适中
    "axes.linewidth": 1.2,      # 加粗坐标轴边框，增加质感
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.alpha": 0.25,         # 弱化网格线
    "grid.linestyle": "--"
})

# Nature/Science 风格建议色
_COLORS: dict[str, str] = {
    "LSTM": "#E64B35",               # 朱红色
    "Transformer": "#4DBBD5",        # 湖蓝色
    "LSTM_Transformer": "#00A087",   # 翠绿色
}
_LABELS: dict[str, str] = {
    "LSTM": "基线 LSTM",
    "Transformer": "基线 Transformer",
    "LSTM_Transformer": "串行 LSTM-Trans",
}

# ── 基础配置 ──────────────────────────────────────────────────────
SEQ_LEN = 60
PRED_LEN = 1
EVAL_BATCH_SIZE = 128
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAMES = ["LSTM", "Transformer", "LSTM_Transformer"]

# 更新最优模型参数
BEST_ARCH_CONFIGS: dict[str, dict[str, Any]] = {
    "LSTM": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.21597852403871323},
    "Transformer": {"d_model": 128, "num_heads": 2, "num_layers": 2, "ffn_dim": 4 * 128, "dropout": 0.2646631095765896},
    "LSTM_Transformer": {"hidden_dim": 128, "num_lstm_layers": 1, "num_transformer_layers": 2, "num_heads": 4, "ffn_dim": 2 * 128, "dropout": 0.40810640767923295},
}

# ── 恢复的真实模型操作逻辑 ──────────────────────────────────────────
def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def _build_model(model_name: str, input_dim: int) -> nn.Module:
    kwargs = BEST_ARCH_CONFIGS[model_name]
    if model_name == "LSTM":
        return LSTMModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "Transformer":
        return TransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "LSTM_Transformer":
        return LSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, seq_len=SEQ_LEN, **kwargs)
    elif model_name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    raise ValueError(f"未知模型: {model_name}")

@torch.no_grad()
def _inference(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    device: torch.device
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]:
    model.eval()
    preds, trues = [],[]
    attn_w_last: NDArray[np.float32] | None = None
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        # 安全的多态接收：兼容返回单个 Tensor 或 (Tensor, Tensor) 的模型
        out = model(X)
        if isinstance(out, tuple):
            pred = out[0]
            extra = out[1]
        else:
            pred = out
            extra = None

        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
        
        # 提取注意力权重 (只保存最后一个 Batch 用于画图)
        if extra is not None and isinstance(extra, torch.Tensor):
            # 取该 Batch 的最后一个样本来画图
            attn_w_last = extra[-1].cpu().numpy()
            
    return np.concatenate(preds), np.concatenate(trues), attn_w_last

def compute_metrics(true: NDArray, pred: NDArray) -> dict[str, float]:
    mse = float(np.mean((true - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100)
    r2 = float(r2_score(true, pred))
    da = float(np.mean(np.sign(np.diff(true.ravel())) == np.sign(np.diff(pred.ravel()))) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "DA": da}


# ── 高级科研可视化库 ──────────────────────────────────────────────

def plot_loss_curves(histories: dict[str, dict[str, list[float]]], save_path: Path) -> None:
    """双线叠加(原始震荡+EMA平滑)的专业 Loss 曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    titles =[("train_loss", "训练集损失 (MSE)"), ("val_loss", "验证集损失 (MSE)")]
    
    for ax, (key, title) in zip(axes, titles):
        for name, hist in histories.items():
            raw_loss = hist[key]
            smoothed_loss = smooth_curve(raw_loss, factor=0.85) # 0.85 平滑度适中
            epochs = range(1, len(raw_loss) + 1)
            
            color = _COLORS.get(name)
            label = _LABELS.get(name, name)
            
            # 1. 绘制底层原始震荡曲线 (半透明)
            ax.plot(epochs, raw_loss, color=color, alpha=0.25, linewidth=1.0)
            # 2. 绘制上层平滑曲线 (实线加粗，代表主要趋势)
            ax.plot(epochs, smoothed_loss, color=color, label=label, linewidth=2.0, alpha=0.9)
                    
        ax.set_xlabel("迭代轮次 (Epochs)", fontweight="bold")
        ax.set_ylabel("损失值 (Loss)", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_yscale("log") 
        ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="upper right")
        ax.grid(True, which="major", ls="--", alpha=0.3)
        sns.despine(ax=ax) # 移除上、右边框，增强学术感
        
    fig.suptitle("模型收敛轨迹对比 (Optuna 超参优化后)", fontsize=15, fontweight="bold")
    _save(fig, save_path)


def plot_predictions_with_inset(true: NDArray, preds_dict: dict[str, NDArray], save_path: Path, last_n: int = 150) -> None:
    """带局部放大 (Inset) 的高质量序列预测图"""
    tail_true = true
    x = np.arange(len(tail_true))
    
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    
    # Ground Truth 使用深灰色而非纯黑，减少生硬感
    ax.plot(x, tail_true, color="#333333", linewidth=2.5, label="真实值 (Ground Truth)", zorder=5)
    
    styles =["-", "--", "-.", ":"]
    for i, (name, pred) in enumerate(preds_dict.items()):
        ax.plot(x, pred, linestyle=styles[i % len(styles)],
                color=_COLORS.get(name), linewidth=1.5,
                label=_LABELS.get(name, name), alpha=0.85)
                
    ax.set_xlabel(f"时间步 (测试集最后 {last_n} 期)", fontweight="bold")
    ax.set_ylabel("沪深300指数 (实际点位)", fontweight="bold")
    ax.set_title("测试集多模型预测结果与真实值拟合对比", fontweight="bold", pad=12)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="upper left")
    sns.despine(ax=ax)
    
    # --- 构造局部放大图 (放大震荡最剧烈的区域) ---
    # zoom_n = 30
    # axins = inset_axes(ax, width="35%", height="40%", loc="lower right", borderpad=2.5)
    # axins.plot(x[-zoom_n:], tail_true[-zoom_n:], color="#333333", linewidth=2.0)
    # for i, (name, pred) in enumerate(preds_dict.items()):
    #     axins.plot(x[-zoom_n:], pred[-zoom_n:], linestyle=styles[i % len(styles)],
    #                color=_COLORS.get(name), linewidth=1.2, alpha=0.9)
    #                
    # axins.set_xlim(x[-zoom_n], x[-1])
    # axins.set_ylim(min(tail_true[-zoom_n:]) * 0.98, max(tail_true[-zoom_n:]) * 1.02)
    # axins.tick_params(axis='both', which='major', labelsize=8)
    # axins.grid(True, linestyle=":", alpha=0.5)
    # 
    # # 绘制连接线
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", linewidth=1.0, alpha=0.6)
    
    _save(fig, save_path)


def plot_error_distribution_kde(true: NDArray, preds_dict: dict[str, NDArray], save_path: Path) -> None:
    """核密度估计 (KDE) 误差分布图"""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    for name, pred in preds_dict.items():
        errors = pred - true
        sns.kdeplot(errors, ax=ax, label=_LABELS.get(name, name), 
                    color=_COLORS.get(name), fill=True, alpha=0.15, linewidth=2, bw_adjust=1.2)
                    
    ax.axvline(x=0, color="#555555", linestyle="--", linewidth=1.5, alpha=0.8, label="零误差基准线")
    
    ax.set_xlabel("预测误差 (预测值 - 真实值)", fontweight="bold")
    ax.set_ylabel("概率密度 (Density)", fontweight="bold")
    ax.set_title("各模型预测误差的核密度估计 (KDE) 分布", fontweight="bold", pad=12)
    ax.legend(frameon=True, fancybox=False, edgecolor="black")
    sns.despine(ax=ax)
    ax.grid(alpha=0.2)
    
    _save(fig, save_path)


def plot_scatter_fit(true: NDArray, preds_dict: dict[str, NDArray], save_path: Path) -> None:
    """拟合度散点图 (R² 表现)"""
    n_models = len(preds_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), constrained_layout=True)
    if n_models == 1: axes = [axes]
    
    min_val = min(np.min(true), min(np.min(p) for p in preds_dict.values())) * 0.98
    max_val = max(np.max(true), max(np.max(p) for p in preds_dict.values())) * 1.02
    
    for ax, (name, pred) in zip(axes, preds_dict.items()):
        ax.scatter(true, pred, alpha=0.4, color=_COLORS.get(name), edgecolors="white", linewidths=0.2, s=25)
        
        ax.plot([min_val, max_val],[min_val, max_val], color='black', linestyle='--', lw=1.2, alpha=0.6, label="完美预测线 (y=x)")
        
        m, b = np.polyfit(true.flatten(), pred.flatten(), 1)
        ax.plot(true, m*true + b, color='#E64B35', lw=1.5, alpha=0.9, label="线性回归拟合")
        
        r2 = r2_score(true, pred)
        ax.text(0.05, 0.95, f"$R^2 = {r2:.4f}$", transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
        
        ax.set_title(f"{_LABELS.get(name, name)} 拟合散点", fontweight="bold")
        ax.set_xlabel("真实值 (Actual)", fontweight="bold")
        if ax == axes[0]: 
            ax.set_ylabel("预测值 (Predicted)", fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
            
        ax.grid(alpha=0.2)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        sns.despine(ax=ax)
        
    fig.suptitle("多模型预测值与真实值线性拟合散点分析", fontsize=15, fontweight="bold")
    _save(fig, save_path)


def plot_attention_heatmap_adaptive(attn: NDArray, save_path: Path) -> None:
    """自适应维度的极致热力图"""
    attn_sq = np.squeeze(attn)
    
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    
    sns.heatmap(
        attn_sq, ax=ax, 
        cmap="mako",           
        xticklabels=10,        
        yticklabels=10, 
        linewidths=0.0, 
        rasterized=True,       
        cbar_kws={"shrink": 0.8, "label": "注意力权重 (Attention Weight)"}
    )
                
    ax.set_xlabel("键序列 Key (历史时间步)", fontweight="bold")
    ax.set_ylabel("查询序列 Query (目标时间步)", fontweight="bold")
    ax.set_title("LSTM-Transformer Pre-Norm 自注意力权重矩阵分布", fontweight="bold", pad=15)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.invert_yaxis()
    
    _save(fig, save_path)


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("启动科研级评估流水线 | 设备: {}", device)

    csv_path = PROJECT_ROOT / "data" / "final_data.csv"
    if not csv_path.exists():
        logger.error("未找到 final_data.csv, 脚本退出。")
        sys.exit(1)
        
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)

    _, _, test_loader, scaler_target = get_dataloaders(
        df_values=df.values, columns=columns,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=EVAL_BATCH_SIZE,
        target_col="close", train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 绘制损失曲线
    histories: dict[str, dict[str, list[float]]] = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}_history.json"
        if path.exists():
            histories[name] = json.loads(path.read_text())
    if histories:
        plot_loss_curves(histories, RESULTS_DIR / "01_loss_curves.png")

    all_metrics: dict[str, dict[str, float]] = {}
    preds_real: dict[str, NDArray] = {}
    true_real: NDArray | None = None
    attn_for_heatmap: NDArray | None = None

    # 执行推理
    for name in MODEL_NAMES:
        weight_path = MODELS_DIR / f"{name}_best.pth"
        if not weight_path.exists():
            logger.warning("跳过 {}: 权重文件不存在", name)
            continue
            
        model = _build_model(name, num_features)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)

        preds_norm, trues_norm, attn_w = _inference(model, test_loader, device)

        pred_r = scaler_target.inverse_transform(preds_norm).ravel()
        true_r = scaler_target.inverse_transform(trues_norm).ravel()

        preds_real[name] = pred_r
        if true_real is None: true_real = true_r

        all_metrics[name] = compute_metrics(true_r, pred_r)
        
        # 提取 LSTM_Transformer 的注意力进行可视化
        if name == "LSTM_Transformer" and attn_w is not None:
            attn_for_heatmap = attn_w

    if not all_metrics:
        logger.error("未找到任何模型的预测结果，请先运行 train.py。")
        return

    # 生成高水平图表集
    assert true_real is not None
    logger.info("生成时间序列拟合图...")
    plot_predictions_with_inset(true_real, preds_real, RESULTS_DIR / "02_predictions_inset.png")
    
    logger.info("生成 KDE 误差分布图...")
    plot_error_distribution_kde(true_real, preds_real, RESULTS_DIR / "03_error_distribution_kde.png")
    
    logger.info("生成 R² 拟合散点图...")
    plot_scatter_fit(true_real, preds_real, RESULTS_DIR / "04_scatter_fit.png")
    
    if attn_for_heatmap is not None:
        logger.info("生成自适应机制注意力热力图...")
        plot_attention_heatmap_adaptive(attn_for_heatmap, RESULTS_DIR / "05_attention_heatmap.png")

    # 打印对齐 Markdown 表格格式的最终 Metrics
    logger.info("\n" + "=" * 80)
    logger.info("{:<18s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>8s} | {:>6s}",
                "网络架构", "MSE", "RMSE", "MAE", "R²", "MAPE%", "DA%")
    logger.info("-" * 80)
    for name in MODEL_NAMES:
        if name in all_metrics:
            m = all_metrics[name]
            logger.info("{:<18s} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.4f} | {:>8.2f} | {:>6.2f}",
                         _LABELS.get(name, name), m["MSE"], m["RMSE"], m["MAE"],
                         m["R2"], m["MAPE"], m["DA"])
    logger.info("=" * 80)
    logger.success(f"所有科研级图表已渲染至: {RESULTS_DIR}")


if __name__ == "__main__":
    main()