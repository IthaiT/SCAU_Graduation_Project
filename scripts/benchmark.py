"""V6.1 跨模态高维特征多轮基准测试: 基于 Optuna 最优超参数。

严格对齐:
- 数据源: csi300_features_advanced.csv (30+维高阶特征)
- 数据流: StandardScaler
- 模型超参数: 独立异构 (配合 FeatureProjection 架构)
- 训练策略: 独立的 LR, Batch Size, Weight Decay + ReduceLROnPlateau
"""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402

from src.data.dataset import get_dataloaders  # noqa: E402
from src.engine.trainer import train_model  # noqa: E402
from src.models.networks import (  # noqa: E402
    LSTMModel,
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# ── 基础与数据切分参数 ────────────────────────────────────────────
NUM_RUNS = 10
SEQ_LEN = 60
PRED_LEN = 1
EPOCHS = 100
PATIENCE = 15     # 严格对齐 Optuna 搜索时的 15
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10

# ── 核心: Optuna 全局最优超参数配置 (⚠️请在跑完新一轮 Optuna 后更新这里) ──
BEST_CONFIGS: dict[str, dict[str, Any]] = {
    "LSTM": {
        "train_args": {"batch_size": 64, "lr": 0.002014, "weight_decay": 2.226e-05},
        "model_args": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.437},
    },
    "Transformer": {
        "train_args": {"batch_size": 32, "lr": 0.000411, "weight_decay": 0.000992},
        "model_args": {"d_model": 64, "num_heads": 8, "num_layers": 2, "ffn_dim": 64, "dropout": 0.131},
    },
    "LSTM_Transformer": {
        "train_args": {"batch_size": 64, "lr": 0.004220, "weight_decay": 2.200e-05},
        "model_args": {"hidden_dim": 64, "num_lstm_layers": 1, "num_transformer_layers": 2, "num_heads": 4, "ffn_dim": 128, "dropout": 0.384},
    },
    "Parallel_LSTM_Transformer": {
        "train_args": {"batch_size": 32, "lr": 0.000414, "weight_decay": 2.680e-06},
        "model_args": {"hidden_dim": 32, "num_lstm_layers": 1, "num_transformer_layers": 1, "num_heads": 4, "ffn_dim": 256, "dropout": 0.184},
    },
}

MODEL_NAMES = list(BEST_CONFIGS.keys())
MODEL_LABELS = {
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "LSTM_Transformer": "Serial LSTM-Trans",
    "Parallel_LSTM_Transformer": "Parallel LSTM-Trans",
}

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "benchmark_results_advanced.txt"


# ── 工具函数 ──────────────────────────────────────────────────────
def _build_model(model_name: str, input_dim: int, kwargs: dict[str, Any]) -> nn.Module:
    """遵循基于字典 kwargs 动态解包的工厂模式，适配 FeatureProjection"""
    if model_name == "LSTM":
        return LSTMModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "Transformer":
        return TransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "LSTM_Transformer":
        return LSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, seq_len=SEQ_LEN, **kwargs)
    elif model_name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    raise ValueError(f"未知模型: {model_name}")


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((true - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / (np.abs(true) + 1e-8))) * 100)
    r2 = float(r2_score(true, pred))
    da = float(np.mean(np.sign(np.diff(true)) == np.sign(np.diff(pred))) * 100)
    return {
        "MSE": round(mse, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "MAPE": round(mape, 4), "R2": round(r2, 4), "DA": round(da, 2),
    }


@torch.no_grad()
def _evaluate_model(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    preds, trues = [],[]
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred, _ = model(X)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


# ── 单轮: 训练 + 评估 (高内聚流水线) ──────────────────────────────
def run_once(
    device: torch.device,
    df_values: np.ndarray,
    columns: list[str],
    num_features: int,
) -> dict[str, dict[str, float]]:
    """单轮完整生命周期：隔离训练状态，独立化 DataLoader。"""
    
    tmp_dir = MODELS_DIR / "benchmark_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict[str, float]] = {}

    for name in MODEL_NAMES:
        config = BEST_CONFIGS[name]
        train_args = config["train_args"]
        model_args = config["model_args"]
        
        # 为该架构创建特定 Batch Size 的 Dataloader (适配 StandardScaler)
        train_loader, val_loader, test_loader, scaler_target = get_dataloaders(
            df_values=df_values, columns=columns,
            seq_len=SEQ_LEN, pred_len=PRED_LEN, batch_size=train_args["batch_size"],
            target_col="close", train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
        )

        model = _build_model(name, num_features, model_args)
        save_path = tmp_dir / f"{name}_best.pth"
        logger.info("  -> 训练: {} (Batch={}, LR={:.1e})", name, train_args["batch_size"], train_args["lr"])

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=train_args["lr"], 
            weight_decay=train_args["weight_decay"]
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        # 训练
        train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, device=device,
            epochs=EPOCHS, patience=PATIENCE, save_path=save_path,
            scheduler=scheduler, max_grad_norm=1.0,
        )

        # 评估 (强制重新加载最佳权重)
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.to(device)
        
        preds_norm, trues_norm = _evaluate_model(model, test_loader, device)
        
        # 此时 scaler_target 已经是 StandardScaler，inverse_transform 依然安全
        pred_r = scaler_target.inverse_transform(preds_norm).ravel()
        true_r = scaler_target.inverse_transform(trues_norm).ravel()
        
        all_metrics[name] = compute_metrics(true_r, pred_r)

        # 彻底释放显存 (防止 40 次高维特征训练挤爆 GPU)
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return all_metrics


# ── 格式化输出 ────────────────────────────────────────────────────
def _fmt_table_header() -> str:
    return (
        f"{'-'*84}\n"
        f"{'Model':<20s} {'MSE':>10s} {'RMSE':>10s} {'MAE':>10s} {'R²':>10s} {'MAPE%':>10s} {'DA%':>10s}\n"
        f"{'-'*84}"
    )


def _fmt_row(label: str, m: dict[str, float]) -> str:
    return f"{label:<20s} {m['MSE']:>10.2f} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} {m['R2']:>10.4f} {m['MAPE']:>10.2f} {m['DA']:>10.2f}"


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: {} | 总轮次: {}", device, NUM_RUNS)

    # ⚠️ 数据源切换：读取高级多模态特征
    csv_path = PROJECT_ROOT / "data" / "csi300_features_advanced.csv"
    if not csv_path.exists():
        logger.error("高级特征数据文件未找到: {}。请先运行 build_advanced_features.py！", csv_path)
        sys.exit(1)
        
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df_values = df.values
    columns = df.columns.tolist()
    num_features = len(columns)

    all_runs: list[dict[str, dict[str, float]]] =[]
    lines: list[str] =[
        f"Benchmark 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"重复次数: {NUM_RUNS}  |  设备: {device}  |  高阶特征数: {num_features}",
        f"超参数: 基于 Optuna HPO 独立异构 (包含 FeatureProjection)",
        f"统一设置: SEQ_LEN={SEQ_LEN}, EPOCHS={EPOCHS}, PATIENCE={PATIENCE}",
        ""
    ]

    for run_idx in range(1, NUM_RUNS + 1):
        logger.info("=" * 60)
        logger.info(">>> 第 {}/{} 轮", run_idx, NUM_RUNS)
        logger.info("=" * 60)

        metrics = run_once(device, df_values, columns, num_features)
        all_runs.append(metrics)

        lines.append(f"===== 第 {run_idx}/{NUM_RUNS} 轮 =====")
        lines.append(_fmt_table_header())
        for name in MODEL_NAMES:
            label = MODEL_LABELS[name]
            m = metrics[name]
            lines.append(_fmt_row(label, m))
            logger.info(
                "  {} → MSE={:.2f}  RMSE={:.2f}  MAE={:.2f}  R²={:.4f}  MAPE={:.2f}%  DA={:.2f}%",
                label, m["MSE"], m["RMSE"], m["MAE"], m["R2"], m["MAPE"], m["DA"]
            )
        lines.append("-" * 84 + "\n")

    # ── 计算均值 & 标准差 ─────────────────────────────────────────
    lines.extend([
        "=" * 120,
        f"  {NUM_RUNS} 轮平均 ± 标准差 (Advanced Features + HPO)",
        "=" * 120,
        f"{'Model':<20s} {'MSE':>16s} {'RMSE':>16s} {'MAE':>16s} {'R²':>16s} {'MAPE%':>16s} {'DA%':>16s}",
        "-" * 120
    ])

    logger.info("=" * 80)
    logger.info("  {} 轮统计汇总", NUM_RUNS)
    logger.info("=" * 80)

    for name in MODEL_NAMES:
        label = MODEL_LABELS[name]
        vals = {k: [run[name][k] for run in all_runs] for k in["MSE", "RMSE", "MAE", "MAPE", "R2", "DA"]}
        means = {k: np.mean(v) for k, v in vals.items()}
        stds = {k: np.std(v) for k, v in vals.items()}

        row = (
            f"{label:<20s}"
            f" {means['MSE']:>7.2f}±{stds['MSE']:<7.2f}"
            f" {means['RMSE']:>7.2f}±{stds['RMSE']:<7.2f}"
            f" {means['MAE']:>7.2f}±{stds['MAE']:<7.2f}"
            f" {means['R2']:>7.4f}±{stds['R2']:<7.4f}"
            f" {means['MAPE']:>7.2f}±{stds['MAPE']:<7.2f}"
            f" {means['DA']:>7.2f}±{stds['DA']:<7.2f}"
        )
        lines.append(row)
        logger.info(
            "{:<20s} MSE={:.2f}±{:.2f}  RMSE={:.2f}±{:.2f}  MAE={:.2f}±{:.2f}  "
            "R²={:.4f}±{:.4f}  MAPE={:.2f}%±{:.2f}  DA={:.2f}%±{:.2f}",
            label, means["MSE"], stds["MSE"], means["RMSE"], stds["RMSE"],
            means["MAE"], stds["MAE"], means["R2"], stds["R2"],
            means["MAPE"], stds["MAPE"], means["DA"], stds["DA"]
        )

    lines.append("=" * 120)
    lines.append(f"\nBenchmark 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 写入文件 ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    
    json_path = RESULTS_DIR / "benchmark_results_advanced.json"
    json_path.write_text(json.dumps(all_runs, indent=2, ensure_ascii=False), encoding="utf-8")
    
    logger.info("完整报告已保存 → {}", OUTPUT_FILE)
    logger.info("JSON 数据已保存 → {}", json_path)


if __name__ == "__main__":
    main()