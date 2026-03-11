"""V6.2 训练流水线: 严格对齐基准测试的高级特征与 Optuna 全局最优超参数。

工程规范:
- 数据源强校验: 强制要求加载 final_data.csv (高维特征集)
- 独立异构策略: 各模型维持独立的 Batch Size 并在循环中独立构建 Dataloader
- 极致整洁: 遵循单一职责，训练后彻底释放 GPU 显存，防止溢出。
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.engine.trainer import train_model
from src.models.networks import (
    LSTMModel,
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# ── 基础任务配置 ──────────────────────────────────────────────────
SEQ_LEN = 60
PRED_LEN = 1
EPOCHS = 100
PATIENCE = 15
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10

# ── 核心: 严格对齐 benchmark.py 的 Optuna 全局最优超参数 ────────
BEST_CONFIGS: dict[str, dict[str, Any]] = {
    "LSTM": {
        "train_args": {"batch_size": 32, "lr": 0.0019883, "weight_decay": 6.536e-05},
        "model_args": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.1747},
    },
    "Transformer": {
        "train_args": {"batch_size": 32, "lr": 0.0001541, "weight_decay": 3.697e-06},
        "model_args": {"d_model": 64, "num_heads": 2, "num_layers": 2, "ffn_dim": 64, "dropout": 0.1016},
    },
    "LSTM_Transformer": {
        "train_args": {"batch_size": 64, "lr": 0.0011091, "weight_decay": 2.067e-05},
        "model_args": {"hidden_dim": 128, "num_lstm_layers": 1, "num_transformer_layers": 2, "num_heads": 8, "ffn_dim": 64, "dropout": 0.4724},
    },
    "Parallel_LSTM_Transformer": {
        "train_args": {"batch_size": 32, "lr": 0.0003143, "weight_decay": 4.412e-05},
        "model_args": {"hidden_dim": 128, "num_lstm_layers": 2, "num_transformer_layers": 2, "num_heads": 4, "ffn_dim": 128, "dropout": 0.1107},
    },
}


def _build_model(model_name: str, input_dim: int, kwargs: dict[str, Any]) -> nn.Module:
    """遵循基于字典 kwargs 动态解包的工厂模式。"""
    if model_name == "LSTM":
        return LSTMModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "Transformer":
        return TransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "LSTM_Transformer":
        return LSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, seq_len=SEQ_LEN, **kwargs)
    elif model_name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    raise ValueError(f"未知模型: {model_name}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("启动高级特征训练流水线 | 设备: {}", device)

    # 1. 数据强校验 (使用 benchmark 对齐的 final_data.csv)
    csv_path = PROJECT_ROOT / "data" / "final_data.csv"
    if not csv_path.exists():
        logger.error("数据文件未找到: {}。请先生成高阶特征！", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    columns = df.columns.tolist()
    num_features = len(columns)
    df_values = df.values

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, config in BEST_CONFIGS.items():
        train_args = config["train_args"]
        model_args = config["model_args"]
        
        logger.info("=" * 60)
        logger.info("🔥 训练架构: {} | 高维特征数: {}", model_name, num_features)

        # 2. 独立构建 Dataloader (适配各模型独立 Batch Size)
        train_loader, val_loader, _, _ = get_dataloaders(
            df_values=df_values,
            columns=columns,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            batch_size=train_args["batch_size"],
            target_col="close",
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
        )

        # 3. 实例化模型与优化器
        model = _build_model(model_name, num_features, model_args)
        logger.info("  -> 模型参数量: {:,}", sum(p.numel() for p in model.parameters() if p.requires_grad))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=train_args["lr"], 
            weight_decay=train_args["weight_decay"]
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        # 4. 委托给核心训练引擎
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=EPOCHS,
            patience=PATIENCE,
            save_path=out_dir / f"{model_name}_best.pth",
            scheduler=scheduler,
            max_grad_norm=1.0,
        )

        # 5. 存储历史数据
        history_path = out_dir / f"{model_name}_history.json"
        history_path.write_text(json.dumps(history, indent=2))
        logger.success("已持久化权重与日志 → {}", out_dir)

        # 6. 工程鲁棒性: 彻底释放显存
        del model, optimizer, scheduler, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()