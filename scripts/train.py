"""V6.1 最终定稿训练脚本: 采用 Optuna 寻优后的独立异构超参数。

学术声明:
为消除超参数偏见, 各模型已分别进行了 50 次 Trial 的全局搜索。
本脚本依据最优解，为各模型分配独立的 Batch Size, LR, Weight Decay 及网络架构。
"""
from __future__ import annotations

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
PATIENCE = 15  # 与搜索时保持一致
TRAIN_RATIO = 0.72
VAL_RATIO = 0.10

# ── 核心: Optuna 全局最优超参数配置 ───────────────────────────────
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


def build_model(model_name: str, input_dim: int, kwargs: dict[str, Any]) -> nn.Module:
    """基于最优配置动态实例化模型。"""
    if model_name == "LSTM":
        return LSTMModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "Transformer":
        return TransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    elif model_name == "LSTM_Transformer":
        # 补充 seq_len 参数 (网络定义中要求的参数)
        return LSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, seq_len=SEQ_LEN, **kwargs)
    elif model_name == "Parallel_LSTM_Transformer":
        return ParallelLSTMTransformerModel(input_dim=input_dim, pred_len=PRED_LEN, **kwargs)
    else:
        raise ValueError(f"未知模型: {model_name}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("启动训练流水线 | 设备: {}", device)

    # 加载基础数据 (仅加载一次)
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
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
        logger.info("🔥 训练架构: {}", model_name)
        logger.info("最优超参数: {}", config)

        # 1. 独立构建 Dataloader (因为 Batch Size 异构)
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

        # 2. 构建特定结构的模型
        model = build_model(model_name, num_features, model_args)
        logger.info("模型参数量: {:,}", sum(p.numel() for p in model.parameters()))

        # 3. 构建特定的优化器与调度器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=train_args["lr"], 
            weight_decay=train_args["weight_decay"]
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        # 4. 执行训练
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

        # 5. 持久化记录
        history_path = out_dir / f"{model_name}_history.json"
        history_path.write_text(json.dumps(history, indent=2))
        logger.info("模型权重与训练历史已保存至: {}", out_dir)


if __name__ == "__main__":
    main()