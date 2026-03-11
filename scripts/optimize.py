"""基于 Optuna 的四架构公平消融搜索脚本。

约束原则:
- 各模型拥有对等的搜索深度 (Trials)。
- 彻底隔离各模型的超参数偏见。
- 引入 ReduceLROnPlateau 解决 Transformer 对固定学习率敏感的问题。
"""
import gc
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR # 显式引入

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_dataloaders
from src.engine.trainer import train_model
from src.models.networks import (
    LSTMModel,
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

# 统一基础设置
SEQ_LEN = 60
PRED_LEN = 1
EPOCHS = 100
PATIENCE = 15  # 早停耐心值
N_TRIALS = 50  # 每个模型搜索 50 组超参数 (可根据算力调整)


def get_model_and_space(trial: optuna.Trial, model_name: str, num_features: int) -> nn.Module:
    """定义并返回与特定架构匹配的超参空间与实例。"""
    
    # 共享的基础超参搜索空间 (结构对等)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    # 注意力头数要求被 hidden_dim 整除，故 hidden_dim 限制在能被 2,4,8 整除的值
    hidden_dim = trial.suggest_categorical("hidden_dim",[32, 64, 128]) 

    if model_name == "LSTM":
        num_layers = trial.suggest_int("num_layers", 1, 3)
        return LSTMModel(
            input_dim=num_features, hidden_dim=hidden_dim,
            num_layers=num_layers, pred_len=PRED_LEN, dropout=dropout
        )

    elif model_name == "Transformer":
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        ffn_dim = trial.suggest_categorical("ffn_dim", [64, 128, 256])
        return TransformerModel(
            input_dim=num_features, d_model=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            ffn_dim=ffn_dim, pred_len=PRED_LEN, dropout=dropout
        )

    elif model_name == "LSTM_Transformer":
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)
        num_trans_layers = trial.suggest_int("num_trans_layers", 1, 2)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        ffn_dim = trial.suggest_categorical("ffn_dim",[64, 128, 256])
        return LSTMTransformerModel(
            input_dim=num_features, hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers, num_heads=num_heads,
            num_transformer_layers=num_trans_layers, ffn_dim=ffn_dim,
            pred_len=PRED_LEN, dropout=dropout
        )

    elif model_name == "Parallel_LSTM_Transformer":
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 2)
        num_trans_layers = trial.suggest_int("num_trans_layers", 1, 2)
        num_heads = trial.suggest_categorical("num_heads",[2, 4, 8])
        ffn_dim = trial.suggest_categorical("ffn_dim",[64, 128, 256])
        return ParallelLSTMTransformerModel(
            input_dim=num_features, hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers, num_heads=num_heads,
            num_transformer_layers=num_trans_layers, ffn_dim=ffn_dim,
            pred_len=PRED_LEN, dropout=dropout
        )
    else:
        raise ValueError(f"未知模型: {model_name}")


def create_objective(model_name: str, df_values: np.ndarray, columns: list[str], device: torch.device) -> Callable:
    """闭包: 生成供 Optuna 调用的 objective 函数。"""
    
    def objective(trial: optuna.Trial) -> float:
        # 1. 训练策略超参搜索
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # 2. 数据准备 (随 Batch Size 动态生成 Dataloader)
        train_loader, val_loader, _, _ = get_dataloaders(
            df_values=df_values,
            columns=columns,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            batch_size=batch_size,
            target_col="close",
            train_ratio=0.72,
            val_ratio=0.10,
        )

        # 3. 模型实例化
        model = get_model_and_space(trial, model_name, len(columns))
        
        # 4. 优化器与自适应调度器 (极其重要: 防止 Transformer 梯度爆炸/不收敛)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        criterion = nn.MSELoss()

        # 5. 执行训练
        try:
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=EPOCHS,
                patience=PATIENCE,
                save_path=PROJECT_ROOT / "models" / f"optuna_{model_name}_temp.pth",
                scheduler=scheduler,
                max_grad_norm=1.0,
                trial=trial,  # 传入 trial 进行剪枝
            )
            # 记录本轮最优验证集 Loss
            best_val_loss = min(history["val_loss"])
            
        except RuntimeError as e:
            # 捕获由于超参数过大导致的显存溢出 (OOM)，直接拒绝该组参数
            if "out of memory" in str(e):
                logger.warning(f"OOM 发生于 {model_name} (Trial {trial.number})，舍弃该组参数。")
                raise optuna.exceptions.TrialPruned()
            else:
                raise e
        finally:
            # 严谨的显存回收
            del model, optimizer, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return best_val_loss

    return objective


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("HPO 平台启动 | 训练设备: {}", device)

    # 仅加载一次数据到内存
    csv_path = PROJECT_ROOT / "data" / "csi300_features.csv"
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df_values = df.values
    columns = df.columns.tolist()

    models_to_tune =[
        "LSTM",
        "Transformer",
        "LSTM_Transformer",
        "Parallel_LSTM_Transformer"
    ]

    opt_dir = PROJECT_ROOT / "optuna_results"
    opt_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model_name in models_to_tune:
        logger.info("=" * 60)
        logger.info("🔥 开始全局搜索: {} ({} Trials)", model_name, N_TRIALS)
        
        # 使用中位值停止剪枝器 (Median Pruner) 提升搜索效率
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(direction="minimize", study_name=model_name, pruner=pruner)
        
        objective_fn = create_objective(model_name, df_values, columns, device)
        study.optimize(objective_fn, n_trials=N_TRIALS, gc_after_trial=True)

        logger.info("✅ {} 搜索完成 | Best Val Loss: {:.6f}", model_name, study.best_value)
        logger.info("最优超参数: {}", study.best_params)
        
        results[model_name] = {
            "best_loss": study.best_value,
            "best_params": study.best_params
        }
        
        # 持久化该模型的最优参数
        df_study = study.trials_dataframe()
        df_study.to_csv(opt_dir / f"{model_name}_trials.csv", index=False)

    # 输出全局终极对比报告
    logger.info("=" * 60)
    logger.info("🏆 公平对比消融实验最终结果 🏆")
    for m, res in results.items():
        logger.info(f"{m:<25} | Best MSE: {res['best_loss']:.6f}")


if __name__ == "__main__":
    main()