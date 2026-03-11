"""训练引擎: 早停 + 学习率调度 + 训练/验证循环 + Optuna剪枝支持。"""
from __future__ import annotations

from pathlib import Path

import optuna
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader


# ── 早停模块 (保持不变) ───────────────────────────────────────────
class EarlyStopping:
    """验证 loss 连续 patience 个 epoch 无改善则触发早停，每次创新低保存权重。"""

    def __init__(self, patience: int = 10, save_path: str | Path = "models/best_model.pth") -> None:
        self.patience = patience
        self.save_path = Path(save_path)
        self.best_loss: float = float("inf")
        self.counter: int = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self._save(model)
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            logger.debug("早停触发: 连续 {} 轮无改善 (best={:.6f})", self.patience, self.best_loss)
            return True
        return False

    def _save(self, model: nn.Module) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.save_path)


# ── 单 epoch 辅助 (保持不变) ──────────────────────────────────────
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 0.0,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)
            loss = criterion(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ── 主训练入口 (重点重构: 增加 trial 参数) ────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
    patience: int = 10,
    save_path: str | Path = "models/best_model.pth",
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    max_grad_norm: float = 1.0,
    trial: optuna.Trial | None = None,  # 新增: Optuna Trial
) -> dict[str, list[float]]:
    """完整训练循环，支持 Optuna 剪枝。"""
    model.to(device)
    es = EarlyStopping(patience=patience, save_path=save_path)
    history: dict[str, list[float]] = {"train_loss":[], "val_loss":[]}

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer, max_grad_norm)
        val_loss = _run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # 动态学习率调度 (核心: 拯救 Transformer 的关键)
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Optuna 剪枝通信
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                logger.debug("Trial 被剪枝: Epoch {} 表现不佳", epoch)
                raise optuna.exceptions.TrialPruned()

        # 仅在非 Trial 搜索阶段打印详细日志，避免终端信息爆炸
        if trial is None:
            lr = optimizer.param_groups[0]["lr"]
            logger.info("Epoch {:>3d}/{} | Train: {:.6f} | Val: {:.6f} | LR: {:.2e}", 
                        epoch, epochs, train_loss, val_loss, lr)

        if es.step(val_loss, model):
            break

    return history