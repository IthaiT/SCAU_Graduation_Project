"""训练引擎: 早停 + 训练/验证循环。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader


# ── 早停 ──────────────────────────────────────────────────────────
class EarlyStopping:
    """验证 loss 连续 patience 个 epoch 无改善则触发早停，每次创新低保存权重。"""

    def __init__(self, patience: int = 10, save_path: str | Path = "models/best_model.pth") -> None:
        self.patience = patience
        self.save_path = Path(save_path)
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.triggered: bool = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """返回 True 表示应停止训练。"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self._save(model)
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
            logger.info("早停触发: 连续 {} 轮无改善 (best={:.6f})", self.patience, self.best_loss)
            return True
        return False

    def _save(self, model: nn.Module) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        logger.debug("权重已保存 → {}", self.save_path)


# ── 单 epoch 辅助 ─────────────────────────────────────────────────
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """跑一个 epoch，optimizer 非空则训练模式，否则验证模式。返回平均 loss。"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)  # 丢弃 attn_weights
            loss = criterion(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ── 主训练入口 ────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    patience: int = 10,
    save_path: str | Path = "models/best_model.pth",
) -> dict[str, list[float]]:
    """完整训练循环，返回 history = {train_loss: [...], val_loss: [...]}。"""
    model.to(device)
    es = EarlyStopping(patience=patience, save_path=save_path)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch {:>3d}/{} | Train Loss: {:.6f} | Val Loss: {:.6f}",
            epoch, epochs, train_loss, val_loss,
        )

        if es.step(val_loss, model):
            break

    logger.info("训练结束: {} 轮, best val_loss={:.6f}", len(history["val_loss"]), es.best_loss)
    return history
