"""LSTM / Transformer / LSTM-Transformer 三套时序预测网络。

所有模型统一签名:
    forward(x: Tensor) -> tuple[Tensor, Tensor | None]
    返回 (predictions, attention_weights)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── 共享组件 ──────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """正弦位置编码。(B, S, D) -> (B, S, D)"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _EncoderLayer(nn.Module):
    """单层 Transformer Encoder (MHA + FFN + LayerNorm)，暴露 attn_weights。"""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.mha(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, attn_w


class TransformerEncoder(nn.Module):
    """N 层 Transformer Encoder，返回最后一层 attention weights。"""

    def __init__(self, d_model: int, num_heads: int, num_layers: int = 2,
                 ffn_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _EncoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_w = torch.empty(0)
        for layer in self.layers:
            x, attn_w = layer(x)
        return x, attn_w


# ── Baseline 1: 纯 LSTM ──────────────────────────────────────────
class LSTMModel(nn.Module):
    """纯 LSTM 基线。(B, S, F) -> LSTM -> 取 last step -> FC -> (B, pred_len)"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 pred_len: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]), None


# ── Baseline 2: 纯 Transformer ───────────────────────────────────
class TransformerModel(nn.Module):
    """纯 Transformer 基线。(B, S, F) -> Proj -> PosEnc -> N层Encoder -> FC -> (B, pred_len)"""

    def __init__(self, input_dim: int, d_model: int = 64, num_heads: int = 4,
                 num_layers: int = 2, ffn_dim: int = 256,
                 pred_len: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, ffn_dim, dropout)
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pe(self.proj(x))
        x, attn_w = self.encoder(x)
        return self.fc(x[:, -1, :]), attn_w


# ── 核心融合: LSTM + Transformer ─────────────────────────────────
class LSTMTransformerModel(nn.Module):
    """LSTM 提取局部时序 → 位置编码 → N层 Transformer 捕获全局依赖 → FC 输出。"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_lstm_layers: int = 2,
                 num_heads: int = 4, num_transformer_layers: int = 2,
                 ffn_dim: int = 256, pred_len: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.pe = PositionalEncoding(hidden_dim, dropout=dropout)
        self.encoder = TransformerEncoder(
            hidden_dim, num_heads, num_transformer_layers, ffn_dim, dropout,
        )
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.lstm(x)
        x = self.pe(x)
        x, attn_w = self.encoder(x)
        return self.fc(x[:, -1, :]), attn_w
