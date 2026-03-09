"""LSTM / Transformer / LSTM-Transformer / Parallel 四套时序分类网络。

所有模型统一签名:
    forward(x: Tensor) -> tuple[Tensor, Tensor | dict | None]
    返回 (logits, extra_info)。logits 形状 (B, num_classes)。
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
    """纯 LSTM 基线。(B, S, F) -> LSTM -> 取 last step -> FC -> (B, num_classes)"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]), None


# ── Baseline 2: 纯 Transformer ───────────────────────────────────
class TransformerModel(nn.Module):
    """纯 Transformer 基线。(B, S, F) -> Proj -> PosEnc -> N层Encoder -> FC -> (B, num_classes)"""

    def __init__(self, input_dim: int, d_model: int = 64, num_heads: int = 4,
                 num_layers: int = 2, ffn_dim: int = 256,
                 num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, ffn_dim, dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pe(self.proj(x))
        x, attn_w = self.encoder(x)
        return self.fc(x[:, -1, :]), attn_w


# ── 核心融合: LSTM + Transformer ─────────────────────────────────
class LSTMTransformerModel(nn.Module):
    """LSTM 提取局部时序 → LayerNorm → 位置编码 → N层 Transformer → 门控融合 → 输出。

    V2: LSTM 与 Transformer 之间插入 LayerNorm，消除方差偏移。
    V3: LSTM 最后时间步输出与 Transformer 最后时间步输出拼接（跳跃连接）。
    V4: 融合层升级为门控瓶颈网络 (Linear→ReLU→Dropout→Linear)，
        实现对双分支特征的非线性过滤与动态加权。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_lstm_layers: int = 2,
                 num_heads: int = 4, num_transformer_layers: int = 2,
                 ffn_dim: int = 256, num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim)  # V2: 稳定 LSTM→Transformer 方差
        self.pe = PositionalEncoding(hidden_dim, dropout=dropout)
        self.encoder = TransformerEncoder(
            hidden_dim, num_heads, num_transformer_layers, ffn_dim, dropout,
        )
        # V4: 门控融合瓶颈网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. LSTM 提取特征
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln(lstm_out)
        lstm_last = lstm_out[:, -1, :]
        
        # 2. Transformer 分支 (⚠️ 去掉 self.pe(lstm_out) )
        # 直接用 LSTM 的输出送入 Transformer，因为已经有足够的时间信息
        trans_out, attn_w = self.encoder(lstm_out) 
        trans_last = trans_out[:, -1, :]
        
        # 3. 门控融合
        fused = torch.cat([lstm_last, trans_last], dim=-1)
        return self.fusion(fused), attn_w

class ParallelLSTMTransformerModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_lstm_layers: int = 2,
                 num_heads: int = 4, num_transformer_layers: int = 2,
                 ffn_dim: int = 256, num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        
        # === 左塔：LSTM ===
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0)
        self.ln_lstm = nn.LayerNorm(hidden_dim)
        
        # === 右塔：Transformer ===
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim, dropout=dropout)
        self.encoder = TransformerEncoder(hidden_dim, num_heads, num_transformer_layers, ffn_dim, dropout)
        self.ln_trans = nn.LayerNorm(hidden_dim)

        # === 动态门控融合机制 (Dynamic Gating Mechanism) ===
        # 计算两个塔输出的权重
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), # 输出 128 维（针对每一个特征）
            nn.Sigmoid() # 用 Sigmoid，不要用 Softmax，允许某些特征同时高亮或同时抑制
        )
        
        # 最终分类头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 左塔: LSTM
        out_lstm, _ = self.lstm(x)
        feat_lstm = self.ln_lstm(out_lstm[:, -1, :])

        # 右塔: Transformer + Global Average Pooling
        out_trans = self.pe(self.proj(x))
        out_trans, attn_w = self.encoder(out_trans)
        feat_trans = self.ln_trans(out_trans.mean(dim=1))

        # 动态门控融合
        concat_feats = torch.cat([feat_lstm, feat_trans], dim=-1)
        gate_weights = self.gate(concat_feats)
        gate_lstm, gate_trans = gate_weights.chunk(2, dim=-1)
        fused = gate_lstm * feat_lstm + gate_trans * feat_trans

        pred = self.predictor(fused)
        meta = {"attn_w": attn_w, "gate_lstm": gate_lstm, "gate_trans": gate_trans}
        return pred, meta