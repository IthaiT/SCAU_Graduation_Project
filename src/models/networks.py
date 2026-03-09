"""LSTM / Transformer / LSTM-Transformer / Parallel — binary classification.

Unified signature:
    forward(x: Tensor) -> tuple[Tensor, Tensor | dict | None]
    Returns (logits, extra). logits shape: (B, num_classes).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Shared components ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
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


# ── Baseline 1: LSTM ─────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]), None


# ── Baseline 2: Transformer ──────────────────────────────────────
class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, num_heads: int = 4,
                 num_layers: int = 2, ffn_dim: int = 256,
                 num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, ffn_dim, dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pe(self.proj(x))
        x, attn_w = self.encoder(x)
        return self.fc(x[:, -1, :]), attn_w


# ── LSTM + Transformer (serial fusion) ───────────────────────────
class LSTMTransformerModel(nn.Module):
    """LSTM extracts local temporal features → LayerNorm → Transformer encodes
    global dependencies → skip-connection fusion → classification head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_lstm_layers: int = 2,
                 num_heads: int = 4, num_transformer_layers: int = 2,
                 ffn_dim: int = 256, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.pe = PositionalEncoding(hidden_dim, dropout=dropout)
        self.encoder = TransformerEncoder(
            hidden_dim, num_heads, num_transformer_layers, ffn_dim, dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln(lstm_out)
        lstm_last = lstm_out[:, -1, :]

        trans_out, attn_w = self.encoder(lstm_out)
        trans_last = trans_out[:, -1, :]

        fused = torch.cat([lstm_last, trans_last], dim=-1)
        return self.fusion(fused), attn_w


# ── Parallel LSTM + Transformer (complementary gating) ───────────
class ParallelLSTMTransformerModel(nn.Module):
    """Two parallel towers fused via complementary gating: g * lstm + (1-g) * trans.

    Gate uses ReLU hidden layer to avoid vanishing gradients from deep sigmoid chains.
    Complementary structure ensures proper routing without dead neurons.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_lstm_layers: int = 2,
                 num_heads: int = 4, num_transformer_layers: int = 2,
                 ffn_dim: int = 256, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()

        # Left tower: LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0)
        self.ln_lstm = nn.LayerNorm(hidden_dim)

        # Right tower: Transformer
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim, dropout=dropout)
        self.encoder = TransformerEncoder(hidden_dim, num_heads, num_transformer_layers, ffn_dim, dropout)
        self.ln_trans = nn.LayerNorm(hidden_dim)

        # Complementary gate: g ∈ [0,1]^hidden_dim
        # fused = g * feat_lstm + (1 - g) * feat_trans
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Left tower
        out_lstm, _ = self.lstm(x)
        feat_lstm = self.ln_lstm(out_lstm[:, -1, :])

        # Right tower
        out_trans = self.pe(self.proj(x))
        out_trans, attn_w = self.encoder(out_trans)
        feat_trans = self.ln_trans(out_trans.mean(dim=1))

        # Complementary gating
        concat = torch.cat([feat_lstm, feat_trans], dim=-1)
        gate = self.gate_net(concat)
        fused = gate * feat_lstm + (1.0 - gate) * feat_trans

        logits = self.predictor(fused)
        meta = {"attn_w": attn_w, "gate_lstm": gate, "gate_trans": 1.0 - gate}
        return logits, meta