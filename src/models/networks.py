"""V5-beta: 五套时序预测网络。

Baseline 1: LSTM — 纯 LSTM 基线
Baseline 2: Transformer — 纯 Transformer 基线
Baseline 3: Serial LSTM-Transformer — 串行 LSTM→Transformer (V1-V4 自研迭代)
Baseline 4: Parallel LSTM-Transformer — 并行双塔 + 动态门控 (V5 自研)
核心模型: LSTM-mTrans-MLP — 论文 Kabir et al. (Sci 2025, 7, 7) 架构

所有模型统一签名:
    forward(x: Tensor) -> tuple[Tensor, Tensor | None]
    返回 (predictions, attention_weights 或 None)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 共享组件 ──────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """正弦位置编码 (仅 Transformer baseline 使用)。(B, S, D) -> (B, S, D)"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.15) -> None:
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


# ── Baseline 1: 纯 LSTM ──────────────────────────────────────────
class LSTMModel(nn.Module):
    """纯 LSTM 基线 (匹配论文 LSTM 部分的参数)。

    Input (B, 60, 1) -> LSTM×2 -> 取 last step -> FC -> (B, 1)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 60,
                 num_layers: int = 2, pred_len: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        out, _ = self.lstm(x)  # (B, S, hidden_dim)
        out = self.dropout(out[:, -1, :])  # (B, hidden_dim)
        return self.fc(out), None


# ── Baseline 2: 纯 Transformer ───────────────────────────────────
class _EncoderLayer(nn.Module):
    """标准 Transformer Encoder (post-norm)，暴露 attn_weights。"""

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


class TransformerModel(nn.Module):
    """纯 Transformer 基线。

    Input (B, 60, 1) -> Proj -> PosEnc -> N层Encoder -> FC -> (B, 1)
    """

    def __init__(self, input_dim: int = 1, d_model: int = 60,
                 num_heads: int = 5, num_layers: int = 2,
                 ffn_dim: int = 128, pred_len: int = 1,
                 dropout: float = 0.15) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            _EncoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pe(self.proj(x))
        attn_w = torch.empty(0)
        for layer in self.layers:
            x, attn_w = layer(x)
        return self.fc(x[:, -1, :]), attn_w


# ── Baseline 3: 串行 LSTM-Trans (V10 自研版) ────────────────────
class LSTMTransformerModel(nn.Module):
    """串行 LSTM-Trans V10: LSTM 骨干 + 单层 Pre-Norm 注意力 + ReZero。

    借鉴 LSTM-mTrans-MLP 的极简设计哲学:
        - LSTM 作为主干网络，承担绝大部分建模
        - 仅用单层 Pre-Norm 自注意力做轻量级精炼
        - ReZero (α=0 初始化) 确保训练初期 = 纯 LSTM
        - 注意力只能渐进增益，永远不会破坏 LSTM 特征
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_lstm_layers: int = 2, num_heads: int = 4,
                 num_transformer_layers: int = 2, ffn_dim: int = 256,
                 pred_len: int = 1, dropout: float = 0.2,
                 seq_len: int = 60) -> None:
        super().__init__()

        # LSTM 骨干 (与 LSTM baseline 同级)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # 单层 Pre-Norm 自注意力精炼 (仅 1 层，借鉴 mTrans 设计)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

        # ReZero: α 初始化为 0 → 模型初始 = 纯 LSTM
        self.alpha = nn.Parameter(torch.zeros(1))

        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # LSTM 骨干
        lstm_out, _ = self.lstm(x)  # (B, S, H)

        # 单层 Pre-Norm 注意力精炼 + ReZero
        normed = self.norm1(lstm_out)
        attn_out, attn_w = self.attn(
            normed, normed, normed,
            need_weights=True, average_attn_weights=True,
        )
        x = lstm_out + self.alpha * self.drop1(attn_out)

        normed = self.norm2(x)
        x = x + self.alpha * self.ffn(normed)

        return self.fc(x[:, -1, :]), attn_w


# ── Baseline 4: 并行双塔 LSTM-Transformer (V12 自研版) ───────────
class ParallelLSTMTransformerModel(nn.Module):
    """并行双路读出 V12: 共享 LSTM 骨干 + 局部/全局双路 + 门控。

    核心思想:
        - 共享 LSTM 骨干处理原始输入 (避免弱右塔问题)
        - 路径 A: 直接取 last step (局部/近期特征)
        - 路径 B: 可学习 query 对 LSTM 隐状态做 cross-attention 池化 (全局/历史)
        - 约束门控逐维度决定信赖哪个视角
        - 极其轻量 (~78K 参数, 1.6x LSTM), 稳定可训练
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_lstm_layers: int = 2, num_heads: int = 4,
                 num_transformer_layers: int = 2, ffn_dim: int = 256,
                 pred_len: int = 1, dropout: float = 0.2) -> None:
        super().__init__()

        # 共享 LSTM 骨干
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_lstm_layers,
            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.ln_lstm = nn.LayerNorm(hidden_dim)

        # 路径 B: Cross-Attention 池化 (全局/历史读出)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ln_pool = nn.LayerNorm(hidden_dim)

        # 约束门控 + 融合
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_fused = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 共享 LSTM 骨干
        lstm_out, _ = self.lstm(x)                          # (B, S, H)
        lstm_out = self.ln_lstm(lstm_out)

        # 路径 A: 直接取 last step (局部/近期)
        feat_local = lstm_out[:, -1, :]                      # (B, H)

        # 路径 B: cross-attention 池化 (全局/历史)
        query = self.query.expand(x.size(0), -1, -1)        # (B, 1, H)
        pooled, attn_w = self.cross_attn(
            query, lstm_out, lstm_out,
            need_weights=True, average_attn_weights=True,
        )
        feat_global = self.ln_pool(pooled.squeeze(1))        # (B, H)

        # 约束门控融合
        gate = torch.sigmoid(self.gate_proj(
            torch.cat([feat_local, feat_global], dim=-1),
        ))
        fused = gate * feat_global + (1 - gate) * feat_local  # (B, H)
        fused = self.ln_fused(fused)

        return self.fc(fused), attn_w


# ══════════════════════════════════════════════════════════════════
# 论文核心模型: LSTM-mTrans-MLP
# ══════════════════════════════════════════════════════════════════

class _CustomMultiHeadAttention(nn.Module):
    """自定义多头注意力 (兼容 d_model=1 + num_heads>1 的论文设定)。

    论文参数: num_heads=5, head_dim=120, d_model=1。
    等价于 Keras MultiHeadAttention(num_heads=5, key_dim=120)。
    内部将 d_model 投射到 num_heads * head_dim，计算缩放点积注意力后投射回 d_model。
    """

    def __init__(self, d_model: int = 1, num_heads: int = 5,
                 head_dim: int = 120, dropout: float = 0.15) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        total_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(d_model, total_dim)
        self.wk = nn.Linear(d_model, total_dim)
        self.wv = nn.Linear(d_model, total_dim)
        self.wo = nn.Linear(total_dim, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, S, d_model) -> (B, S, d_model), attn_weights: (B, S, S)"""
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.wq(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        k = self.wk(x).view(B, S, H, D).transpose(1, 2)
        v = self.wv(x).view(B, S, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_drop(attn_w)

        out = torch.matmul(attn_w, v)  # (B, H, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, H * D)  # (B, S, total_dim)
        out = self.wo(out)  # (B, S, d_model)

        # 返回平均 attention weights (跨头平均)
        avg_attn_w = attn_w.mean(dim=1)  # (B, S, S)
        return out, avg_attn_w


class _mTransformerBlock(nn.Module):
    """论文中的 Modified Transformer Block (单层, Pre-Norm)。

    结构:
        1. LayerNorm → MHA → Dropout → Residual (+ input)
        2. LayerNorm → FFN(Dense5→ReLU→Drop→Dense1) → Residual (+ step1)

    论文关键修改:
        - Pre-normalization (LayerNorm 在 attention/FFN 之前)
        - 无位置编码
        - 无 input embedding
        - 仅 encoder（decoder 被 MLP 取代）
    """

    def __init__(self, d_model: int = 1, num_heads: int = 5,
                 head_dim: int = 120, ffn_mid: int = 5,
                 dropout: float = 0.15) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = _CustomMultiHeadAttention(d_model, num_heads, head_dim, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mid, d_model),
        )

        # ReZero init: 让 mTransformer 初始为恒等映射，
        # 避免 d_model=1 时 LayerNorm 退化导致的随机偏移干扰 MLP 优化。
        nn.init.zeros_(self.attn.wo.weight)
        nn.init.zeros_(self.attn.wo.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-Norm → MHA → Dropout → Residual
        normed = self.norm1(x)
        attn_out, attn_w = self.attn(normed)
        x = x + self.drop1(attn_out)

        # Pre-Norm → FFN → Residual
        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x, attn_w


class LSTMmTransMLPModel(nn.Module):
    """论文核心模型: LSTM-mTrans-MLP。

    严格按照 Kabir et al. (2025) 的架构实现:
        Block 1 — LSTM: 2层 LSTM(units=60) → Reshape → Dropout
        Block 2 — mTrans: Pre-Norm + MHA(heads=5, head_dim=120) + FFN(5) + 残差
        Block 3 — MLP: GAP → Dropout → Dense(30,ReLU) → Dense(1,linear)

    Training: MSE loss, Adam(lr=0.001), batch_size=1
    """

    def __init__(self, input_dim: int = 1, lstm_hidden: int = 60,
                 num_lstm_layers: int = 2, num_heads: int = 5,
                 head_dim: int = 120, ffn_mid: int = 5,
                 pred_len: int = 1,
                 lstm_dropout: float = 0.1,
                 trans_dropout: float = 0.15,
                 mlp_dropout: float = 0.1) -> None:
        super().__init__()

        # ── Block 1: LSTM ──
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        # ── Block 2: Modified Transformer (单层 encoder, pre-norm) ──
        # d_model = 1 (LSTM hidden reshape 为 (seq_len, 1))
        self.mtrans = _mTransformerBlock(
            d_model=1, num_heads=num_heads,
            head_dim=head_dim, ffn_mid=ffn_mid,
            dropout=trans_dropout,
        )

        # ── Block 3: MLP ──
        # GAP(channels_first) 将 (B, lstm_hidden, 1) -> (B, lstm_hidden)
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden, 30),
            nn.ReLU(),
            nn.Linear(30, pred_len),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, seq_len, input_dim)  — 论文中 seq_len=60, input_dim=1
        返回: (predictions (B, 1), attn_weights (B, S, S))
        """
        # Block 1: LSTM
        # LSTM layer1 (return_sequences=True) + layer2 (return_sequences=False)
        # PyTorch LSTM 返回全部时间步和最后隐状态
        lstm_out, _ = self.lstm(x)  # (B, seq_len, lstm_hidden)
        # 取最后时间步 (等效 return_sequences=False)
        lstm_final = lstm_out[:, -1, :]  # (B, lstm_hidden)
        # Reshape: (B, 60) -> (B, 60, 1)
        lstm_reshaped = self.lstm_dropout(lstm_final).unsqueeze(-1)  # (B, lstm_hidden, 1)

        # Block 2: mTransformer
        # 输入 (B, lstm_hidden, 1) 即 (B, 60, 1)
        trans_out, attn_w = self.mtrans(lstm_reshaped)  # (B, 60, 1)

        # Block 3: MLP
        # GlobalAveragePooling1D(channels_first): (B, 60, 1) → avg over last dim → (B, 60)
        gap_out = trans_out.mean(dim=-1)  # (B, lstm_hidden)
        gap_out = self.mlp_dropout(gap_out)
        pred = self.mlp(gap_out)  # (B, pred_len)

        return pred, attn_w