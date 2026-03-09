"""V5-beta: 验证三套模型的输出张量形状 (LSTM / Transformer / LSTM-mTrans-MLP)。"""
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.models import LSTMModel, LSTMmTransMLPModel, TransformerModel  # noqa: E402

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

# 论文参数: 仅收盘价 (input_dim=1), seq_len=60
BATCH, SEQ_LEN, FEATURES = 64, 60, 1
LSTM_HIDDEN, NUM_HEADS, HEAD_DIM, PRED_LEN = 60, 5, 120, 1


def main() -> None:
    x = torch.randn(BATCH, SEQ_LEN, FEATURES)
    logger.info("输入: X {}", tuple(x.shape))

    models = {
        "LSTM": LSTMModel(
            input_dim=FEATURES, hidden_dim=LSTM_HIDDEN,
            num_layers=2, pred_len=PRED_LEN, dropout=0.1,
        ),
        "Transformer": TransformerModel(
            input_dim=FEATURES, d_model=LSTM_HIDDEN,
            num_heads=NUM_HEADS, num_layers=2,
            ffn_dim=128, pred_len=PRED_LEN, dropout=0.15,
        ),
        "LSTM-mTrans-MLP": LSTMmTransMLPModel(
            input_dim=FEATURES, lstm_hidden=LSTM_HIDDEN,
            num_lstm_layers=2, num_heads=NUM_HEADS,
            head_dim=HEAD_DIM, ffn_mid=5, pred_len=PRED_LEN,
            lstm_dropout=0.1, trans_dropout=0.15, mlp_dropout=0.1,
        ),
    }

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred, attn = model(x)

        attn_info = f"attn: {tuple(attn.shape)}" if attn is not None else "attn: None"
        logger.info("{:<18s} pred: {}  {}", name, tuple(pred.shape), attn_info)

        # 断言形状正确
        assert pred.shape == (BATCH, PRED_LEN), f"{name} pred shape mismatch"
        if attn is not None:
            # LSTM-mTrans-MLP: attn (B, lstm_hidden, lstm_hidden)
            # Transformer: attn (B, SEQ_LEN, SEQ_LEN)
            assert attn.dim() == 3, f"{name} attn should be 3D"

    params = {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()}
    for name, cnt in params.items():
        logger.info("{:<18s} params: {:,}", name, cnt)

    logger.info("全部断言通过 ✓")


if __name__ == "__main__":
    main()
