"""验证三套模型的输出张量形状。"""
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.models import LSTMModel, LSTMTransformerModel, TransformerModel  # noqa: E402

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

BATCH, SEQ_LEN, FEATURES = 64, 30, 17
HIDDEN_DIM, NUM_HEADS, PRED_LEN = 64, 4, 1


def main() -> None:
    x = torch.randn(BATCH, SEQ_LEN, FEATURES)
    logger.info("输入: X {}", tuple(x.shape))

    models = {
        "LSTM": LSTMModel(input_dim=FEATURES, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN),
        "Transformer": TransformerModel(input_dim=FEATURES, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
        "LSTM+Transformer": LSTMTransformerModel(input_dim=FEATURES, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, pred_len=PRED_LEN),
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
            assert attn.shape == (BATCH, SEQ_LEN, SEQ_LEN), f"{name} attn shape mismatch"

    params = {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()}
    for name, cnt in params.items():
        logger.info("{:<18s} params: {:,}", name, cnt)

    logger.info("全部断言通过 ✓")


if __name__ == "__main__":
    main()
