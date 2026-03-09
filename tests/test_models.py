"""验证四套模型的输出张量形状 (三分类)。"""
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.models import (  # noqa: E402
    LSTMModel,
    LSTMTransformerModel,
    ParallelLSTMTransformerModel,
    TransformerModel,
)

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}")

BATCH, SEQ_LEN, FEATURES = 64, 30, 17
HIDDEN_DIM, NUM_HEADS, NUM_CLASSES = 64, 4, 3


def main() -> None:
    x = torch.randn(BATCH, SEQ_LEN, FEATURES)
    logger.info("输入: X {}", tuple(x.shape))

    models = {
        "LSTM": LSTMModel(input_dim=FEATURES, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES),
        "Transformer": TransformerModel(input_dim=FEATURES, d_model=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES),
        "LSTM-Transformer": LSTMTransformerModel(input_dim=FEATURES, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES),
        "Parallel-LSTM-Transformer": ParallelLSTMTransformerModel(input_dim=FEATURES, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_classes=NUM_CLASSES),
    }

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred, extra = model(x)

        extra_info = "None"
        if isinstance(extra, dict):
            extra_info = f"attn: {tuple(extra['attn_w'].shape)}"
        elif extra is not None:
            extra_info = f"attn: {tuple(extra.shape)}"
        logger.info("{:<28s} pred: {}  {}", name, tuple(pred.shape), extra_info)

        assert pred.shape == (BATCH, NUM_CLASSES), f"{name} pred shape mismatch"

    params = {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()}
    for name, cnt in params.items():
        logger.info("{:<28s} params: {:,}", name, cnt)

    logger.info("全部断言通过 ✓")


if __name__ == "__main__":
    main()
