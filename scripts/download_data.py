"""下载沪深300日线数据的入口脚本。"""
import logging
import sys
from pathlib import Path

# 项目根目录加入 sys.path，确保 src 包可导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import fetch_and_clean_data, save_csv  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    out_path = PROJECT_ROOT / "data" / "csi300_raw.csv"

    df = fetch_and_clean_data(
        symbol="sh000300",
        start_date="20060101",
        end_date="20260101",
    )

    save_csv(df, out_path)
    logger.info("预览:\n%s", df.head())


if __name__ == "__main__":
    main()
