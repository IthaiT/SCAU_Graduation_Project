import pandas as pd
from pathlib import Path
from loguru import logger

# Define file paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_CSV = DATA_DIR / "csi300_features_advanced.csv"
OUTPUT_CSV = DATA_DIR / "final_data.csv"
REPORT_FILE = DATA_DIR / "csi300_features_report.txt"

def check_and_clean_csv():
    logger.info(f"Loading CSV file: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        logger.error(f"File not found: {INPUT_CSV}")
        return

    # Load the CSV file
    df = pd.read_csv(INPUT_CSV)

    # Check for empty columns
    empty_columns = [col for col in df.columns if df[col].isnull().all()]
    logger.info(f"Found {len(empty_columns)} empty columns.")

    # Check for empty or zero-only columns
    empty_or_zero_columns = [col for col in df.columns if df[col].isnull().all() or (df[col] == 0).all()]
    logger.info(f"Found {len(empty_or_zero_columns)} empty or zero-only columns.")

    # Generate report
    with REPORT_FILE.open("w", encoding="utf-8") as report:
        report.write("CSV File Report\n")
        report.write(f"Total columns: {len(df.columns)}\n")
        report.write(f"Empty columns: {len(empty_columns)}\n")
        if empty_columns:
            report.write("Empty columns list:\n")
            for col in empty_columns:
                report.write(f"- {col}\n")
        report.write(f"Empty or zero-only columns: {len(empty_or_zero_columns)}\n")
        if empty_or_zero_columns:
            report.write("Empty or zero-only columns list:\n")
            for col in empty_or_zero_columns:
                report.write(f"- {col}\n")

    # Drop empty or zero-only columns
    if empty_or_zero_columns:
        df.drop(columns=empty_or_zero_columns, inplace=True)
        logger.info(f"Dropped {len(empty_or_zero_columns)} empty or zero-only columns.")

    # Log the number of rows remaining after cleaning
    remaining_rows = len(df)
    logger.info(f"Remaining rows after cleaning: {remaining_rows}")

    # Print all remaining columns to the console and their count
    remaining_columns = df.columns.tolist()
    logger.info("Remaining columns:")
    logger.info(remaining_columns)
    logger.info(f"Total remaining columns: {len(remaining_columns)}")

    # Save the cleaned CSV
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Cleaned CSV saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    check_and_clean_csv()