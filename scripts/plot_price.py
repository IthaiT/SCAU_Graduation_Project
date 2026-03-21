import pandas as pd
import matplotlib.pyplot as plt


def plot_price(csv_path, output_path, date_col="date", price_col="close"):
    """
    绘制价格走势图并保存高质量图片

    参数:
    csv_path: CSV文件路径
    output_path: 输出图片路径 (pdf 或 png)
    date_col: 日期列名
    price_col: 价格列名
    """

    # 读取数据
    df = pd.read_csv(csv_path)

    # 日期转换
    df[date_col] = pd.to_datetime(df[date_col])

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(df[date_col], df[price_col], linewidth=1.5)

    plt.xlabel("date")
    plt.ylabel("close price")

    plt.tight_layout()

    # 保存高质量图片
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_price(
        csv_path="data/final_data.csv",
        output_path="price_trend.pdf"
    )