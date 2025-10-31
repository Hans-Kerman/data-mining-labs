# -*- coding: utf-8 -*-
"""
Titanic 训练集数据分析与可视化
任务：
(a) 查看缺失值情况
(b) 年龄Age用均值填充
(c) 年龄分布可视化（直方图+核密度/箱线图边际）
(d) 基于性别的年龄箱线图
(e) 登船地点（S、C、Q）人数占比饼图

输出：
- output/age_hist.html
- output/age_by_sex_box.html
- output/embarked_pie.html
- output/titanic_filled.csv（Age已均值填充）
"""

from __future__ import annotations
import argparse
import os
import pandas as pd
import plotly.express as px

def load_data(path: str) -> pd.DataFrame:
    # 读入 CSV；若存在空白值、NaN等，pandas会自动解析为NaN
    df = pd.read_csv(path)
    return df

def check_missing(df: pd.DataFrame) -> pd.Series:
    miss = df.isna().sum().sort_values(ascending=False)
    print("\n=== (a) 各列缺失值统计 ===")
    print(miss.to_string())
    total_cells = df.shape[0] * df.shape[1]
    print(f"\n总单元格数：{total_cells}，缺失单元格数：{int(miss.sum())}，缺失占比：{miss.sum() / total_cells:.2%}")
    return miss

def fill_age_with_mean(df: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    # 仅对 Age 列进行均值填充
    df = df.copy()
    n_missing = int(df["Age"].isna().sum())
    age_mean = float(df["Age"].mean(skipna=True))
    df["Age"] = df["Age"].fillna(age_mean)
    print("\n=== (b) 年龄均值填充 ===")
    print(f"Age 缺失个数：{n_missing}；均值：{age_mean:.4f}（已用该值填充缺失Age）")
    return df, age_mean, n_missing

def plot_age_distribution(df: pd.DataFrame, outdir: str, age_mean: float):
    # 直方图 + 边际箱线图
    fig = px.histogram(
        df, x="Age", nbins=30, marginal="box",
        title="年龄分布（直方图 + 边际箱线图）",
        labels={"Age": "Age（岁）"},
        opacity=0.85
    )
    fig.add_vline(x=age_mean, line_color="red", line_width=2, line_dash="dash",
                  annotation_text=f"Mean={age_mean:.2f}", annotation_position="top right")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "age_hist.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[saved] {path}")

def plot_age_box_by_sex(df: pd.DataFrame, outdir: str):
    fig = px.box(
        df, x="Sex", y="Age", points="outliers",
        title="基于性别的乘客年龄分布（箱线图）",
        labels={"Sex": "性别", "Age": "Age（岁）"},
        color="Sex"
    )
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "age_by_sex_box.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[saved] {path}")

def plot_embarked_pie(df: pd.DataFrame, outdir: str):
    # 仅统计 S/C/Q 三个登船地点，排除缺失
    mask = df["Embarked"].isin(["S", "C", "Q"])
    counts = df.loc[mask, "Embarked"].value_counts().reset_index()
    counts.columns = ["Embarked", "Count"]
    fig = px.pie(
        counts, names="Embarked", values="Count",
        title="登船地点（S、C、Q）人数占比",
        hole=0.0
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "embarked_pie.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[saved] {path}")

def main():
    parser = argparse.ArgumentParser(description="Titanic 训练集数据分析与可视化")
    parser.add_argument("--file", "-f", default="titanic-train.csv", help="CSV 文件路径（默认 titanic-train.csv）")
    parser.add_argument("--outdir", "-o", default="output", help="输出目录（默认 output）")
    args = parser.parse_args()

    df = load_data(args.file)

    # (a) 缺失值检查
    check_missing(df)

    # (b) 年龄均值填充
    df_filled, age_mean, _ = fill_age_with_mean(df)

    # (c) 年龄分布可视化
    plot_age_distribution(df_filled, args.outdir, age_mean)

    # (d) 基于性别的年龄箱线图
    plot_age_box_by_sex(df_filled, args.outdir)

    # (e) 登船地点饼图
    plot_embarked_pie(df_filled, args.outdir)

    # 额外：保存填充后的数据
    out_csv = os.path.join(args.outdir, "titanic_filled.csv")
    df_filled.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

if __name__ == "__main__":
    main()