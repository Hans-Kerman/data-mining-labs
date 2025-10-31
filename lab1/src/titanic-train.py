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

def load_data(path):
    # 读入 CSV 文件
    data = pd.read_csv(path)
    return data

def check_missing(df_input):
    # 统计缺失值
    missing_stats = df_input.isna().sum().sort_values(ascending=False)
    print("\n=== (a) 各列缺失值统计 ===")
    print(missing_stats.to_string())
    total_cells_num = df_input.shape[0] * df_input.shape[1]
    print(f"\n总单元格数：{total_cells_num}，缺失单元格数：{int(missing_stats.sum())}，缺失占比：{missing_stats.sum() / total_cells_num:.2%}")
    return missing_stats

def fill_age_with_mean(df_original):
    # 对 Age 列进行均值填充
    df_copy = df_original.copy()
    count_missing = int(df_copy["Age"].isna().sum())
    mean_age_val = float(df_copy["Age"].mean(skipna=True))
    df_copy["Age"] = df_copy["Age"].fillna(mean_age_val)
    print("\n=== (b) 年龄均值填充 ===")
    print(f"Age 缺失个数：{count_missing}；均值：{mean_age_val:.4f}（已用该值填充缺失Age）")
    return df_copy, mean_age_val, count_missing

def plot_age_distribution(df_for_plot, output_folder, calculated_mean):
    # 画直方图 和 边际箱线图
    fig_histogram = px.histogram(
        df_for_plot, x="Age", nbins=30, marginal="box",
        title="年龄分布（直方图 + 边际箱线图）",
        labels={"Age": "Age（岁）"},
        opacity=0.85
    )
    # 添加均值线
    fig_histogram.add_vline(x=calculated_mean, line_color="red", line_width=2, line_dash="dash",
                  annotation_text=f"Mean={calculated_mean:.2f}", annotation_position="top right")
    os.makedirs(output_folder, exist_ok=True)
    file_path_hist = os.path.join(output_folder, "age_hist.html")
    fig_histogram.write_html(file_path_hist, include_plotlyjs="cdn")
    print(f"文件已保存: {file_path_hist}")

def plot_age_box_by_sex(df_data, folder_out):
    # 根据性别画箱线图
    box_fig = px.box(
        df_data, x="Sex", y="Age", points="outliers",
        title="基于性别的乘客年龄分布（箱线图）",
        labels={"Sex": "性别", "Age": "Age（岁）"},
        color="Sex"
    )
    os.makedirs(folder_out, exist_ok=True)
    box_file_path = os.path.join(folder_out, "age_by_sex_box.html")
    box_fig.write_html(box_file_path, include_plotlyjs="cdn")
    print(f"文件已保存: {box_file_path}")

def plot_embarked_pie(df_source, dir_output):
    # 画登船地点饼图
    # 先过滤掉不是 S/C/Q 的
    valid_mask = df_source["Embarked"].isin(["S", "C", "Q"])
    embarked_counts = df_source.loc[valid_mask, "Embarked"].value_counts().reset_index()
    embarked_counts.columns = ["Embarked", "Count"]
    pie_chart = px.pie(
        embarked_counts, names="Embarked", values="Count",
        title="登船地点（S、C、Q）人数占比",
        hole=0.0
    )
    pie_chart.update_traces(textposition="inside", textinfo="percent+label")
    os.makedirs(dir_output, exist_ok=True)
    pie_file_path = os.path.join(dir_output, "embarked_pie.html")
    pie_chart.write_html(pie_file_path, include_plotlyjs="cdn")
    print(f"文件已保存: {pie_file_path}")

def main():
    parser_object = argparse.ArgumentParser(description="Titanic 训练集数据分析与可视化")
    parser_object.add_argument("--file", "-f", default="titanic-train.csv", help="CSV 文件路径（默认 titanic-train.csv）")
    parser_object.add_argument("--outdir", "-o", default="/output/titanic-train", help="输出目录（默认 /output/titanic-train）")
    parsed_arguments = parser_object.parse_args()

    dataframe = load_data(parsed_arguments.file)

    check_missing(dataframe)
    filled_dataframe, mean_of_age, _ = fill_age_with_mean(dataframe)
    plot_embarked_pie(filled_dataframe, parsed_arguments.outdir)
    plot_age_distribution(filled_dataframe, parsed_arguments.outdir, mean_of_age)
    plot_age_box_by_sex(filled_dataframe, parsed_arguments.outdir)

    # 最后保存填充后的 csv
    output_csv_path = os.path.join(parsed_arguments.outdir, "titanic_filled.csv")
    filled_dataframe.to_csv(output_csv_path, index=False)
    print(f"文件已保存: {output_csv_path}")

if __name__ == "__main__":
    main()
