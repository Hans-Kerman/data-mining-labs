# -*- coding: utf-8 -*-
"""
Iris 数据处理与特征选择
功能：
1) z-score 标准化（仅数值特征）
2) 每个数值属性的离散化：
   - 15 个等宽区间 (equal-width)
   - 15 个等高区间 (equal-frequency)
3) 特征选择（相关系数法）：
   - 将多分类目标 Species 进行 LabelEncode（0/1/2）
   - 计算：每个数值特征与编码后的 y 的 Pearson 相关系数（coef_y）
   - 同时进行“一对多（One-vs-Rest）”二元相关（对每个类构造 y_c∈{0,1}，与特征做 Pearson），
     以 max(|corr(feature, y_c)|) 作为多分类下的特征评分 score_ovr
   - 按 score_ovr 选择前 K 个最佳特征
输出：
- output/standardized_iris.csv
- output/discretized_equiwidth_iris.csv
- output/discretized_equifreq_iris.csv
- output/correlations.csv （相关性评分与排名）
"""

from __future__ import annotations
import argparse
import json
import os
# 这几个是必须的
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_iris_csv(path):
    # 读取 CSV
    data_table = pd.read_csv(path)
    # 如果第一列是索引，就扔掉
    col1_name = data_table.columns[0]
    if str(col1_name).lower().startswith("unnamed") or (
        data_table[col1_name].astype(str).str.fullmatch(r"\d+").all() and data_table.shape[1] == 6
    ):
        data_table = data_table.drop(columns=[col1_name])
    return data_table


def split_features_target(input_df):
    # 找到数值列
    numeric_columns = input_df.select_dtypes(include=[np.number]).columns.tolist()
    # 有些列可能是字符串，但其实是数字，试试看能不能转
    for column_name in input_df.columns:
        if column_name not in numeric_columns and column_name != "Species":
            try:
                input_df[column_name] = pd.to_numeric(input_df[column_name], errors="raise")
                numeric_columns.append(column_name)
            except Exception:
                pass # 转不了就算了
    
    features_data = input_df[numeric_columns].copy()
    target_series = input_df["Species"].astype(str).copy()
    return features_data, target_series


def zscore_standardize(some_data_X):
    my_scaler = StandardScaler()
    result_df = pd.DataFrame(my_scaler.fit_transform(some_data_X), columns=some_data_X.columns, index=some_data_X.index)
    return result_df


def discretize_equal_width(data_in, bins=15):
    new_df = pd.DataFrame(index=data_in.index)
    for col in data_in.columns:
        new_df[f"{col}_ew"] = pd.cut(data_in[col], bins=bins, labels=False, include_lowest=True)
    return new_df


def discretize_equal_freq(data_in, q=15):
    new_df = pd.DataFrame(index=data_in.index)
    for col in data_in.columns:
        # 用 qcut，可能会报错，所以加个 duplicates='drop'
        new_df[f"{col}_ef"] = pd.qcut(data_in[col], q=q, labels=False, duplicates="drop")
    return new_df


def correlations_with_target(x_features, y_target):
    """
    计算相关性
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_target)

    list_of_results = []
    class_names = list(encoder.classes_)

    for feature_name in x_features.columns:
        feature_values = x_features[feature_name].astype(float).values
        # 和编码后的 y 的相关性
        c1, p1 = pearsonr(feature_values, y_encoded)

        # 一对多
        ovr_dict = {}
        for idx, class_name in enumerate(class_names):
            y_binary = (y_encoded == idx).astype(int)
            c2, _ = pearsonr(feature_values, y_binary)
            ovr_dict[str(class_name)] = float(c2)

        final_score = float(max(abs(v) for v in ovr_dict.values()))

        # 把结果存起来
        list_of_results.append(
            dict(
                feature=feature_name,
                coef_y=float(c1),
                pval_y=float(p1),
                ovr_coeffs=json.dumps(ovr_dict, ensure_ascii=False),
                score_ovr=final_score,
            )
        )

    result_table = pd.DataFrame(list_of_results).sort_values("score_ovr", ascending=False).reset_index(drop=True)
    result_table["rank"] = np.arange(1, len(result_table) + 1)
    return result_table


def save_csv(dataframe_to_save, file_path):
    # 确保文件夹存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    dataframe_to_save.to_csv(file_path, index=False)
    print(f"文件已保存至: {file_path}")


def main():
    arg_parser = argparse.ArgumentParser(description="Iris 数据标准化/离散化/相关系数特征选择")
    arg_parser.add_argument("--file", "-f", default="iris.csv", help="数据文件路径（默认 iris.csv）")
    # 唯一重要的改动：默认输出路径
    arg_parser.add_argument("--outdir", "-o", default="output/iris", help="输出目录（默认 output/iris）")
    arg_parser.add_argument("--bins", type=int, default=15, help="等宽/等高分箱数（默认 15）")
    arg_parser.add_argument("--k", type=int, default=2, help="选择前 K 个最优特征（默认 2）")
    parsed_args = arg_parser.parse_args()

    # 1. 加载数据
    raw_data = load_iris_csv(parsed_args.file)
    assert "Species" in raw_data.columns, "数据中必须包含目标列 `Species`"
    features, target = split_features_target(raw_data)

    # 2. 先计算特征选择，打乱原来的顺序
    correlation_results = correlations_with_target(features, target)
    path1 = os.path.join(parsed_args.outdir, "correlations.csv")
    save_csv(correlation_results, path1)

    # 3. 再做 z-score 标准化
    standardized_features = zscore_standardize(features)
    final_df_1 = pd.concat([standardized_features, target.rename("Species")], axis=1)
    path2 = os.path.join(parsed_args.outdir, "standardized_iris.csv")
    save_csv(final_df_1, path2)

    # 4. 然后做离散化
    ew_features = discretize_equal_width(features, bins=parsed_args.bins)
    final_df_2 = pd.concat([ew_features, target.rename("Species")], axis=1)
    path3 = os.path.join(parsed_args.outdir, "discretized_equiwidth_iris.csv")
    save_csv(final_df_2, path3)

    ef_features = discretize_equal_freq(features, q=parsed_args.bins)
    final_df_3 = pd.concat([ef_features, target.rename("Species")], axis=1)
    path4 = os.path.join(parsed_args.outdir, "discretized_equifreq_iris.csv")
    save_csv(final_df_3, path4)

    # 5. 最后打印总结
    top_k_value = max(1, min(parsed_args.k, len(features.columns)))
    top_features_list = correlation_results.head(top_k_value)["feature"].tolist()

    print("\n=== 数据处理结果摘要 ===")
    print(f"已识别的数值特征: {list(features.columns)}")
    print(f"z-score 标准化文件已生成: {path2}")
    print(f"等宽离散化文件已生成: {path3}")
    print(f"等高离散化文件已生成: {path4}")
    print(f"特征相关性评分文件已生成: {path1}")
    print("\n— 特征相关性排序 (基于 OVR 评分, Top 10) —")
    print(correlation_results[["rank", "feature", "coef_y", "pval_y", "score_ovr"]].head(10).to_string(index=False))
    print(f"\n根据 score_ovr 筛选出的 Top-{top_k_value} 特征为: {top_features_list}\n")


if __name__ == "__main__":
    main()
