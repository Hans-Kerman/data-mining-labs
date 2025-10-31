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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_iris_csv(path: str) -> pd.DataFrame:
    # 读取 CSV，并尽量健壮地处理可能存在的“索引列”
    df = pd.read_csv(path)
    # 若第一列是 Unnamed: 0 或看起来是纯数字索引，则丢弃
    first_col = df.columns[0]
    if str(first_col).lower().startswith("unnamed") or (
        df[first_col].astype(str).str.fullmatch(r"\d+").all() and df.shape[1] == 6
    ):
        df = df.drop(columns=[first_col])
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # 数值特征列（不包含 Species）
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 数据可能把数值列读成 object，这里再尝试强制转换
    for c in df.columns:
        if c not in num_cols and c != "Species":
            # 尝试把字符串数字转为数值
            try:
                df[c] = pd.to_numeric(df[c], errors="raise")
                num_cols.append(c)
            except Exception:
                pass
    X = df[num_cols].copy()
    y = df["Species"].astype(str).copy()
    return X, y


def zscore_standardize(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    Xz = pd.DataFrame(scaler.fit_transform(X), columns=[f"{c}" for c in X.columns], index=X.index)
    return Xz


def discretize_equal_width(X: pd.DataFrame, bins: int = 15) -> pd.DataFrame:
    X_ew = pd.DataFrame(index=X.index)
    for c in X.columns:
        X_ew[f"{c}_ew"] = pd.cut(X[c], bins=bins, labels=False, include_lowest=True)
    return X_ew


def discretize_equal_freq(X: pd.DataFrame, q: int = 15) -> pd.DataFrame:
    X_ef = pd.DataFrame(index=X.index)
    for c in X.columns:
        # 如果重复值过多，qcut 可能会自动减少箱数；duplicates='drop' 可避免报错
        X_ef[f"{c}_ef"] = pd.qcut(X[c], q=q, labels=False, duplicates="drop")
    return X_ef


def correlations_with_target(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    返回含以下列的 DataFrame：
    - feature
    - coef_y: 与 LabelEncode(y) 的 Pearson 相关系数
    - pval_y: 上述相关系数的 p 值
    - ovr_coeffs: 一对多二元相关的字典（类别 -> 相关系数）
    - score_ovr: max(|ovr_coeff|)
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    rows: List[Dict] = []
    classes = list(le.classes_)

    for feat in X.columns:
        x = X[feat].astype(float).values
        # 与编码 y 的相关
        coef, pval = pearsonr(x, y_enc)

        # 一对多（每个类别的二元相关）
        ovr = {}
        for i, cls in enumerate(classes):
            y_bin = (y_enc == i).astype(int)
            coef_c, _ = pearsonr(x, y_bin)
            ovr[str(cls)] = float(coef_c)

        score = float(max(abs(v) for v in ovr.values()))

        rows.append(
            dict(
                feature=feat,
                coef_y=float(coef),
                pval_y=float(pval),
                ovr_coeffs=json.dumps(ovr, ensure_ascii=False),
                score_ovr=score,
            )
        )

    out = pd.DataFrame(rows).sort_values("score_ovr", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def main():
    parser = argparse.ArgumentParser(description="Iris 数据标准化/离散化/相关系数特征选择")
    parser.add_argument("--file", "-f", default="iris.csv", help="数据文件路径（默认 iris.csv）")
    parser.add_argument("--outdir", "-o", default="output", help="输出目录（默认 output）")
    parser.add_argument("--bins", type=int, default=15, help="等宽/等高分箱数（默认 15）")
    parser.add_argument("--k", type=int, default=2, help="选择前 K 个最优特征（默认 2）")
    args = parser.parse_args()

    df = load_iris_csv(args.file)
    assert "Species" in df.columns, "数据中必须包含目标列 `Species`"
    X, y = split_features_target(df)

    # 1) z-score 标准化
    Xz = zscore_standardize(X)
    df_standardized = pd.concat([Xz, y.rename("Species")], axis=1)
    save_csv(df_standardized, os.path.join(args.outdir, "standardized_iris.csv"))

    # 2) 离散化：等宽 & 等高
    X_ew = discretize_equal_width(X, bins=args.bins)
    df_ew = pd.concat([X_ew, y.rename("Species")], axis=1)
    save_csv(df_ew, os.path.join(args.outdir, "discretized_equiwidth_iris.csv"))

    X_ef = discretize_equal_freq(X, q=args.bins)
    df_ef = pd.concat([X_ef, y.rename("Species")], axis=1)
    save_csv(df_ef, os.path.join(args.outdir, "discretized_equifreq_iris.csv"))

    # 3) 特征选择：相关系数法
    corr_df = correlations_with_target(X, y)
    save_csv(corr_df, os.path.join(args.outdir, "correlations.csv"))

    # 选择前 K 个
    k = max(1, min(args.k, len(X.columns)))
    topk = corr_df.head(k)["feature"].tolist()

    print("\n=== 处理结果摘要 ===")
    print(f"数值特征：{list(X.columns)}")
    print(f"z-score 标准化文件：{os.path.join(args.outdir, 'standardized_iris.csv')}")
    print(f"等宽离散化文件：{os.path.join(args.outdir, 'discretized_equiwidth_iris.csv')}")
    print(f"等高离散化文件：{os.path.join(args.outdir, 'discretized_equifreq_iris.csv')}")
    print(f"相关性评分文件：{os.path.join(args.outdir, 'correlations.csv')}")
    print("\n— 相关性（按 OVR 最大绝对相关降序 Top 10）—")
    print(corr_df[["rank", "feature", "coef_y", "pval_y", "score_ovr"]].head(10).to_string(index=False))
    print(f"\n按 score_ovr 选择的前 K={k} 个最佳特征：{topk}\n")


if __name__ == "__main__":
    main()