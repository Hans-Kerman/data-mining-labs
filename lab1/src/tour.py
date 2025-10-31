# -*- coding: utf-8 -*-
"""
Tour.xlsx 旅游业数据分析与可视化（城乡居民旅游人次）
- 读取 Excel（支持中文列名）
- 计算总量/占比/同比增速/CAGR
- 生成 Plotly 交互图（趋势、占比、同比增速）
- 保存为 HTML 到 ./output
"""
from __future__ import annotations
import argparse
import os
import re
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


def _normalize_number_series(s: pd.Series) -> pd.Series:
    # 去除千分位逗号与空白，转为数值
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _find_column_by_patterns(columns: List[str], patterns: List[str]) -> Optional[str]:
    """
    在 columns 中按一组正则模式依次查找匹配列名，返回首个命中的列名
    """
    for pat in patterns:
        regex = re.compile(pat, re.IGNORECASE)
        for col in columns:
            if regex.search(col):
                return col
    return None


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    自动识别常见中文/中英混合列名
    返回字典：year, urban, rural, total（有些表可能缺 total）
    """
    cols = df.columns.tolist()

    year_col = _find_column_by_patterns(
        cols,
        [
            r"统计.*年|年份|^年$|year",
        ],
    )
    urban_col = _find_column_by_patterns(
        cols,
        [
            r"城镇.*(居民)?.*国内.*旅游.*人次",
            r"城市.*(居民)?.*国内.*旅游.*人次",
            r"城镇.*人次",
        ],
    )
    rural_col = _find_column_by_patterns(
        cols,
        [
            r"农村.*(居民)?.*国内.*旅游.*人次",
            r"乡村.*(居民)?.*国内.*旅游.*人次",
            r"农村.*人次",
        ],
    )
    total_col = _find_column_by_patterns(
        cols,
        [
            r"国内.*旅游.*总.*人次",
            r"总.*人次",
        ],
    )

    return dict(year=year_col, urban=urban_col, rural=rural_col, total=total_col)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算：
    - TotalTrips（若原表未给出，则用 城镇+农村 之和）
    - 城镇/农村占比
    - YoY 同比增速
    """
    df = df.copy().sort_values("Year")
    # 总人次：优先使用数据列，否则用和
    if "TotalTrips_src" in df.columns and df["TotalTrips_src"].notna().any():
        df["TotalTrips"] = df["TotalTrips_src"]
    else:
        df["TotalTrips"] = df["UrbanTrips"] + df["RuralTrips"]

    # 占比
    total = df["UrbanTrips"] + df["RuralTrips"]
    df["UrbanShare"] = df["UrbanTrips"] / total
    df["RuralShare"] = df["RuralTrips"] / total

    # 同比增速
    df["UrbanYoY"] = df["UrbanTrips"].pct_change()
    df["RuralYoY"] = df["RuralTrips"].pct_change()
    df["TotalYoY"] = df["TotalTrips"].pct_change()

    # CAGR（用于文本摘要）
    def cagr(s: pd.Series) -> Optional[float]:
        s = s.dropna()
        if len(s) < 2:
            return np.nan
        n = len(s) - 1
        first, last = s.iloc[0], s.iloc[-1]
        if first <= 0:
            return np.nan
        return (last / first) ** (1 / n) - 1

    df.attrs["CAGR_Urban"] = cagr(df["UrbanTrips"])
    df.attrs["CAGR_Rural"] = cagr(df["RuralTrips"])
    df.attrs["CAGR_Total"] = cagr(df["TotalTrips"])
    df.attrs["YearStart"] = int(df["Year"].min())
    df.attrs["YearEnd"] = int(df["Year"].max())

    return df


def fig_trend(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Year"], y=df["UrbanTrips"],
            mode="lines+markers", name="城镇居民人次",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="年份=%{x}<br>城镇人次=%{y}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Year"], y=df["RuralTrips"],
            mode="lines+markers", name="农村居民人次",
            line=dict(color="#2ca02c", width=2),
            hovertemplate="年份=%{x}<br>农村人次=%{y}<extra></extra>"
        )
    )
    # 可选：总量
    if "TotalTrips" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Year"], y=df["TotalTrips"],
                mode="lines", name="国内旅游总人次",
                line=dict(color="#9467bd", width=2, dash="dot"),
                hovertemplate="年份=%{x}<br>总人次=%{y}<extra></extra>"
            )
        )

    fig.update_layout(
        title="城乡居民国内旅游人次趋势",
        xaxis_title="年份",
        yaxis_title="人次（单位与原表一致）",
        hovermode="x unified",
        template="plotly_white",
        font=dict(family="Microsoft YaHei, Noto Sans CJK SC, SimHei, Arial", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=True))
    )
    return fig


def fig_share(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Year"], y=df["UrbanShare"],
            mode="lines", stackgroup="one", name="城镇占比",
            line=dict(color="#1f77b4"),
            hovertemplate="年份=%{x}<br>城镇占比=%{y:.1%}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Year"], y=df["RuralShare"],
            mode="lines", stackgroup="one", name="农村占比",
            line=dict(color="#2ca02c"),
            hovertemplate="年份=%{x}<br>农村占比=%{y:.1%}<extra></extra>"
        )
    )
    fig.update_layout(
        title="城乡旅游人次结构占比（堆叠面积）",
        xaxis_title="年份",
        yaxis_title="占比",
        yaxis=dict(tickformat=".0%"),
        hovermode="x unified",
        template="plotly_white",
        font=dict(family="Microsoft YaHei, Noto Sans CJK SC, SimHei, Arial", size=12),
        xaxis=dict(rangeslider=dict(visible=True))
    )
    return fig


def fig_yoy(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["Year"], y=df["TotalYoY"] * 100,
            name="总人次同比(%)",
            marker_color="#9467bd",
            hovertemplate="年份=%{x}<br>总人次同比=%{y:.2f}%<extra></extra>",
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df["Year"], y=df["UrbanYoY"] * 100,
            mode="lines+markers", name="城镇同比(%)",
            line=dict(color="#1f77b4"),
            hovertemplate="年份=%{x}<br>城镇同比=%{y:.2f}%<extra></extra>",
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=df["Year"], y=df["RuralYoY"] * 100,
            mode="lines+markers", name="农村同比(%)",
            line=dict(color="#2ca02c"),
            hovertemplate="年份=%{x}<br>农村同比=%{y:.2f}%<extra></extra>",
        ),
        secondary_y=True
    )
    fig.update_layout(
        title="旅游人次同比增速",
        xaxis_title="年份",
        template="plotly_white",
        hovermode="x unified",
        font=dict(family="Microsoft YaHei, Noto Sans CJK SC, SimHei, Arial", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="总人次同比(%)", secondary_y=False)
    fig.update_yaxes(title_text="城镇/农村同比(%)", secondary_y=True)
    return fig


def load_and_prepare(
    file_path: str,
    year_col: Optional[str] = None,
    urban_col: Optional[str] = None,
    rural_col: Optional[str] = None,
    total_col: Optional[str] = None,
    sheet_name: Optional[str | int] = 0,
) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 自动识别列名
    auto_map = auto_detect_columns(df)

    use_year = year_col or auto_map["year"]
    use_urban = urban_col or auto_map["urban"]
    use_rural = rural_col or auto_map["rural"]
    use_total = total_col or auto_map["total"]

    missing = [k for k, v in dict(year=use_year, urban=use_urban, rural=use_rural).items() if not v]
    if missing:
        raise ValueError(f"无法识别以下必需列，请使用命令行参数指定列名：{missing}；检测到的列={df.columns.tolist()}")

    out = pd.DataFrame()
    out["Year"] = pd.to_numeric(df[use_year], errors="coerce").astype("Int64")
    out["UrbanTrips"] = _normalize_number_series(df[use_urban])
    out["RuralTrips"] = _normalize_number_series(df[use_rural])
    if use_total:
        out["TotalTrips_src"] = _normalize_number_series(df[use_total])

    out = out.dropna(subset=["Year", "UrbanTrips", "RuralTrips"])
    out["Year"] = out["Year"].astype(int)
    return compute_metrics(out)


def save_fig(fig: go.Figure, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[saved] {path}")


def print_summary(df: pd.DataFrame):
    y0, y1 = df.attrs.get("YearStart"), df.attrs.get("YearEnd")
    cagr_u = df.attrs.get("CAGR_Urban")
    cagr_r = df.attrs.get("CAGR_Rural")
    cagr_t = df.attrs.get("CAGR_Total")
    urban_first, urban_last = df["UrbanTrips"].iloc[0], df["UrbanTrips"].iloc[-1]
    rural_first, rural_last = df["RuralTrips"].iloc[0], df["RuralTrips"].iloc[-1]
    total_first, total_last = df["TotalTrips"].iloc[0], df["TotalTrips"].iloc[-1]

    def pct(x):
        return "—" if x is None or np.isnan(x) else f"{x*100:.2f}%"

    print("\n=== 关键指标摘要 ===")
    print(f"时间范围：{y0} — {y1}")
    print(f"城镇人次：{urban_first:.2f} -> {urban_last:.2f}，CAGR={pct(cagr_u)}")
    print(f"农村人次：{rural_first:.2f} -> {rural_last:.2f}，CAGR={pct(cagr_r)}")
    print(f"总  人 次：{total_first:.2f} -> {total_last:.2f}，CAGR={pct(cagr_t)}")
    # 峰谷年份
    u_pk_year = int(df.loc[df["UrbanTrips"].idxmax(), "Year"])
    r_pk_year = int(df.loc[df["RuralTrips"].idxmax(), "Year"])
    print(f"城镇人次峰值年份：{u_pk_year}；农村人次峰值年份：{r_pk_year}")
    print("===================\n")


def main():
    parser = argparse.ArgumentParser(description="Tour.xlsx 城乡居民旅游人次分析与可视化")
    parser.add_argument("--file", "-f", required=True, help="Excel 文件路径（如 Tour.xlsx）")
    parser.add_argument("--sheet", default=0, help="工作表名或索引（默认 0）")
    parser.add_argument("--year-col", help="年份列名（可选）")
    parser.add_argument("--urban-col", help="城镇居民国内旅游人次列名（可选）")
    parser.add_argument("--rural-col", help="农村居民国内旅游人次列名（可选）")
    parser.add_argument("--total-col", help="国内旅游总人次列名（可选）")
    parser.add_argument("--outdir", default="output", help="输出目录（默认 ./output）")
    args = parser.parse_args()

    df = load_and_prepare(
        file_path=args.file,
        year_col=args.year_col,
        urban_col=args.urban_col,
        rural_col=args.rural_col,
        total_col=args.total_col,
        sheet_name=args.sheet,
    )

    print_summary(df)

    fig1 = fig_trend(df)
    fig2 = fig_share(df)
    fig3 = fig_yoy(df)

    save_fig(fig1, os.path.join(args.outdir, "trend_urban_rural_trips.html"))
    save_fig(fig2, os.path.join(args.outdir, "share_urban_rural.html"))
    save_fig(fig3, os.path.join(args.outdir, "yoy_growth.html"))

    # 直接展示（可选）
    # fig1.show(); fig2.show(); fig3.show()


if __name__ == "__main__":
    main()