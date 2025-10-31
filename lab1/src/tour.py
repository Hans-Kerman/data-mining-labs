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
# 这几个是必须的
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# 初步处理数字
def _normalize_number_series(s):
    # 先转成字符串，然后去掉逗号和空格，最后转成数字
    temp_str = s.astype(str).str.replace(",", "", regex=False)
    cleaned_str = temp_str.str.strip()
    return pd.to_numeric(cleaned_str, errors="coerce")


def _find_column_by_patterns(all_cols, patterns_to_try):
    """
    在 columns 中按一组正则模式依次查找匹配列名，返回首个命中的列名
    """
    for p in patterns_to_try:
        my_regex = re.compile(p, re.IGNORECASE)
        for c in all_cols:
            if my_regex.search(c):
                return c # 找到了就返回
    return None


def auto_detect_columns(df):
    """
    自动识别常见中文/中英混合列名
    返回字典：year, urban, rural, total（有些表可能缺 total）
    """
    column_list = df.columns.tolist()

    # 找年份
    year_col_name = _find_column_by_patterns(
        column_list,
        [
            r"统计.*年|年份|^年$|year",
        ],
    )
    # 找城镇
    urban_col_name = _find_column_by_patterns(
        column_list,
        [
            r"城镇.*(居民)?.*国内.*旅游.*人次",
            r"城市.*(居民)?.*国内.*旅游.*人次",
            r"城镇.*人次",
        ],
    )
    # 找农村
    rural_col_name = _find_column_by_patterns(
        column_list,
        [
            r"农村.*(居民)?.*国内.*旅游.*人次",
            r"乡村.*(居民)?.*国内.*旅游.*人次",
            r"农村.*人次",
        ],
    )
    # 找总数
    total_col_name = _find_column_by_patterns(
        column_list,
        [
            r"国内.*旅游.*总.*人次",
            r"总.*人次",
        ],
    )

    # 把找到的放进一个字典里
    found_stuff = dict(year=year_col_name, urban=urban_col_name, rural=rural_col_name, total=total_col_name)
    return found_stuff


def compute_metrics(data_in):
    """
    计算各种指标
    """
    my_df = data_in.copy().sort_values("Year")
    
    # 总人次：优先使用数据列，否则用和
    if "TotalTrips_src" in my_df.columns and my_df["TotalTrips_src"].notna().any():
        my_df["TotalTrips"] = my_df["TotalTrips_src"]
    else:
        my_df["TotalTrips"] = my_df["UrbanTrips"] + my_df["RuralTrips"]

    # 算占比
    temp_total = my_df["UrbanTrips"] + my_df["RuralTrips"]
    my_df["UrbanShare"] = my_df["UrbanTrips"] / temp_total
    my_df["RuralShare"] = my_df["RuralTrips"] / temp_total

    # 算同比
    my_df["UrbanYoY"] = my_df["UrbanTrips"].pct_change()
    my_df["RuralYoY"] = my_df["RuralTrips"].pct_change()
    my_df["TotalYoY"] = my_df["TotalTrips"].pct_change()

    # 算CAGR
    def cagr(some_series):
        s = some_series.dropna()
        if len(s) < 2:
            return np.nan
        years = len(s) - 1
        val1, val2 = s.iloc[0], s.iloc[-1]
        # 开始值不能是0
        if val1 <= 0:
            return np.nan
        return (val2 / val1) ** (1 / years) - 1

    my_df.attrs["CAGR_Urban"] = cagr(my_df["UrbanTrips"])
    my_df.attrs["CAGR_Rural"] = cagr(my_df["RuralTrips"])
    my_df.attrs["CAGR_Total"] = cagr(my_df["TotalTrips"])
    my_df.attrs["YearStart"] = int(my_df["Year"].min())
    my_df.attrs["YearEnd"] = int(my_df["Year"].max())

    return my_df


def fig_trend(data_for_plot):
    f = go.Figure()
    f.add_trace(
        go.Scatter(
            x=data_for_plot["Year"], y=data_for_plot["UrbanTrips"],
            mode="lines+markers", name="城镇居民人次",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="年份=%{x}<br>城镇人次=%{y}<extra></extra>"
        )
    )
    f.add_trace(
        go.Scatter(
            x=data_for_plot["Year"], y=data_for_plot["RuralTrips"],
            mode="lines+markers", name="农村居民人次",
            line=dict(color="#2ca02c", width=2),
            hovertemplate="年份=%{x}<br>农村人次=%{y}<extra></extra>"
        )
    )
    if "TotalTrips" in data_for_plot.columns:
        f.add_trace(
            go.Scatter(
                x=data_for_plot["Year"], y=data_for_plot["TotalTrips"],
                mode="lines", name="国内旅游总人次",
                line=dict(color="#9467bd", width=2, dash="dot"),
                hovertemplate="年份=%{x}<br>总人次=%{y}<extra></extra>"
            )
        )
    
    # 设置一下样式
    font_style = dict(family="Microsoft YaHei, Noto Sans CJK SC, SimHei, Arial", size=12)
    f.update_layout(
        title="城乡居民国内旅游人次趋势",
        xaxis_title="年份",
        yaxis_title="人次（单位与原表一致）",
        hovermode="x unified",
        template="plotly_white",
        font=font_style,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=True))
    )
    return f


def fig_share(df_for_share_plot):
    the_plot = go.Figure()
    the_plot.add_trace(
        go.Scatter(
            x=df_for_share_plot["Year"], y=df_for_share_plot["UrbanShare"],
            mode="lines", stackgroup="one", name="城镇占比",
            line=dict(color="#1f77b4"),
            hovertemplate="年份=%{x}<br>城镇占比=%{y:.1%}<extra></extra>"
        )
    )
    the_plot.add_trace(
        go.Scatter(
            x=df_for_share_plot["Year"], y=df_for_share_plot["RuralShare"],
            mode="lines", stackgroup="one", name="农村占比",
            line=dict(color="#2ca02c"),
            hovertemplate="年份=%{x}<br>农村占比=%{y:.1%}<extra></extra>"
        )
    )

    # 设置一下样式
    font_style = dict(family="Microsoft YaHei, Noto Sans CJK SC, SimHei, Arial", size=12)
    the_plot.update_layout(
        title="城乡旅游人次结构占比（堆叠面积）",
        xaxis_title="年份",
        yaxis_title="占比",
        yaxis=dict(tickformat=".0%"),
        hovermode="x unified",
        template="plotly_white",
        font=font_style,
        xaxis=dict(rangeslider=dict(visible=True))
    )
    return the_plot


def fig_yoy(yoy_data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=yoy_data["Year"], y=yoy_data["TotalYoY"] * 100,
            name="总人次同比(%)",
            marker_color="#9467bd",
            hovertemplate="年份=%{x}<br>总人次同比=%{y:.2f}%<extra></extra>",
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=yoy_data["Year"], y=yoy_data["UrbanYoY"] * 100,
            mode="lines+markers", name="城镇同比(%)",
            line=dict(color="#1f77b4"),
            hovertemplate="年份=%{x}<br>城镇同比=%{y:.2f}%<extra></extra>",
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=yoy_data["Year"], y=yoy_data["RuralYoY"] * 100,
            mode="lines+markers", name="农村同比(%)",
            line=dict(color="#2ca02c"),
            hovertemplate="年份=%{x}<br>农村同比=%{y:.2f}%<extra></extra>",
        ),
        secondary_y=True
    )

    # 设置一下样式
    font_style = dict(family="Microsoft YaHei, Noto Sans CJK SC, SimHei, Arial", size=12)
    fig.update_layout(
        title="旅游人次同比增速",
        xaxis_title="年份",
        template="plotly_white",
        hovermode="x unified",
        font=font_style,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="总人次同比(%)", secondary_y=False)
    fig.update_yaxes(title_text="城镇/农村同比(%)", secondary_y=True)
    return fig


def load_and_prepare(
    file_path,
    year_col = None,
    urban_col = None,
    rural_col = None,
    total_col = None,
    sheet_name = 0,
):
    raw_df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 自动识别列名
    detected_cols = auto_detect_columns(raw_df)

    if year_col:
        a = year_col
    else:
        a = detected_cols["year"]
    
    if urban_col:
        b = urban_col
    else:
        b = detected_cols["urban"]

    if rural_col:
        c = rural_col
    else:
        c = detected_cols["rural"]

    if total_col:
        d = total_col
    else:
        d = detected_cols["total"]

    # 检查必须的列
    missing = []
    if not a: 
        missing.append("year")
    if not b: 
        missing.append("urban")
    if not c: 
        missing.append("rural")
    if missing:
        raise ValueError(f"找不到这几列：{missing}；只找到了以下列={raw_df.columns.tolist()}")

    # 创建一个新的干净的DataFrame
    clean_data = pd.DataFrame()
    clean_data["Year"] = pd.to_numeric(raw_df[a], errors="coerce").astype("Int64")
    clean_data["UrbanTrips"] = _normalize_number_series(raw_df[b])
    clean_data["RuralTrips"] = _normalize_number_series(raw_df[c])
    if d:
        clean_data["TotalTrips_src"] = _normalize_number_series(raw_df[d])

    # 去掉有问题的行
    clean_data = clean_data.dropna(subset=["Year", "UrbanTrips", "RuralTrips"])
    clean_data["Year"] = clean_data["Year"].astype(int)

    # 最后计算指标
    final_data = compute_metrics(clean_data)
    return final_data


def save_fig(figure_to_save, path_to_save):
    # 确保文件夹存在
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    figure_to_save.write_html(path_to_save, include_plotlyjs="cdn")
    print(f"已经保存到: {path_to_save}")


def print_summary(df_for_summary):
    # 打印总结信息
    y0 = df_for_summary.attrs.get("YearStart")
    y1 = df_for_summary.attrs.get("YearEnd")

    def pct(x):
        return "N/A" if x is None or np.isnan(x) else f"{x*100:.2f}%"

    print("\n=== 数据总结 ===")
    print(f"时间范围：{y0} — {y1}")

    cagr_u = df_for_summary.attrs.get("CAGR_Urban")
    urban_first, urban_last = df_for_summary["UrbanTrips"].iloc[0], df_for_summary["UrbanTrips"].iloc[-1]
    print(f"城镇人次：{urban_first:.2f} -> {urban_last:.2f}，CAGR={pct(cagr_u)}")
    
    cagr_r = df_for_summary.attrs.get("CAGR_Rural")
    rural_first, rural_last = df_for_summary["RuralTrips"].iloc[0], df_for_summary["RuralTrips"].iloc[-1]
    print(f"农村人次：{rural_first:.2f} -> {rural_last:.2f}，CAGR={pct(cagr_r)}")
    
    cagr_t = df_for_summary.attrs.get("CAGR_Total")
    total_first, total_last = df_for_summary["TotalTrips"].iloc[0], df_for_summary["TotalTrips"].iloc[-1]
    print(f"总  人 次：{total_first:.2f} -> {total_last:.2f}，CAGR={pct(cagr_t)}")

    # 找找最大值在哪年
    u_pk_year = int(df_for_summary.loc[df_for_summary["UrbanTrips"].idxmax(), "Year"])
    r_pk_year = int(df_for_summary.loc[df_for_summary["RuralTrips"].idxmax(), "Year"])
    print(f"城镇人次顶峰年份：{u_pk_year}；农村人次顶峰年份：{r_pk_year}")
    print("===================\n")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Tour.xlsx 城乡居民旅游人次分析与可视化")
    parser.add_argument("--file", "-f", required=True, help="Excel 文件路径（如 Tour.xlsx）")
    parser.add_argument("--sheet", default=0, help="工作表名或索引（默认 0）")
    parser.add_argument("--year-col", help="年份列名（可选）")
    parser.add_argument("--urban-col", help="城镇居民国内旅游人次列名（可选）")
    parser.add_argument("--rural-col", help="农村居民国内旅游人次列名（可选）")
    parser.add_argument("--total-col", help="国内旅游总人次列名（可选）")
    parser.add_argument("--outdir", default="output/tour/", help="输出目录（默认 ./output）")
    args = parser.parse_args()

    # 加载和准备数据
    my_dataframe = load_and_prepare(
        file_path=args.file,
        year_col=args.year_col,
        urban_col=args.urban_col,
        rural_col=args.rural_col,
        total_col=args.total_col,
        sheet_name=args.sheet,
    )

    # 打印总结
    print_summary(my_dataframe)

    # 画图和保存
    print("开始生成图表...")
    
    # 趋势图
    fig1 = fig_trend(my_dataframe)
    path1 = os.path.join(args.outdir, "trend_urban_rural_trips.html")
    save_fig(fig1, path1)

    # 占比图
    fig2 = fig_share(my_dataframe)
    path2 = os.path.join(args.outdir, "share_urban_rural.html")
    save_fig(fig2, path2)

    # 增速图
    fig3 = fig_yoy(my_dataframe)
    path3 = os.path.join(args.outdir, "yoy_growth.html")
    save_fig(fig3, path3)



if __name__ == "__main__":
    main()
