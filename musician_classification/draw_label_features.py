#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot comparative charts for music‑genre reports with consistent colours.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 画图全局字体 / 样式 ----------
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


# ---------- 1. 解析文本报告 ----------
# ========================== 新版 parse_report ==========================
def parse_report(text: str) -> pd.DataFrame:
    """
    解析“流派分析报告”文本块，兼容“====”分隔线和括号内指标。
    返回：每行 = 一个流派的统计特征
    """
    # --- ① 先用正则把所有 (genre, block) 提取出来 ---
    pattern_block = re.compile(
        r'流派分析报告[:：]\s*([A-Z\-]+)\s*={5,}\s*'   # genre 行，后跟至少 5 个 '='
        r'(.*?)'                                       # 捕获到下一个流派或文本结尾
        r'(?=流派分析报告[:：]|\Z)',
        flags=re.S | re.I)
    matches = pattern_block.findall(text)

    rows = []
    for genre, block in matches:
        genre = genre.lower()

        # ---------- ② 从 block 中抓数字 ----------
        num = r'([\d,.]+)'
        pct = r'([\d.]+)\s*%'

        basics = re.search(
            rf'平均每个音乐家的听众数[^0-9]*{num}.*?'
            rf'总收听次数[^0-9]*{num}.*?'
            rf'听众平均收听次数[^0-9]*{num}.*?'
            rf'平均最高收听[^0-9]*{num}.*?'
            rf'平均头部用户占比[^0-9]*{pct}.*?'
            rf'平均核心听众比例[^0-9]*{pct}',
            block, re.S)

        social = re.search(
            rf'平均听众好友覆盖率[^0-9]*{pct}.*?'
            rf'平均好友数[^0-9]*{num}.*?'
            rf'平均潜在传播对象[^0-9]*{num}.*?'
            rf'平均听众聚类系数[^0-9]*{num}',
            block, re.S)

        if basics and social:
            rows.append({
                "genre":            genre,
                "listeners":        float(basics.group(1)),
                "total_plays":      float(basics.group(2)),
                "avg_plays":        float(basics.group(3)),
                "max_plays":        float(basics.group(4).replace(',', '')),
                "top_user_ratio":   float(basics.group(5))/100,
                "core_listeners":   float(basics.group(6))/100,
                "friend_coverage":  float(social.group(1))/100,
                "avg_friends":      float(social.group(2)),
                "potential_spread": float(social.group(3)),
                "clustering":       float(social.group(4))
            })

    return pd.DataFrame(rows)


# ---------- 2. 生成可视化 ----------
def plot_genre_comparison(df: pd.DataFrame, out_file: str = "genre_comparison_english.png") -> None:
    # —— 统一颜色表 —— #
    unique_genres = df['genre'].unique()
    palette      = sns.color_palette('tab20', len(unique_genres))
    genre_colors = dict(zip(unique_genres, palette))

    plt.figure(figsize=(16, 14))
    plt.suptitle("Music Genre Feature Comparison", fontsize=18, y=1.02)

    # 2‑1. 基础特征雷达图
    metrics = ["listeners", "total_plays", "avg_plays", "core_listeners"]
    angles  = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles  = np.concatenate([angles, angles[:1]])

    ax1 = plt.subplot2grid((3, 2), (0, 0), polar=True)
    for _, row in df.iterrows():
        vals = row[metrics].values.tolist()
        vals += vals[:1]
        ax1.plot(angles, vals, "o-", color=genre_colors[row["genre"]],
                 linewidth=2, markersize=6, label=row["genre"].title())
        ax1.fill(angles, vals, color=genre_colors[row["genre"]], alpha=0.15)

    ax1.set_thetagrids(angles[:-1]*180/np.pi,
                       ["Listeners", "Total Plays", "Plays/Listener", "Core Listeners"])
    ax1.set_title("Basic Features Radar Chart", pad=20, fontsize=14)
    ax1.legend(bbox_to_anchor=(1.25, 1.1), fontsize=9, title="Genres")

    # 2‑2. 社交气泡图
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    for _, row in df.iterrows():
        ax2.scatter(row["friend_coverage"], row["clustering"],
                    s=row["potential_spread"]*3,
                    color=genre_colors[row["genre"]], alpha=0.8)
        ax2.text(row["friend_coverage"], row["clustering"],
                 row["genre"].title(), ha="center", va="center", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))

    ax2.set_xlabel("Friend Coverage")
    ax2.set_ylabel("Clustering Coefficient")
    ax2.set_title("Social Network Features\n(Bubble Size = Potential Reach)")
    ax2.grid(True, alpha=0.3)

    # 2‑3. 收听量柱状图（三指标，同色+纹理区分）
    ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    bar_width = 0.25
    x         = np.arange(len(df))
    hatches   = ["", "//", "xx"]
    play_metrics = ["total_plays", "avg_plays", "max_plays"]

    for j, metric in enumerate(play_metrics):
        ax3.bar(x + j*bar_width, df[metric],
                width=bar_width,
                color=[genre_colors[g] for g in df["genre"]],
                alpha=0.85,
                hatch=hatches[j],
                label=metric.replace("_", " ").title())

    ax3.set_title("Play Count Metrics Comparison")
    ax3.set_ylabel("Play Count")
    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels([g.title() for g in df["genre"]])
    ax3.legend()

    # 标签
    for i, genre in enumerate(df["genre"]):
        for j, metric in enumerate(play_metrics):
            height = df.loc[i, metric]
            ax3.text(i + j*bar_width, height*1.02, f"{height:,.0f}",
                     ha="center", va="bottom", fontsize=7)

    # 2‑4. 忠诚度热图
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    heat = df[["genre", "core_listeners", "top_user_ratio"]].set_index("genre")
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="YlOrRd", ax=ax4,
                cbar_kws={"label": "Percentage"})
    ax4.set_title("Listener Loyalty Metrics")
    ax4.set_yticklabels([g.title() for g in heat.index], rotation=0)

    # 2‑5. 社交影响堆叠条  ——  加入不同纹理
    ax5 = plt.subplot2grid((3, 2), (2, 1))

    bottom = np.zeros(len(df))
    hatches = ["", "xx"]  # avg_friends 用 //, potential_spread 用 xx
    labels = ["Avg Friends", "Potential Spread"]

    for metric, hatch, label in zip(["avg_friends", "potential_spread"], hatches, labels):
        bars = ax5.bar(df["genre"],
                       df[metric],
                       bottom=bottom,
                       color=[genre_colors[g] for g in df["genre"]],
                       alpha=0.9,
                       hatch=hatch,
                       label=label)
        bottom += df[metric]

        # 数值标签
        for bar, val in zip(bars, df[metric]):
            ax5.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + bar.get_y(),
                     f"{val:,.1f}",
                     ha="center", va="bottom", fontsize=7)

    ax5.set_title("Social Influence Metrics")
    ax5.set_ylabel("Count")
    ax5.set_xticklabels([g.title() for g in df["genre"]], rotation=45, ha="right")
    ax5.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {out_file}")


# ---------- 3. 主程序 ----------
if __name__ == "__main__":
    with open("analysis_report.txt", "r", encoding="utf-8") as f:
        report_text = f.read()

    df_genre = parse_report(report_text)
    if df_genre.empty:
        raise ValueError("No data parsed – please check the format of analysis_report.txt")

    plot_genre_comparison(df_genre)
