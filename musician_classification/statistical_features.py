
import os
import pandas as pd
from collections import defaultdict

# 获取当前文件的父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 构建文件路径
artists_path = os.path.join(parent_dir, "data", "tlf2k", "raw", "artists.csv")
user_artist_path = os.path.join(parent_dir, "data", "tlf2k", "raw", "user_artists.csv")
user_friends_path = os.path.join(parent_dir, "data", "tlf2k", "raw", "user_friends.csv")

# 加载表格
artists_df = pd.read_csv(artists_path)
user_artist_df = pd.read_csv(user_artist_path)
user_friends_df = pd.read_csv(user_friends_path)


import os
import pandas as pd
from collections import defaultdict

# 获取当前文件的父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 构建文件路径
artists_path = os.path.join(parent_dir, "data", "tlf2k", "raw", "artists.csv")
user_artist_path = os.path.join(parent_dir, "data", "tlf2k", "raw", "user_artists.csv")
user_friends_path = os.path.join(parent_dir, "data", "tlf2k", "raw", "user_friends.csv")

# 加载表格
artists_df = pd.read_csv(artists_path)
user_artist_df = pd.read_csv(user_artist_path)
user_friends_df = pd.read_csv(user_friends_path)


def analyze_genre_stats(genre):
    """分析特定流派的音乐家统计数据"""
    try:
        # 获取该流派的所有音乐家
        genre_artists = artists_df[artists_df["label"] == genre]
        if len(genre_artists) == 0:
            return f"没有找到'{genre}'流派的音乐家"

        artist_ids = genre_artists["artistID"].tolist()

        # 初始化统计数据
        stats = {
            "total_artists": len(genre_artists),
            "total_listeners": 0,
            "total_plays": 0,
            "avg_listeners_per_artist": 0,
            "avg_plays_per_artist": 0,
            "avg_plays_per_listener_per_artist": 0,
            "artist_stats": [],
            "social_stats": {
                "avg_friends": 0,
                "friend_coverage": 0,
                "clustering_coeff": 0,
                "potential_spread": 0
            }
        }

        # 初始化流派级别的统计
        avg_plays_25 = []
        avg_plays_50 = []
        avg_plays_75 = []
        avg_plays_max = []
        avg_top_user_ratio = []
        avg_core_listeners = []

        # 分析每个音乐家
        for artist_id in artist_ids:
            artist = genre_artists[genre_artists["artistID"] == artist_id].iloc[0]
            listeners = user_artist_df[user_artist_df["artistID"] == artist_id]

            # 基础统计
            num_listeners = len(listeners)
            total_plays = listeners["weight"].sum()
            avg_plays = listeners["weight"].mean()

            # 如果没有听众数据，设置默认值
            if num_listeners == 0:
                play_stats = {"25%": 0, "50%": 0, "75%": 0, "max": 0}
            else:
                play_stats = listeners["weight"].describe(percentiles=[.25, .5, .75])

            # 核心听众分析
            core_listeners = len(listeners[listeners["weight"] > avg_plays * 2]) / max(1, num_listeners)

            # 社交网络特征
            if num_listeners > 0:
                listener_ids = set(listeners["userID"])
                all_friends = user_friends_df[user_friends_df["userID"].isin(listener_ids)]
                unique_friends = set(all_friends["friendID"])

                # 计算社交指标
                friend_coverage = len(unique_friends & listener_ids) / max(1, len(unique_friends))
                user_friend_counts = user_friends_df["userID"].value_counts()
                listener_friend_counts = listeners["userID"].map(user_friend_counts).fillna(0)  # 填充缺失值为0
                avg_friends = listener_friend_counts.mean()
                potential_spread = len(unique_friends - listener_ids)
                listener_pairs = len(listener_ids) * (len(listener_ids) - 1) / 2
                actual_connections = len(all_friends[all_friends["friendID"].isin(listener_ids)])
                clustering_coeff = actual_connections / max(1, listener_pairs)
            else:
                friend_coverage = avg_friends = potential_spread = clustering_coeff = 0

            # 保存单个音乐家统计
            artist_stat = {
                "artist_id": artist_id,
                "name": artist["name"],
                "listeners": num_listeners,
                "plays": total_plays,
                "avg_plays_per_listener": total_plays / max(1, num_listeners),
                "play_stats": play_stats,
                "core_listeners": core_listeners,
                "friend_coverage": friend_coverage,
                "avg_friends": avg_friends,
                "potential_spread": potential_spread,
                "clustering_coeff": clustering_coeff
            }
            stats["artist_stats"].append(artist_stat)

            # 更新流派级别的统计
            avg_plays_25.append(play_stats["25%"])
            avg_plays_50.append(play_stats["50%"])
            avg_plays_75.append(play_stats["75%"])
            avg_plays_max.append(play_stats["max"])
            avg_top_user_ratio.append(len(listeners[listeners["weight"] > play_stats["75%"]]) / max(1, num_listeners))
            avg_core_listeners.append(core_listeners)

        # 计算平均值
        if stats["total_artists"] > 0:
            stats["avg_listeners_per_artist"] = sum(artist["listeners"] for artist in stats["artist_stats"]) / stats["total_artists"]
            stats["avg_plays_per_artist"] = sum(artist["plays"] for artist in stats["artist_stats"]) / stats["total_artists"]
            stats["avg_plays_per_listener_per_artist"] = sum(artist["avg_plays_per_listener"] for artist in stats["artist_stats"]) / stats["total_artists"]
            stats["social_stats"]["avg_friends"] = sum(artist["avg_friends"] for artist in stats["artist_stats"]) / stats["total_artists"]
            stats["social_stats"]["friend_coverage"] = sum(artist["friend_coverage"] for artist in stats["artist_stats"]) / stats["total_artists"]
            stats["social_stats"]["clustering_coeff"] = sum(artist["clustering_coeff"] for artist in stats["artist_stats"]) / stats["total_artists"]
            stats["social_stats"]["potential_spread"] = sum(artist["potential_spread"] for artist in stats["artist_stats"]) / stats["total_artists"]

        # 计算流派级别的平均统计
        stats["avg_plays_25"] = sum(avg_plays_25) / len(avg_plays_25) if avg_plays_25 else 0
        stats["avg_plays_50"] = sum(avg_plays_50) / len(avg_plays_50) if avg_plays_50 else 0
        stats["avg_plays_75"] = sum(avg_plays_75) / len(avg_plays_75) if avg_plays_75 else 0
        stats["avg_plays_max"] = sum(avg_plays_max) / len(avg_plays_max) if avg_plays_max else 0
        stats["avg_top_user_ratio"] = sum(avg_top_user_ratio) / len(avg_top_user_ratio) if avg_top_user_ratio else 0
        stats["avg_core_listeners"] = sum(avg_core_listeners) / len(avg_core_listeners) if avg_core_listeners else 0

        # 计算流派占比
        total_artists = len(artists_df)
        stats["genre_proportion"] = stats["total_artists"] / total_artists

        return stats

    except Exception as e:
        return f"分析流派时出错: {str(e)}"


def generate_genre_report(genre_stats, genre):
    """生成流派分析报告"""
    report = f"""
流派分析报告: {genre.upper()}
====================================

基本信息:
--------
该流派在全部音乐家流派中的占比: {genre_stats['genre_proportion']:.1%}
平均每个音乐家的听众数: {genre_stats.get('avg_listeners_per_artist', 0):.1f}
平均每个音乐家的总收听次数: {genre_stats.get('avg_plays_per_artist', 0):.1f}
平均每个音乐家的听众平均收听次数: {genre_stats.get('avg_plays_per_listener_per_artist', 0):.1f}
平均最高收听: {genre_stats.get('avg_plays_max', 0):,}次 (平均头部用户占比{genre_stats.get('avg_top_user_ratio', 0):.1%})
平均核心听众比例: {genre_stats.get('avg_core_listeners', 0):.1%} (收听>平均2倍)

社交网络特征:
----------
平均听众好友覆盖率: {genre_stats['social_stats']['friend_coverage']:.1%} 
平均好友数: {genre_stats['social_stats']['avg_friends']:.1f}
平均潜在传播对象: {genre_stats['social_stats']['potential_spread']:,}
平均听众聚类系数: {genre_stats['social_stats']['clustering_coeff']:.3f}

"""

    return report


# 示例使用
if __name__ == "__main__":
    MUSIC_GENRES = [
        "electronic", "rock", "metal", "pop", "jazz",
        "soul", "punk", "hip-hop", "reggae", "latin",
        "country"  # 保留未知选项
    ]
    for genre in MUSIC_GENRES:
        stats = analyze_genre_stats(genre)

        if isinstance(stats, str):
            print(stats)  # 打印错误信息
        else:
            report = generate_genre_report(stats, genre)
            print(report)

            # 可选: 保存报告到文件
            with open(f"analysis_report.txt", "a", encoding="utf-8") as f:
                f.write(report)

