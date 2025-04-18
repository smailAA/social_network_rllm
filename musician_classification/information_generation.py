# 该文件用来生成每个音乐家的描述信息，包括基本信息、收听情况、听众画像等等
import os
import pandas as pd

# 获取当前文件的父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 构建文件路径
artists_path = os.path.join(parent_dir, "data","tlf2k","raw", "artists.csv")
user_artist_path = os.path.join(parent_dir, "data","tlf2k","raw", "user_artists.csv")
user_friends_path = os.path.join(parent_dir, "data","tlf2k","raw", "user_friends.csv")

# 加载表格
artists_df = pd.read_csv(artists_path)  # 音乐家特征
user_artist_df = pd.read_csv(user_artist_path)  # 用户-音乐家互动
user_friends_df = pd.read_csv(user_friends_path)  # 用户社交关系


def build_artist_description(artist_id):
    try:
        # 获取艺术家信息
        artist = artists_df[artists_df["artistID"] == artist_id].iloc[0]

        # 基础听众统计
        listeners = user_artist_df[user_artist_df["artistID"] == artist_id]
        num_listeners = len(listeners)
        total_plays = listeners["weight"].sum()
        avg_plays = listeners["weight"].mean()

        # 新增特征1：收听次数分布统计
        play_stats = listeners["weight"].describe(percentiles=[.25, .5, .75])

        # 新增特征2：核心听众比例（收听次数>平均值的2倍）
        core_listeners = len(listeners[listeners["weight"] > 2 * avg_plays]) / max(1, num_listeners)

        # 社交网络特征（充分利用user_friends）
        if num_listeners > 0:
            listener_ids = set(listeners["userID"])

            # 新增特征3：听众好友覆盖率（有多少比例的好友也是听众）
            all_friends = user_friends_df[user_friends_df["userID"].isin(listener_ids)]
            unique_friends = set(all_friends["friendID"])
            friend_coverage = len(unique_friends & listener_ids) / max(1, len(unique_friends))

            # 新增特征4：听众社交影响力（听众的平均好友数）
            user_friend_counts = user_friends_df["userID"].value_counts()
            avg_friends = listeners["userID"].map(user_friend_counts).mean()

            # 新增特征5：社交传播潜力（听众的好友中非听众的数量）
            potential_spread = len(unique_friends - listener_ids)

            # 新增特征6：社交聚类系数（听众之间互相是好友的比例）
            listener_pairs = len(listener_ids) * (len(listener_ids) - 1) / 2
            actual_connections = len(all_friends[all_friends["friendID"].isin(listener_ids)])
            clustering_coeff = actual_connections / max(1, listener_pairs)
        else:
            friend_coverage = avg_friends = potential_spread = clustering_coeff = 0

        # 构建描述文本
        description = f"""
音乐家信息:
------------
名称: {artist['name']}
类型: {artist['type']}
出生日期: {artist.get('born', '未知')}
活跃时期: {artist['yearsActive']}
地区: {artist.get('location', '未知')}
简介: {artist.get('biography', '暂无简介')}
更多信息: {artist.get('url', '无')}

基础听众统计:
--------
听众数: {num_listeners:,}
总收听次数: {total_plays:,}
听众平均收听次数: {avg_plays:.1f}
25%用户收听 ≤ {play_stats['25%']:.0f}次
50%用户收听 ≤ {play_stats['50%']:.0f}次
75%用户收听 ≤ {play_stats['75%']:.0f}次
最高收听: {play_stats['max']:,}次 (头部用户占比{len(listeners[listeners['weight'] > play_stats['75%']])/len(listeners):.1%})
核心听众比例: {core_listeners:.1%} (收听>平均2倍)

社交网络特征:
----------
听众好友覆盖率: {friend_coverage:.1%} 
平均好友数: {avg_friends:.1f}
潜在传播对象: {potential_spread:,}
听众聚类系数: {clustering_coeff:.3f}


"""

        return description

    except Exception as e:
        return f"生成描述时出错: {str(e)}"



if __name__ == '__main__':
    description = build_artist_description(87)
    print(description)
