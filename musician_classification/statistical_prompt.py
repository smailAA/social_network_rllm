# 该文件在方法2投票机制的基础上，提前提供给大模型各个流派的统计特征，也就是领域知识
# 算方法3
from openai import OpenAI
from information_generation import  *
from collections import Counter

# 该文件在方法1的基础上增加了投票机制，算是方法2
# 调用deepseek-v3
client = OpenAI(api_key="your_api_key", base_url="http://111.186.56.172:3000/v1")
# client = OpenAI(api_key="your_api_key")

# 候选音乐流派列表
MUSIC_GENRES = [
    "electronic", "rock", "metal", "pop", "jazz",
    "soul", "punk", "hip-hop", "reggae", "latin",
    "country"  # 保留未知选项
]

import os
from collections import Counter

# 假设 MUSIC_GENRES 是一个全局变量，包含所有可能的流派
MUSIC_GENRES = ["rock", "pop", "jazz", "classical", "electronic", "hip-hop", "country", "folk", "reggae", "metal"]


def read_analysis_report(file_path: str) -> str:
    """
    从文件中读取领域知识
    """
    if not os.path.exists(file_path):
        return "领域知识文件未找到，请检查路径。"
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def classify_music_genre(artist_id: int) -> str:
    """
    基于投票机制的音乐流派分类器

    参数:
        artist_id: 艺术家的唯一标识符

    返回:
        预测的流派名称或"Unknown"
    """
    # 读取领域知识
    analysis_report = read_analysis_report("analysis_report.txt")

    # 1. 生成艺术家描述
    description = build_artist_description(artist_id)

    # 2. 准备三种视角的提示模板
    prompts = {
        "analyst": f"""
作为音乐数据分析专家，请基于以下统计特征判断流派：
----- 领域知识 -----
{analysis_report}

----- 数据摘要 -----
{description}

----- 分析要求 -----
1. 重点关注听众统计和社交网络特征
2. 必须从候选列表中选择
3. 只输出最可能的流派名称,不要解释
候选流派: {", ".join(MUSIC_GENRES)}
""",

        "fan": f"""
作为拥有20年听歌经验的资深乐迷，请根据艺术家的背景信息判断流派：

----- 艺术家档案 -----
{description}

----- 判断依据 -----
1. 基于音乐风格、活跃时期和地域特征
2. 必须从候选列表中选择
3. 只输出最可能的流派名称,不要解释
可选流派: {", ".join(MUSIC_GENRES)}
""",

        "industry": f"""
作为唱片公司A&R总监，请从商业角度评估该音乐家所属流派：

----- 市场数据 -----
{description}

----- 评估标准 -----
1. 考虑厂牌、听众画像和商业潜力
2. 必须从候选列表中选择
3. 只输出最可能的流派名称,不要解释
可选分类: {", ".join(MUSIC_GENRES)}
"""
    }

    # 3. 多视角预测
    votes = []
    role_votes = {}  # 新增：记录每个角色的投票
    for role, prompt in prompts.items():
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                # model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"你是一个{role}视角的音乐流派分类器"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,  # 适度创造性
                max_tokens=15,
                stream=False
            )
            vote = response.choices[0].message.content.strip()
            if vote in MUSIC_GENRES:
                votes.append(vote)
                role_votes[role] = vote  # 记录角色对应的投票
                print(f"{role}投票: {vote}")  # 调试用
        except Exception as e:
            print(f"{role}视角预测失败: {str(e)}")
            continue

    # 4. 投票决策
    if not votes:
        return "Unknown"

    vote_counts = Counter(votes)

    # 优先选择至少两票的流派
    if len(vote_counts) < 3:
        return vote_counts.most_common(1)[0][0]

    # 三票分散时选择分析师投票
    if "analyst" in role_votes:
        return role_votes["analyst"]

    # 如果没有分析师投票，返回第一个投票
    return votes[0] if votes else "Unknown"



def evaluate_classification_accuracy(artists_df: pd.DataFrame, sample_size: int = 1000) -> dict[str, float]:
    """
    评估模型分类准确率
    :param artists_df: 包含artistID和label的DataFrame
    :param sample_size: 抽样数量
    :return: 包含准确率指标的字典
    """
    # 随机抽样（确保有标签的艺术家）
    valid_artists = artists_df[artists_df['label'].notna() & artists_df['label'].isin(MUSIC_GENRES)]
    sampled_artists = valid_artists.sample(min(sample_size, len(valid_artists)), random_state=42)

    results = {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'unknown': 0,
        'details': []
    }

    for _, row in sampled_artists.iterrows():
        artist_id = row['artistID']
        true_label = row['label']
        predicted_label = classify_music_genre(artist_id)

        is_correct = predicted_label == true_label
        results['total'] += 1
        if predicted_label == "Unknown":
            results['unknown'] += 1
        elif is_correct:
            results['correct'] += 1
        else:
            results['incorrect'] += 1

        results['details'].append({
            'artistID': artist_id,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': is_correct
        })
        print(f"\n 当前艺术家id：{artist_id}")
        print(f"\n 真实标签： {true_label}")
        print(f"\n 预测类别： {predicted_label}")
        print(f"\n 是否正确： {is_correct}")

    # 计算准确率（排除Unknown预测）
    valid_predictions = results['correct'] + results['incorrect']
    results['accuracy'] = results['correct'] / valid_predictions if valid_predictions > 0 else 0
    results['coverage'] = 1 - (results['unknown'] / results['total']) if results['total'] > 0 else 0

    return results


if __name__ == "__main__":
    # 加载数据（假设artists_df已包含label列）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    artists_path = os.path.join(parent_dir, "data", "tlf2k", "raw", "artists.csv")
    artists_df = pd.read_csv(artists_path)

    # 评估模型
    evaluation_results = evaluate_classification_accuracy(artists_df)

    # 打印结果
    print(f"\n评估结果（共{evaluation_results['total']}个样本）：")
    print(f"准确率: {evaluation_results['accuracy']:.2%}")
    print(f"覆盖率: {evaluation_results['coverage']:.2%} (非Unknown预测比例)")
    print(
        f"正确: {evaluation_results['correct']} | 错误: {evaluation_results['incorrect']} | Unknown: {evaluation_results['unknown']}")

    # 保存详细结果
    pd.DataFrame(evaluation_results['details']).to_csv("statistical_vote_result.csv", index=False)
    print("\n详细结果已保存到 statistical_vote_result.csv")