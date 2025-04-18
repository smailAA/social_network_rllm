from openai import OpenAI
from information_generation import  *
# 该文件使用最基础的提示模板调用大模型进行分类，算是方法1

# 调用deepseek-v3
client = OpenAI(api_key="your_api_key", base_url="http://111.186.56.172:3000/v1")
# client = OpenAI(api_key="your_api_key")
# 候选音乐流派列表
MUSIC_GENRES = [
    "electronic", "rock", "metal", "pop", "jazz",
    "soul", "punk", "hip-hop", "reggae", "latin",
    "country"  # 保留未知选项
]


def classify_music_genre(artist_id):
    """
    根据艺术家ID预测音乐流派

    参数:
        artist_id: 艺术家的唯一标识符

    返回:
        预测的流派名称或"Unknown"
    """
    # 1. 生成艺术家描述
    description = build_artist_description(artist_id)

    # 2. 准备分类提示
    prompt = f"""
你是一个音乐流派分类专家，需要根据艺术家特征和听众行为预测流派。

----- 输入数据 -----
{description}

----- 候选流派 -----
{", ".join(MUSIC_GENRES[:-1])}

----- 任务要求 -----
1. 只输出最可能的流派名称，不要解释
2. 必须从候选列表中选择
"""

    # 3. 调用API进行分类
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            # model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个专业的音乐流派分类器"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 降低随机性
            max_tokens=20,
            stream=False
        )

        # 4. 处理并验证响应
        predicted_genre = response.choices[0].message.content.strip()
        return predicted_genre if predicted_genre in MUSIC_GENRES else "Unknown"

    except Exception as e:
        print(f"分类出错: {str(e)}")
        return "Unknown"


def evaluate_classification_accuracy(artists_df: pd.DataFrame, sample_size: int = 500) -> dict[str, float]:
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
    pd.DataFrame(evaluation_results['details']).to_csv("simple_prompt_result_500.csv", index=False)
    print("\n详细结果已保存到 simple_prompt_result_500.csv")