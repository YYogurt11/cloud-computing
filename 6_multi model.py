import pandas as pd
import os
import re
from tqdm import tqdm
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===================================================================
# 1. 配置与函数定义
# ===================================================================
# 设置Matplotlib中文字体 (确保此路径在您的系统上有效)
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 文件与Ollama模型配置
DATA_PATH = '/Users/yogurtsmacbook/Cloudcomputing/'
INPUT_FILENAME = 'news_dataset_1000_with_features.csv'
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL_NAME = "deepseek-r1"

# 设置代理（以防万一）
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'


def get_sentiment(news_text: str) -> str:
    """使用本地LLM为文本打上情感标签"""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    prompt = f"""请分析以下新闻文本的情感倾向。只回答“正面”、“负面”或“中立”。\n\n新闻内容：\n{news_text}"""
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()
        if "正面" in result: return "正面"
        if "负面" in result: return "负面"
        return "中立"
    except Exception as e:
        print(f"情感分析API调用失败: {e}")
        return "中立"  # 出错时默认为中立


# ===================================================================
# 2. 主流程
# ===================================================================
def main():
    # --- 步骤1: 加载数据并生成情感特征 ---
    input_path = os.path.join(DATA_PATH, INPUT_FILENAME)
    print(f"正在加载数据: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{INPUT_FILENAME}'。请确保第二部分的脚本已成功运行。")
        return

    print("数据加载成功。开始为每条新闻生成情感标签...")
    tqdm.pandas(desc="正在进行情感分析")
    df['sentiment'] = df['cleaned_text'].progress_apply(get_sentiment)

    print("情感特征已生成。")
    print(df['sentiment'].value_counts())  # 查看情感分布

    # --- 步骤2: 特征工程与融合 ---
    print("\n正在进行特征工程（独热编码）...")

    # 对'dominant_topic'进行独热编码
    topic_dummies = pd.get_dummies(df['dominant_topic'], prefix='topic')

    # 对'sentiment'进行独热编码
    sentiment_dummies = pd.get_dummies(df['sentiment'], prefix='sentiment')

    # 特征融合：将两个独热编码后的DataFrame拼接起来
    X_fused = pd.concat([topic_dummies, sentiment_dummies], axis=1)

    # 我们的目标变量 y 是新闻的真伪标签
    y = df['label_numeric']

    print("特征融合完成。融合后的特征矩阵维度:", X_fused.shape)
    print("部分融合特征展示:")
    print(X_fused.head())

    # --- 步骤3: 模型训练与评估 ---
    print("\n正在划分训练集和测试集...")
    # 80%的数据用于训练，20%用于测试
    X_train, X_test, y_train, y_test = train_test_split(
        X_fused, y, test_size=0.2, random_state=42, stratify=y
    )

    print("开始训练分类模型...")
    # 我们选用随机森林分类器，它通常表现稳健
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("模型训练完成。")

    print("\n--- 模型性能评估 ---")
    y_pred = model.predict(X_test)

    # 打印准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"多模态融合模型在测试集上的准确率: {accuracy:.4f}")

    # 打印详细的分类报告
    print("\n分类报告 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Fake News (0)', 'Real News (1)']))

    # 绘制混淆矩阵
    print("混淆矩阵 (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测为假', '预测为真'],
                yticklabels=['实际为假', '实际为真'])
    plt.title('多模态融合模型混淆矩阵', fontsize=16)
    plt.ylabel('实际标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.show()


if __name__ == '__main__':
    main()