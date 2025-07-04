import os

# 在脚本最开始设置环境变量，告诉程序不要使用代理
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
# 为了更保险，可以同时清空其他代理变量
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import os

# ===================================================================
# 1. 设置Ollama连接参数和文件路径
# ===================================================================
# Ollama 默认的API地址
OLLAMA_BASE_URL = "http://localhost:11434/v1"
# 您在Ollama中运行的模型名称
OLLAMA_MODEL_NAME = "deepseek-r1"

DATA_PATH = '/Users/yogurtsmacbook/Cloudcomputing/'
# 我们将使用N=1000的子集进行分析
DATASET_TO_ANALYZE = 'news_dataset_cleaned_20.csv'
dataset_path = os.path.join(DATA_PATH, DATASET_TO_ANALYZE)


# ===================================================================
# 2. 定义与本地Ollama模型交互的函数
# ===================================================================
def query_local_llm(prompt_text: str):
    """向本地部署的Ollama LLM发送请求。"""
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        response = client.chat.completions.create(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"连接Ollama时出错: {e}")
        print("请确保Ollama服务正在运行，并且模型 'deepseek-r1' 已经下载并可用。")
        print("您可以在终端运行 'ollama list' 来查看可用的模型列表。")
        return "连接失败"


# ===================================================================
# 3. 定义分析任务函数 (使用新的query_local_llm)
# ===================================================================
def predict_truthfulness(news_text: str) -> str:
    """任务1: 基础真假新闻判别"""
    prompt = f"""请判断以下新闻的真伪。只回答“真新闻”或“假新闻”。\n\n新闻内容：\n{news_text}"""
    result = query_local_llm(prompt)
    return "真新闻" if "真新闻" in result else ("假新闻" if "假新闻" in result else "无法判断")


def predict_truthfulness_with_sentiment(news_text: str) -> str:
    """任务3: 融合情感的真假新闻判别"""
    prompt = f"""请扮演一个新闻事实核查员，并特别关注新闻文本的情感色彩，因为假新闻常使用煽动性语言。综合判断后，只回答“真新闻”或“假新闻”。\n\n新闻内容：\n{news_text}"""
    result = query_local_llm(prompt)
    return "真新闻" if "真新闻" in result else ("假新闻" if "假新闻" in result else "无法判断")


# ===================================================================
# 4. 主分析流程
# ===================================================================
def analyze():
    print(f"正在加载待分析的数据集: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 为tqdm注册pandas apply
    tqdm.pandas(desc="任务1: 基础预测")
    df['pred_basic'] = df['cleaned_text'].progress_apply(predict_truthfulness)

    tqdm.pandas(desc="任务3: 融合情感预测")
    df['pred_sentiment_aware'] = df['cleaned_text'].progress_apply(predict_truthfulness_with_sentiment)

    # 保存带有预测结果的DataFrame
    output_path = os.path.join(DATA_PATH, 'analysis_results_1000.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"分析结果已保存至: {output_path}")

    # --- 结果评估 ---
    def calculate_accuracy(df_results, pred_col, label_col='label'):
        df_filtered = df_results[df_results[pred_col] != "无法判断"].copy()

        # 将文本标签统一为 "真新闻" 和 "假新闻"
        df_filtered['label_text'] = df_filtered[label_col].apply(lambda x: "真新闻" if x == 1 else "假新闻")

        accuracy = accuracy_score(df_filtered['label_text'], df_filtered[pred_col])

        try:
            # 确保标签顺序正确
            cm = confusion_matrix(df_filtered['label_text'], df_filtered[pred_col], labels=['真新闻', '假新闻'])
            tn, fp, fn, tp = cm.ravel()
            accuracy_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 假新闻的检出率
            accuracy_true = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 真新闻的检出率
        except ValueError:
            print(f"在计算列 '{pred_col}' 的混淆矩阵时出错，可能因为模型只预测了一种类别。")
            return {"Accuracy": accuracy, "Accuracy_fake": "N/A", "Accuracy_true": "N/A"}

        return {"Accuracy": accuracy, "Accuracy_fake": accuracy_fake, "Accuracy_true": accuracy_true}

    results_basic = calculate_accuracy(df, 'pred_basic', 'label_numeric')
    results_sentiment_aware = calculate_accuracy(df, 'pred_sentiment_aware', 'label_numeric')

    print("\n--- 实验结果 ---")
    print(f"数据集: {DATASET_TO_ANALYZE}")
    print("\n基础预测模型:")
    for key, value in results_basic.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n融合情感的预测模型:")
    for key, value in results_sentiment_aware.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n--- 作业要求的准确率 ---")
    print(
        f"Accuracy = {results_basic['Accuracy']:.4f} (基础模型) / {results_sentiment_aware['Accuracy']:.4f} (情感模型)")
    print(
        f"Accuracy_fake = {results_basic['Accuracy_fake']:.4f} (基础模型) / {results_sentiment_aware['Accuracy_fake']:.4f} (情感模型)")
    print(
        f"Accuracy_true = {results_basic['Accuracy_true']:.4f} (基础模型) / {results_sentiment_aware['Accuracy_true']:.4f} (情感模型)")


if __name__ == '__main__':
    analyze()