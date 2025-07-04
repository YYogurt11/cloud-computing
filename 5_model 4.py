# 文件名: part3_final_analysis_v2.py

import pandas as pd
import os
import re
from tqdm import tqdm
from openai import OpenAI
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

DATA_PATH = '/Users/yogurtsmacbook/Cloudcomputing/Dataset/'
INPUT_FILENAME = 'news_dataset_cleaned_100.csv'
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL_NAME = "deepseek-r1"


# (核心修改) 使用一个新的、更鲁棒的Prompt
def get_sentiment_label(text: str, client: OpenAI) -> str:
    prompt = f"""
    你的任务是分析以下新闻文本的情感。请遵循以下步骤：
    1.  阅读新闻内容，识别其中带有感情色彩的词汇或句子（例如，积极的词如'成功'、'创纪录'，消极的词如'震惊'、'暴力'、'危机'）。
    2.  根据这些带有感情色彩的表达，综合判断整篇文章是倾向于“正面”（Positive），“负面”（Negative），还是“中立”（Neutral）。
    3.  你的最终回答必须是 "Positive"、"Negative" 或 "Neutral" 这三个词中的一个，不要包含任何其他文字或解释。

    新闻内容：
    ---
    {text}
    ---
    最终分类：
    """
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()

        # 使用更严格的匹配，并处理大小写问题
        if "positive" in result.lower(): return "Positive"
        if "negative" in result.lower(): return "Negative"
        return "Neutral"
    except Exception:
        return "Neutral"


def get_topic_labels(df: pd.DataFrame, num_topics: int = 5) -> pd.DataFrame:
    print("开始为数据集生成主题标签...")

    def preprocess_for_lda(text):
        if not isinstance(text, str): return []
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.lower().split()
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    processed_texts = df['cleaned_text'].apply(preprocess_for_lda).tolist()
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    print(f"正在训练LDA模型 (k={num_topics})...")
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)
    doc_topics = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    dominant_topics = [sorted(topics, key=lambda x: x[1], reverse=True)[0][0] + 1 if topics else 0 for topics in
                       doc_topics]
    df['dominant_topic'] = dominant_topics
    print("主题标签生成完毕。")
    return df


def main():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True);
        nltk.download('wordnet', quiet=True)

    input_path = os.path.join(DATA_PATH, INPUT_FILENAME)
    print(f"加载数据: {input_path}")
    df = pd.read_csv(input_path)

    df = get_topic_labels(df, num_topics=5)

    print("\n开始为每条新闻生成情感标签（此步骤耗时较长）...")
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    tqdm.pandas(desc="情感分析进度")
    df['sentiment'] = df['cleaned_text'].progress_apply(get_sentiment_label, client=client)
    print("情感特征已生成。")
    print("情感分布情况：")
    print(df['sentiment'].value_counts())

    # 检查情感特征是否有多样性
    if df['sentiment'].nunique() < 2:
        print("\n严重警告：情感特征缺乏多样性，所有或大部分样本被归为同一类。")
        print("这会导致模型性能不佳。请检查您的LLM或Prompt。")
        return

    print("\n进行特征工程与融合...")
    topic_dummies = pd.get_dummies(df['dominant_topic'], prefix='topic')
    sentiment_dummies = pd.get_dummies(df['sentiment'], prefix='sentiment')
    X_fused = pd.concat([topic_dummies, sentiment_dummies], axis=1)
    y = df['label_numeric']

    X_train, X_test, y_train, y_test = train_test_split(X_fused, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
    }

    results = []
    for name, model in models.items():
        print(f"--- 正在训练: {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append({"Model": name, "Accuracy": accuracy, "F1-Score": f1})

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("     最终多模态模型（主题+情感）性能对比")
    print("=" * 50)
    print(results_df)
    print("=" * 50)


if __name__ == '__main__':
    main()