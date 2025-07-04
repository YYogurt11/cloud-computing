# ===================================================================
# 0. 环境修复与设置
# ===================================================================
import os
import warnings
from matplotlib import font_manager
import matplotlib.pyplot as plt

# 【代理修复】
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
# 【警告优化】
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================================================================
# 1. 导入库
# ===================================================================
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import re
from tqdm import tqdm
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI

# ===================================================================
# 2. 字体与路径配置
# ===================================================================
# ！！！请在这里再次确认您已替换为在“字体册”中找到的真实中文字体路径！！！
FONT_PATH_MAC = "/System/Library/Fonts/STHeiti Medium.ttc"  # 这是一个示例路径，请务必替换

try:
    font_manager.fontManager.addfont(FONT_PATH_MAC)
    plt.rcParams['font.sans-serif'] = font_manager.FontProperties(fname=FONT_PATH_MAC).get_name()
    print(f"Matplotlib字体已成功设置为: {font_manager.FontProperties(fname=FONT_PATH_MAC).get_name()}")
except FileNotFoundError:
    print(f"严重警告：在 '{FONT_PATH_MAC}' 无法找到字体文件。图表和词云中的中文可能无法显示。")
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# (修改) 项目路径和文件名
DATA_PATH = '/Users/yogurtsmacbook/Cloudcomputing/'
INPUT_FILENAME = 'news_dataset_cleaned_1000.csv'
OUTPUT_PREFIX = 'lda_results_1000'

# Ollama 连接参数
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL_NAME = "deepseek-r1"


# ===================================================================
# 3. 函数定义模块 (无变化)
# ===================================================================
def preprocess_text(text):
    if not isinstance(text, str): return []
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.lower().split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def find_optimal_topics(dictionary, corpus, texts, limit=16):  # (修改) 扩大主题搜索范围
    coherence_values, model_list = [], []
    for num_topics in tqdm(range(2, limit), desc="正在寻找最优主题数"):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, limit), coherence_values, marker='o')
    plt.title('主题数量与一致性得分关系图', fontsize=16)
    plt.xlabel('主题数量', fontsize=12)
    plt.ylabel('一致性得分', fontsize=12)
    plt.xticks(range(2, limit))
    plt.grid(True)
    plt.savefig(os.path.join(DATA_PATH, f'{OUTPUT_PREFIX}_coherence_scores.png'))
    plt.show()
    best_result_index = np.argmax(coherence_values)
    optimal_model = model_list[best_result_index]
    optimal_num_topics = range(2, limit)[best_result_index]
    print(f"分析完成：最佳主题数量为 {optimal_num_topics}，一致性得分为 {max(coherence_values):.4f}")
    return optimal_model, optimal_num_topics


def interpret_topics_with_llm(lda_model, num_topics):
    """调用本地Ollama大模型来解释LDA主题（增强版解析逻辑）。"""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    topics_keywords = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
    topic_interpretations = []

    for topic_id, keywords_probs in tqdm(topics_keywords, desc="正在调用大模型解读主题"):
        keywords = ", ".join([word for word, prob in keywords_probs])

        prompt = f"""你是一位数据分析师。以下是通过LDA模型从新闻文本中提取出的一个主题的关键词列表：
关键词："{keywords}"

请完成两项任务：
1. 为这个主题起一个精炼、概括性的中文标题。
2. 用一句话简要解释这个主题的核心内容。

请严格按照以下格式返回：
标题：[你的标题]
解释：[你的解释]
"""
        title = "解析失败"
        explanation = "未知错误"

        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            interpretation = response.choices[0].message.content.strip()

            # --- 全新的、更强大的解析逻辑 ---
            # 尝试用正则表达式精确匹配
            title_match = re.search(r"标题：(.*?)(?:\n|解释：|$)", interpretation, re.DOTALL)
            explanation_match = re.search(r"解释：(.*?)$", interpretation, re.DOTALL)

            if title_match:
                title = title_match.group(1).strip()

            if explanation_match:
                explanation = explanation_match.group(1).strip()
            else:
                # 如果找不到“解释：”，但找到了“标题：”，则将标题之后的所有内容作为解释
                if title_match:
                    explanation = interpretation.split(title_match.group(0), 1)[-1].strip()
                # 如果两者都找不到，则将模型的整个回答作为解释
                else:
                    explanation = interpretation

        except Exception as e:
            # 捕获所有可能的异常，包括连接错误和解析错误
            print(f"主题{topic_id+1}处理失败: {e}")
            explanation = str(e) if str(e) else "调用或解析Ollama时出错"

        topic_interpretations.append({
            "主题ID": f"主题 {topic_id+1}",
            "关键词": keywords,
            "大模型解读标题": title,
            "大模型解读解释": explanation
        })

    return pd.DataFrame(topic_interpretations)


# ===================================================================
# 4. 主工作流程
# ===================================================================
def main():
    # 检查和下载NLTK资源
    try:
        stopwords.words('english')
        WordNetLemmatizer().lemmatize('test')
    except LookupError:
        nltk.download('stopwords', quiet=True);
        nltk.download('wordnet', quiet=True)

    df = pd.read_csv(os.path.join(DATA_PATH, INPUT_FILENAME))
    if 'text_for_lda' not in df.columns:
        tqdm.pandas(desc="正在预处理文本")
        df['Processed_Tokens'] = df['cleaned_text'].progress_apply(preprocess_text)
    else:
        tqdm.pandas(desc="正在分词")
        df['Processed_Tokens'] = df['text_for_lda'].progress_apply(lambda x: x.split() if isinstance(x, str) else [])

    processed_texts = df['Processed_Tokens'].tolist()

    print("正在创建词典和语料库...")
    dictionary = Dictionary(processed_texts)
    # (修改) 调整过滤参数以适应更大的数据集
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    lda_optimal_model, optimal_num_topics = find_optimal_topics(dictionary, corpus, processed_texts, limit=16)

    print("正在生成可视化结果...")
    lda_vis = pyLDAvis.gensim_models.prepare(lda_optimal_model, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(lda_vis, os.path.join(DATA_PATH, f'{OUTPUT_PREFIX}_visualization.html'))
    print("pyLDAvis交互图已保存。")

    plt.figure(figsize=(16, 8 * ((optimal_num_topics + 1) // 2)))
    for i in range(optimal_num_topics):
        plt.subplot((optimal_num_topics + 1) // 2, 2, i + 1)
        topic_words = dict(lda_optimal_model.show_topic(i, topn=20))
        try:
            wordcloud = WordCloud(width=800, height=500, background_color='white',
                                  font_path=FONT_PATH_MAC).generate_from_frequencies(topic_words)
            plt.imshow(wordcloud, interpolation='bilinear')
        except Exception as e:
            print(f"生成词云图时出错: {e}")
            plt.text(0.5, 0.5, "词云图生成失败", horizontalalignment='center')
        plt.title(f'主题 {i + 1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_PATH, f'{OUTPUT_PREFIX}_wordclouds.png'), dpi=300)
    plt.close()
    print("主题词云图已保存。")

    df_interpretations = interpret_topics_with_llm(lda_optimal_model, optimal_num_topics)
    print("\n--- 大模型主题解读结果 ---")
    print(df_interpretations)
    df_interpretations.to_excel(os.path.join(DATA_PATH, f'{OUTPUT_PREFIX}_interpretations.xlsx'), index=False)
    print("主题解读结果已保存到Excel文件。")


if __name__ == '__main__':
    main()