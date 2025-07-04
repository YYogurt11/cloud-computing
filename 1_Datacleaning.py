import pandas as pd
import re
import os

# ===================================================================
# 1. 设置文件路径和文件名
# ===================================================================
# 您的数据集所在的文件夹路径
DATA_PATH = '/Users/yogurtsmacbook/Cloudcomputing/'
# 您的原始Excel文件名
INPUT_FILENAME = 'Dataset.xlsx'

# 定义输出文件名
OUTPUT_FULL_CLEANED = 'news_dataset_cleaned_full.csv'
OUTPUT_SUBSET_100 = 'news_dataset_cleaned_100.csv'
OUTPUT_SUBSET_1000 = 'news_dataset_cleaned_1000.csv'

input_file_path = os.path.join(DATA_PATH, INPUT_FILENAME)


# ===================================================================
# 2. 定义数据清洗函数 (与之前相同)
# ===================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s*\([^)]*\)\s*', ' ', text)
    text = re.sub(r'^[A-Z\s]+\s*\(Reuters\)\s*-\s*', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ===================================================================
# 3. 主清洗与子集划分流程
# ===================================================================
def main():
    print(f"正在从 '{input_file_path}' 加载Excel数据...")

    try:
        # (修改) 使用 read_excel 读取数据
        df = pd.read_excel(input_file_path)
    except FileNotFoundError:
        print(f"错误：文件未找到！请确认您的文件名 '{INPUT_FILENAME}' 是否正确。")
        return
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return

    print("数据加载成功，开始清洗...")

    # 清洗流程 (与之前相同)
    df.dropna(subset=['text', 'label'], inplace=True)
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].astype(str).str.strip()
    df['label_numeric'] = df['label'].apply(lambda x: 1 if x.lower() == 'real' else 0)
    df_cleaned = df[['cleaned_text', 'label_numeric', 'label']].copy()

    # 保存完整的清洗后数据集
    full_output_path = os.path.join(DATA_PATH, OUTPUT_FULL_CLEANED)
    df_cleaned.to_csv(full_output_path, index=False, encoding='utf-8')
    print(f"完整的清洗后数据集 ({len(df_cleaned)}条) 已保存至: {OUTPUT_FULL_CLEANED}")

    # --- (新增) 创建均衡的子集 ---
    df_real = df_cleaned[df_cleaned['label_numeric'] == 1]
    df_fake = df_cleaned[df_cleaned['label_numeric'] == 0]

    print(f"\n数据集中包含 {len(df_real)} 条真新闻和 {len(df_fake)} 条假新闻。")
    print("正在创建均衡子集...")

    # 创建 N=100 的子集
    if len(df_real) >= 50 and len(df_fake) >= 50:
        sample_real_100 = df_real.sample(n=50, random_state=42)
        sample_fake_100 = df_fake.sample(n=50, random_state=42)
        df_subset_100 = pd.concat([sample_real_100, sample_fake_100]).sample(frac=1, random_state=42).reset_index(
            drop=True)

        subset_100_path = os.path.join(DATA_PATH, OUTPUT_SUBSET_100)
        df_subset_100.to_csv(subset_100_path, index=False, encoding='utf-8')
        print(f"N=100 的均衡子集已保存至: {OUTPUT_SUBSET_100}")
    else:
        print("警告：数据不足，无法创建N=100的均衡子集。")

    # 创建 N=1000 的子集
    if len(df_real) >= 500 and len(df_fake) >= 500:
        sample_real_1000 = df_real.sample(n=500, random_state=42)
        sample_fake_1000 = df_fake.sample(n=500, random_state=42)
        df_subset_1000 = pd.concat([sample_real_1000, sample_fake_1000]).sample(frac=1, random_state=42).reset_index(
            drop=True)

        subset_1000_path = os.path.join(DATA_PATH, OUTPUT_SUBSET_1000)
        df_subset_1000.to_csv(subset_1000_path, index=False, encoding='utf-8')
        print(f"N=1000 的均衡子集已保存至: {OUTPUT_SUBSET_1000}")
    else:
        print("警告：数据不足，无法创建N=1000的均衡子集。")


if __name__ == '__main__':
    main()