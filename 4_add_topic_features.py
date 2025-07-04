import pandas as pd
import os
import re
from tqdm import tqdm
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def main():
    # --- Configuration ---
    DATA_PATH = '/Users/yogurtsmacbook/Cloudcomputing/'
    INPUT_FILENAME = 'news_dataset_cleaned_1000.csv'
    OUTPUT_FILENAME = 'news_dataset_1000_with_features.csv'
    NUM_TOPICS = 8  # We already know the optimal number of topics

    input_file_path = os.path.join(DATA_PATH, INPUT_FILENAME)
    output_file_path = os.path.join(DATA_PATH, OUTPUT_FILENAME)

    # --- Loading and Processing ---
    print(f"Loading data: {input_file_path}")
    df = pd.read_csv(input_file_path)

    # Define the preprocessing function inside main
    def preprocess_for_lda(text):
        if not isinstance(text, str): return []
        # Keep only letters and spaces, then tokenize
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.lower().split()
        # Additional cleaning like stopword removal could be done here if needed
        # but for this step, simple tokenization is often enough.
        return tokens

    print("Tokenizing text for LDA...")
    # (FIXED) Use the 'cleaned_text' column which we know exists
    tqdm.pandas(desc="Tokenizing")
    df['Processed_Tokens'] = df['cleaned_text'].progress_apply(preprocess_for_lda)
    processed_texts = df['Processed_Tokens'].tolist()

    print("Creating dictionary and corpus...")
    dictionary = Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # --- Train a single LDA model ---
    print(f"Training LDA model with {NUM_TOPICS} topics...")
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS, passes=15, random_state=42)
    print("Model training complete.")

    # --- Assign topics and save ---
    print("Assigning dominant topic to each news article...")
    doc_topics = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]

    dominant_topics = []
    for doc_topic in doc_topics:
        if doc_topic:
            dominant_topic = sorted(doc_topic, key=lambda x: x[1], reverse=True)[0][0]
            dominant_topics.append(dominant_topic + 1)  # Topics numbered from 1
        else:
            dominant_topics.append(0)  # 0 for unassigned

    df['dominant_topic'] = dominant_topics

    final_df = df[['cleaned_text', 'label_numeric', 'label', 'dominant_topic']]
    final_df.to_csv(output_file_path, index=False, encoding='utf-8')

    print("-" * 50)
    print("🎉 Success!")
    print(f"File with topic labels has been saved to: '{output_file_path}'")
    print("-" * 50)


if __name__ == '__main__':
    # Ensure NLTK resources are downloaded
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords');
        nltk.download('wordnet')
    main()