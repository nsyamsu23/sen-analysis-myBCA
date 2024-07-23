import streamlit as st
import global_data 
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk
from nlp_id.tokenizer import Tokenizer
from nlp_id.postag import PosTag
from nlp_id.stopword import StopWord
from nlp_id.lemmatizer import Lemmatizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

postagger = PosTag()
tokenizer = Tokenizer()
stopword_nlp_id = StopWord()
lemmatizer = Lemmatizer()
emoji_dict = global_data.emojiDict()
nltk.download('punkt')
nltk.download('vader_lexicon')
inset_neg_url = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/_json_inset-neg.txt'
inset_pos_url = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/_json_inset-pos.txt'
sentiwords_id_url = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/_json_sentiwords_id.txt'

def load_data(df):
    df['predicted_label'] = df['predicted_label'].map({0: 'negatif', 1: 'netral', 2: 'positif'})
    df_positif = df[df['predicted_label'] == 'positif'][['content','content_proses_stemming_nlp_id', 'predicted_label', 'score']]
    df_netral = df[df['predicted_label'] == 'netral'][['content','content_proses_stemming_nlp_id', 'predicted_label', 'score']]
    df_negatif = df[df['predicted_label'] == 'negatif'][['content','content_proses_stemming_nlp_id', 'predicted_label', 'score']]
    return df_positif, df_netral, df_negatif

# Fungsi untuk membuat dan menampilkan WordCloud
def create_wordcloud(content, colormap):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(content)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig
def evaluate_and_plot_knn(y_test, y_pred):
    # Calculate accuracy, confusion matrix, and classification report
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        # Print accuracy and classification report
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(class_report)

    with col2:
        # Plot the confusion matrix using matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(conf_matrix, cmap='Blues')
        fig.colorbar(cax)

        # Set the axis labels
        num_labels = conf_matrix.shape[0]
        ax.set_xticks(np.arange(num_labels))
        ax.set_yticks(np.arange(num_labels))
        ax.set_xticklabels(np.arange(num_labels), rotation=45)
        ax.set_yticklabels(np.arange(num_labels))

        # Annotate the matrix
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')

        # Display the plot in Streamlit
        st.pyplot(fig)
        plt.close(fig)
def train_and_predict_knn(X_train, y_train, X_test, k):
    # Initialize the KNN classifier with the optimal number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the KNN model
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    return y_pred
def display_tfidf_ranking(tfidf_vectorizer, X_train_tfidf_matrix, top_n=20):
    # Get feature names
    terms = tfidf_vectorizer.get_feature_names_out()

    # Sum TF-IDF frequency of each term through documents
    sums = X_train_tfidf_matrix.sum(axis=0)

    # Connecting term to its sums frequency
    data_tfidf = []
    for col, term in enumerate(terms):
        data_tfidf.append((term, sums[0, col]))

    # Create a DataFrame with terms and their corresponding ranks
    ranking = pd.DataFrame(data_tfidf, columns=['term', 'rank'])
    ranking = ranking.sort_values('rank', ascending=False)

    # Display the ranking DataFrame in Streamlit
    st.write(f"Top {top_n} terms by TF-IDF rank")
    st.dataframe(ranking.head(top_n), selection_mode="multi-row", use_container_width=True)

    # Plot the top N terms by TF-IDF rank
    plt.figure(figsize=(10, 8))
    top_terms = ranking.head(top_n)
    plt.barh(top_terms['term'], top_terms['rank'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel('TF-IDF train Rank')
    plt.title(f'Top {top_n} Terms by TF-IDF Rank')

    # Display the plot in Streamlit
    st.pyplot(plt)
def plot_cross_validation_scores(k_scores,k_range=range(1, 31)):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores, marker='o')
    plt.title('Cross-Validation Scores for Different k Values')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.grid(True)
    st.pyplot(plt)


def plot_sentiment_histogram(y_resampled, bin_labels=None, colors=None, title='Histogram of Sentiment', xlabel='Sentiment Categories', ylabel='Frequency'):
    fig, ax = plt.subplots()
    
    # Create a histogram with 3 bins
    counts, bins, patches = ax.hist(y_resampled, bins=3, edgecolor='black', alpha=0.7)

    # Define default bin labels and colors if not provided
    if bin_labels is None:
        bin_labels = ['Negative', 'Neutral', 'Positive']
    if colors is None:
        colors = ['red', 'yellow', 'green']

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set x-axis tick labels at the center of each bin
    ax.set_xticks([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)])
    ax.set_xticklabels(bin_labels)

    # Color each bin separately
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    # Annotate the counts on the histogram
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        ax.annotate(f'{int(count)}', xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
    
    st.pyplot(fig)

    # Set x-axis tick labels at the center of each bin
    plt.xticks([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)], bin_labels)

    # Color each bin separately
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    # Annotate the counts on the histogram
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        plt.annotate(f'{int(count)}', xy=(patch.get_x() + patch.get_width() / 2, height),
                     xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    # Show the plot
    plt.show()


def apply_smote(X, y, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

def process_and_vectorize_reviews(df, label_col, text_col, id_col, test_size=0.3, random_state=42, max_features=1000):
    # Extract documents, labels, and ids
    documents = df[text_col]
    labels = df[label_col]
    ids = df[id_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(documents, labels, ids, test_size=test_size, random_state=random_state)

    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, binary=True)

    # Fit and transform training documents into TF-IDF matrix
    X_train_tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)

    # Transform testing documents into TF-IDF matrix
    X_test_tfidf_matrix = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf_matrix, X_test_tfidf_matrix, y_train, y_test, id_train, id_test, tfidf_vectorizer
def combine_dataframes(df_reviews_all, df_reviews_all_modelling):
    df_reviews_all_modelling_filtered = df_reviews_all_modelling[df_reviews_all_modelling['reviewId'].isin(df_reviews_all['reviewId'])]
    df_combined = pd.merge(df_reviews_all, df_reviews_all_modelling_filtered, on='reviewId', how='right', suffixes=('_x', '_y'))
    return df_combined


def download_and_update_lexicons():
    # Membuat instance SentimentIntensityAnalyzer dan membersihkan leksikon default
    sia1A, sia1B, sia2 = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()
    sia1A.lexicon.clear()
    sia1B.lexicon.clear()
    sia2.lexicon.clear()

    # Fungsi untuk mengunduh data dari URL
    def download_lexicon(url):
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        return json.loads(response.text)

    # Mendapatkan dan memperbarui leksikon dari URL
    sia1A.lexicon.update(download_lexicon(inset_neg_url))
    sia1B.lexicon.update(download_lexicon(inset_pos_url))
    sia2.lexicon.update(download_lexicon(sentiwords_id_url))

    return sia1A, sia1B, sia2


# Fungsi untuk mengklasifikasikan sentimen
def classify_sentiment(text, sia):
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score > 0:
        return 'positif'
    elif compound_score < 0:
        return 'negatif'
    else:
        return 'netral'
def prepare_data(df,kamus_tidak_baku,chat_words_mapping):
    df_proses = df.copy()
    df_proses.drop(columns=['userName', 'userImage', 'replyContent', 'repliedAt', 'reviewCreatedVersion', 'thumbsUpCount', 'replyContent', 'repliedAt', 'appVersion', 'at'], inplace=True)
    df_proses = df_proses.loc[:, ['reviewId', 'score', 'content']]
    df_proses['content_cleaning'] = df_proses['content'].str.lower()
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(replace_emojis_with_meanings)
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(remove_urls)
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(remove_html_tags)
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(hapus_simbol)
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(remove_pattern)

    # Normalisasi Kata
    df_proses['content_cleaning_normalized'] = df_proses['content_cleaning'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku)[0])
    df_proses['content_cleaning_normalized'] = df_proses['content_cleaning_normalized'].apply(lambda x: expand_chat_words(x, chat_words_mapping))
    df_proses['content_cleaning_normalized'] = df_proses['content_cleaning_normalized'].apply(remove_stop_words_nlp_id)
    df_proses = df_proses[df_proses['content_cleaning_normalized'].str.strip() != '']

    # Tokenisasi Teks
    df_proses['content_tokenizing'] = df_proses['content_cleaning_normalized'].apply(tokenizing_words)

    # Part of Speech (POS)
    df_proses['content_part_of_speech'] = df_proses['content_cleaning_normalized'].apply(pos_words)
    pos_tags_to_remove = ["PR", "RP", "UH", "SC", "SYM", "IN", "DT", "CC", "FW"]
    for tag in pos_tags_to_remove:
        df_proses['content_part_of_speech'] = df_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list, tag1=tag))
    df_proses['content_tokenizing'] = df_proses['content_part_of_speech'].apply(pos_to_tokens)

    # Stemming dan Lemmatisasi
    reset_total_changed_count()
    df_proses['content_proses_stemming_nlp_id'] = df_proses['content_tokenizing'].progress_apply(process_and_count_changes)
    sia1A, sia1B, sia2 = download_and_update_lexicons()
    df_proses['content_proses_stemming_nlp_id'] = df_proses['content_proses_stemming_nlp_id'].progress_apply(remove_stop_words_nltk)
    df_proses['sentiment'] = df_proses['content_proses_stemming_nlp_id'].apply(lambda x: classify_sentiment(x, sia2))
    return df_proses
# Definisikan fungsi untuk lemmatisasi token
def lemmatize_wrapper(tokens):
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    original_vs_lemmatized = list(zip(tokens, lemmatized_tokens))

    # Menghitung kata yang telah diubah
    changed_count = sum(1 for original, lemmatized in original_vs_lemmatized if original != lemmatized)

    return ' '.join(lemmatized_tokens), changed_count  # Mengembalikan token yang telah di-lemmatize dan jumlah kata yang diubah


# Variable untuk menghitung total kata yang diubah
total_changed_count = 0
def get_total_changed_count():
    global total_changed_count
    return total_changed_count

def reset_total_changed_count():
    global total_changed_count
    total_changed_count = 0

# Terapkan lemmatisasi dan hitung perubahan dengan progress bar
def process_and_count_changes(tokens):
    global total_changed_count
    lemmatized_tokens, changed_count = lemmatize_wrapper(tokens)
    total_changed_count += changed_count
    return lemmatized_tokens


def remove_pronouns(pos_list,tag1):
    return [(word, tag) for word, tag in pos_list if tag != tag1]

def pos_to_tokens(pos_list):
    # Mengubah list pasangan kata-tag menjadi kalimat
    sentence = ' '.join([word.lower() for word, tag in pos_list])
    # Tokenisasi kalimat
    tokens = tokenizer.tokenize(hapus_simbol(sentence))
    return tokens
def pos_words(text):
    tokens =postagger.get_pos_tag(text)
    return tokens
def tokenizing_words(text):
    tokens = tokenizer.tokenize(remove_stop_words_nlp_id(text))
    return tokens
# Definisikan fungsi untuk mengonversi angka ke huruf
def angka_ke_huruf(angka):
    satuan = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"]

    if angka < 12:
        return satuan[angka]
    elif angka < 20:
        return satuan[angka - 10] + " belas"
    elif angka < 100:
        return satuan[angka // 10] + " puluh" + (" " + satuan[angka % 10] if (angka % 10 != 0) else "")
    elif angka < 200:
        return "seratus" + (" " + angka_ke_huruf(angka - 100) if (angka > 100) else "")
    elif angka < 1000:
        return satuan[angka // 100] + " ratus" + (" " + angka_ke_huruf(angka % 100) if (angka % 100 != 0) else "")
    elif angka < 2000:
        return "seribu" + (" " + angka_ke_huruf(angka - 1000) if (angka > 1000) else "")
    elif angka < 1000000:
        return angka_ke_huruf(angka // 1000) + " ribu" + (" " + angka_ke_huruf(angka % 1000) if (angka % 1000 != 0) else "")
    elif angka < 1000000000:
        return angka_ke_huruf(angka // 1000000) + " juta" + (" " + angka_ke_huruf(angka % 1000000) if (angka % 1000000 != 0) else "")
    else:
        return "Angka terlalu besar"

# Definisikan fungsi untuk mengonversi angka dalam teks menjadi huruf tanpa memperhatikan satuan
def remove_pattern(text):
    def ganti_angka(match):
        angka_str = match.group(0)
        angka = int(re.sub(r'\D', '', angka_str))  # Menghapus karakter non-digit
        return angka_ke_huruf(angka)

    return re.sub(r'\b\d+\b', ganti_angka, text)

# Fungsi penggantian kata tidak baku
def remove_stop_words_nltk(text):
    stop_words = stopwords.words('indonesian')
    stop_words.extend([
        "bca","mybca","MYBCA","myBCA","flash"
    ])
    stop_words = set(stop_words)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
def remove_stop_words_nlp_id(text):
    return stopword_nlp_id.remove_stopword(text)

def expand_chat_words(text, chat_words_mapping):
    words = text.split()
    expanded_words = [chat_words_mapping[word] if word in chat_words_mapping else word for word in words]
    return ' '.join(expanded_words)

def replace_taboo_words(text, kamus_tidak_baku):
    if isinstance(text, str) and isinstance(kamus_tidak_baku, dict):
        words = text.split()
        replaced_words = []
        kalimat_baku = []
        kata_diganti = []
        kata_tidak_baku_hash = []

        for word in words:
            if word in kamus_tidak_baku:
                baku_word = kamus_tidak_baku[word]
                if isinstance(baku_word, str) and all(char.isalpha() for char in baku_word):
                    replaced_words.append(baku_word)
                    kalimat_baku.append(baku_word)
                    kata_diganti.append(word)
                    kata_tidak_baku_hash.append(hash(word))
                else:
                    replaced_words.append(word)  # Append original word if baku_word is not valid
            else:
                replaced_words.append(word)

        replaced_text = ' '.join(replaced_words)
    else:
        replaced_text = ''
        kalimat_baku = []
        kata_diganti = []
        kata_tidak_baku_hash = []

    return replaced_text, kalimat_baku, kata_diganti, kata_tidak_baku_hash
def replace_emojis_with_meanings(text):
    emoji_dict = global_data.emojiDict()
    def replace(match):
        emoji_char = match.group()
        emoji_meaning = emoji_dict.get(emoji_char, "")
        return f" {emoji_meaning} "

    # Pola untuk menemukan semua emotikon dalam teks
    emoji_pattern = re.compile("|".join(map(re.escape, emoji_dict.keys())))
    # Mengganti semua emotikon yang ditemukan dengan artinya
    text_with_meanings = emoji_pattern.sub(replace, text)

    # Menghapus emotikon yang tidak dikenal
    non_known_emoji_pattern = re.compile(r'[^\w\s,.?!]')
    text_cleaned = non_known_emoji_pattern.sub('', text_with_meanings)

    # Menghapus spasi tambahan yang mungkin muncul setelah penggantian
    return ' '.join(text_cleaned.split())
def remove_html_tags(text):
    clean_text = re.sub('<.*>', '', text)
    return clean_text

# hapus simbol smbol
def hapus_simbol(teks):
    return teks.translate(str.maketrans('', '', string.punctuation))

# hapus url
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    clean_text = re.sub(url_pattern, '', text)
    return clean_text