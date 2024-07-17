import streamlit as st
import global_data
import re
import string
import pandas as pd
import  openpyxl
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nlp_id.stopword import StopWord
from nlp_id.tokenizer import Tokenizer
from nlp_id.postag import PosTag
from nlp_id.stopword import StopWord
from nlp_id.lemmatizer import Lemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import reprlib
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import imblearn.over_sampling
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import itertools
def main():
    st.title("Persiapan Data")
    st.write("Halaman ini berfokus pada Pemahaman dan Pembersihan Data.")
    
    # Mendapatkan data global
    df_reviews_all = global_data.get_data("reviews")
    
    # Menampilkan data asli
    st.subheader("Data Asli")
    st.write("Ini adalah dataset asli yang mengandung berbagai kolom.")
    st.dataframe(df_reviews_all.head(5), selection_mode="multi-row", use_container_width=True)
    
    # Menghapus kolom yang tidak diperlukan
    st.subheader("Menghapus Data")
    st.write("Menghapus kolom yang tidak diperlukan untuk analisis kita.")
    df_reviews_all_proses = df_reviews_all.copy()
    df_reviews_all_proses.drop(columns=['reviewId', 'userImage', 'replyContent', 'repliedAt', 'reviewCreatedVersion', 'thumbsUpCount', 'replyContent', 'repliedAt', 'appVersion', 'at'], inplace=True)
    st.dataframe(df_reviews_all_proses.head(5), selection_mode="multi-row", use_container_width=True)
    
    # Mengatur ulang kolom
    st.subheader("Mengatur Ulang Kolom")
    st.write("Mengatur ulang kolom untuk fokus pada userName, score, dan content.")
    df_reviews_all_proses = df_reviews_all_proses.loc[:, ['userName', 'score', 'content']]
    st.dataframe(df_reviews_all_proses.head(5), selection_mode="multi-row", use_container_width=True)
    
    st.title("Pembersihan Data")
    st.write("Bagian ini melibatkan pembersihan data untuk mempersiapkannya untuk analisis.")
    
    # Mengubah teks ke huruf kecil
    st.subheader("Mengubah Teks ke Huruf Kecil")
    st.write("Mengubah semua teks di kolom 'content' menjadi huruf kecil untuk keseragaman.")
    df_reviews_all_proses['content_cleaning'] = df_reviews_all_proses['content'].str.lower()
    st.dataframe(df_reviews_all_proses.head(5), selection_mode="multi-row", use_container_width=True)
    
    # Mengganti emotikon dengan arti teks
    st.subheader("Mengganti Emotikon dengan Arti Teks")
    st.write("Mengganti emotikon di kolom 'content' dengan arti teks yang sesuai.")
    
    emoji_dict = global_data.get_data("emoji")
    
    def replace_emojis_with_meanings(text):
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

    # Terapkan fungsi pengganti emotikon
    df_reviews_all_proses['content_cleaning'] = df_reviews_all_proses['content_cleaning'].apply(replace_emojis_with_meanings)
    st.dataframe(df_reviews_all_proses.head(5), selection_mode="multi-row", use_container_width=True)
    
    # Menghapus kode HTML, simbol-simbol, dan URL
    st.subheader("Menghapus Kode HTML, Simbol, dan URL")
    st.write("Menghapus kode HTML, simbol-simbol, dan URL dari teks di kolom 'content_cleaning'.")

    # Menghapus kode HTML
    def remove_html_tags(text):
        clean_text = re.sub('<.*?>', '', text)
        return clean_text

    # Menghapus simbol
    def hapus_simbol(teks):
        return teks.translate(str.maketrans('', '', string.punctuation))

    # Menghapus URL
    def remove_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        clean_text = re.sub(url_pattern, '', text)
        return clean_text

    df_reviews_all_proses['content_cleaning'] = df_reviews_all_proses['content_cleaning'].apply(remove_urls)
    df_reviews_all_proses['content_cleaning'] = df_reviews_all_proses['content_cleaning'].apply(remove_html_tags)
    df_reviews_all_proses['content_cleaning'] = df_reviews_all_proses['content_cleaning'].apply(hapus_simbol)
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)
    # Konversi angka ke huruf
    st.subheader("Konversi Angka ke Huruf")
    st.write("Mengonversi angka dalam teks menjadi huruf.")


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

    # Definisikan fungsi untuk mengonversi angka dalam teks menjadi huruf
    def remove_pattern(text):
        def ganti_angka(match):
            angka_str = match.group(0)
            angka = int(re.sub(r'\D', '', angka_str))  # Menghapus karakter non-digit
            return angka_ke_huruf(angka)

        return re.sub(r'\b\d+\b', ganti_angka, text)

    # Terapkan fungsi pada kolom 'content_cleaning'
    df_reviews_all_proses['content_cleaning'] = df_reviews_all_proses['content_cleaning'].apply(remove_pattern)

    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)
    
    
    st.title("Normalisasi Kata")
    st.write("weqwe")


    st.subheader("frefquensi kata")
    st.write("Menghitung jumlah frefquensi kata")
    
    #@title frefquensi kata
    text = " ".join(df_reviews_all_proses['content_cleaning'])
    tokens = text.split()

    # Menghitung frekuensi kemunculan setiap kata
    word_counts = Counter(tokens)

    # Mengambil kata dengan frekuensi kemunculan tertinggi
    top_words = word_counts.most_common(25000)

    word, count = zip(*top_words)
    data_kata = pd.DataFrame({'kata': word,'count':count})
    st.dataframe(data_kata.head(5), selection_mode="multi-row", use_container_width=True)
    
    #@title mengubah kata tidak baku menjadi baku
    st.subheader("mengubah kata tidak baku menjadi baku")
    st.write("Menghitung jumlah frefquensi kata")
    # Fungsi penggantian kata tidak baku
    def replace_taboo_words(text, kamus_tidak_baku):
        if isinstance(text, str):
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
                    replaced_words.append(word)
            replaced_text = ' '.join(replaced_words)
        else:
            replaced_text = ''
            kalimat_baku = []
            kata_diganti = []
            kata_tidak_baku_hash = []

        return replaced_text, kalimat_baku, kata_diganti, kata_tidak_baku_hash

    # Baca kamus kata tidak baku
    kamus_data = global_data.get_data("baku")
    kamus_tidak_baku = pd.Series(kamus_data["kata_baku"],index=kamus_data["tidak_baku"]).to_dict()


    # Aplikasikan fungsi replace_taboo_words pada kolom content_cleaning
    df_reviews_all_proses['content_cleaning_normalized'] = df_reviews_all_proses['content_cleaning'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku)[0])
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)
    
    # @title kata yang disingkat diperpanjang dan slangword
    st.subheader("kata yang disingkat diperpanjang dan slangword")
    st.write("Menghitung jumlah frefquensi kata")
    chat_words_mapping = global_data.get_data("mapping")
    # Fungsi untuk memperluas kata-kata chat
    def expand_chat_words(text, chat_words_mapping):
        words = text.split()
        expanded_words = [chat_words_mapping[word] if word in chat_words_mapping else word for word in words]
        return ' '.join(expanded_words)

    # Normalisasi kata
    df_reviews_all_proses['content_cleaning_normalized'] = df_reviews_all_proses['content_cleaning_normalized'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku)[0])

    # Aplikasikan fungsi expand_chat_words pada kolom content_cleaning_normalized
    df_reviews_all_proses['content_cleaning_normalized'] = df_reviews_all_proses['content_cleaning_normalized'].apply(lambda x: expand_chat_words(x, chat_words_mapping))
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)
    
    st.subheader("menghapus kata henti dan kata yang tidak bermakna")
    st.write("Menghitung jumlah frefquensi kata")

    nltk.download('stopwords')
    from nlp_id.stopword import StopWord
    stopword_nlp_id = StopWord()
    def remove_stop_words_nltk(text):
        stop_words = stopwords.words('indonesian')
        stop_words.extend([
            "pmm","merdeka mengajar","nya"
        ])
        stop_words = set(stop_words)
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    def remove_stop_words_nlp_id(text):
        rm_stopword = remove_stop_words_nltk(text)
        return stopword_nlp_id.remove_stopword(rm_stopword)

    df_reviews_all_proses['content_cleaning_normalized'] = df_reviews_all_proses['content_cleaning_normalized'].apply(remove_stop_words_nlp_id)

    df_reviews_all_proses['content_cleaning_normalized'] = df_reviews_all_proses['content_cleaning_normalized'].apply(lambda x: expand_chat_words(x, chat_words_mapping))
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)

    df_reviews_all_proses = df_reviews_all_proses[df_reviews_all_proses['content_cleaning_normalized'].str.strip() != '']
    
    st.subheader("Tokenizer")
    st.write("Menghitung jumlah frefquensi kata")
    tokenizer = Tokenizer()
    def tokenizing_words(text):
        tokens = tokenizer.tokenize(text)
        return tokens

    df_reviews_all_proses['content_tokenizing']  = df_reviews_all_proses['content_cleaning_normalized'] .apply(tokenizing_words)
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)
    
    st.subheader("POS")
    st.write("Menghitung jumlah frefquensi kata")
    nltk.download('punkt')
    postagger = PosTag()
    def pos_words(text):
        tokens =postagger.get_pos_tag(text)
        return tokens

    df_reviews_all_proses['content_part_of_speech']  = df_reviews_all_proses['content_cleaning_normalized'].apply(pos_words)
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)

    st.subheader("edit di POS")
    st.write("Menghitung jumlah frefquensi kata")
    #@title edit di POS
    def remove_pronouns(pos_list,tag1):
        return [(word, tag) for word, tag in pos_list if tag != tag1]

    # Apply the function to the DataFrame
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="PR"))
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="RP"))
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="UH"))
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="SC"))
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="SYM"))
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="IN"))
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="DT"))
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="CC"))
    df_reviews_all_proses['content_part_of_speech'] = df_reviews_all_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list,tag1="FW"))
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)

    st.subheader("Lemmatizer")
    st.write("Menghitung jumlah frefquensi kata")
    # Inisialisasi StopWord dan Lemmatizer
    # Inisialisasi lemmatizer
    lemmatizer = Lemmatizer()

    # Definisikan fungsi untuk lemmatisasi token
    def lemmatize_wrapper(tokens):
        # Lemmatize each token
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    # Asumsi df_reviews_all_proses sudah didefinisikan sebelumnya dan memiliki kolom 'content_tokenizing'
    df_reviews_all_proses['content_proses_stemming_nlp_id'] = df_reviews_all_proses['content_tokenizing'].apply(lemmatize_wrapper)

    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)

    st.subheader("world cloud")
    st.write("Menghitung jumlah frefquensi kata")
    # Create the text for the word cloud
    text = ' '.join(df_reviews_all_proses['content_proses_stemming_nlp_id'].apply(lambda x: str(x) if isinstance(x, (str, int, float)) else ''))

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plotting the word cloud
    plt.figure(figsize=(10, 5))
    plt.title("WordCloud Stemming menggunakan NLP_id", fontsize=18, fontweight='bold', pad=20)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # Save the plot to a file
    plt.savefig('wordcloud.png')

    # Use Streamlit to display the word cloud
    st.title("WordCloud Stemming menggunakan NLP_id")
    st.image('wordcloud.png')

    st.subheader("Labeling Sentiment")
    st.write("Menghitung jumlah frefquensi kata")

    # Unduh VADER lexicon dari nltk
    nltk.download('vader_lexicon')

    # Membuat instance SentimentIntensityAnalyzer dan membersihkan leksikon default
    sia1A, sia1B, sia2 = SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer(), SentimentIntensityAnalyzer()
    sia1A.lexicon.clear()
    sia1B.lexicon.clear()
    sia2.lexicon.clear()

    # URL leksikon InSet dan SentiWords
    inset_neg_url = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/_json_inset-neg.txt'
    inset_pos_url = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/_json_inset-pos.txt'
    sentiwords_id_url = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/_json_sentiwords_id.txt'

    # Fungsi untuk mengunduh data dari URL
    def download_lexicon(url):
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.text

    # Mendapatkan data dari URL
    data1A = download_lexicon(inset_neg_url)
    data1B = download_lexicon(inset_pos_url)
    data2 = download_lexicon(sentiwords_id_url)

    # Mengubah leksikon menjadi dictionary
    insetNeg = json.loads(data1A)
    insetPos = json.loads(data1B)
    senti = json.loads(data2)

    # Memperbarui leksikon VADER yang sudah 'dimodifikasi'
    sia1A.lexicon.update(insetNeg)
    sia1B.lexicon.update(insetPos)
    sia2.lexicon.update(senti)

    def classify_sentiment(text, sia):
        scores = sia.polarity_scores(text)
        compound_score = scores['compound']
        if compound_score > 0:
            return 'positif'
        elif compound_score < 0:
            return 'negatif'
        else:
            return 'netral'
    
    # Terapkan fungsi pada kolom DataFrame menggunakan leksikon yang diperbarui
    df_reviews_all_proses['sentiment'] = df_reviews_all_proses['content_proses_stemming_nlp_id'].apply(lambda x: classify_sentiment(x, sia2))
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)

    st.title("Sentiment Analysis with Label Encoding")
    st.write("Below is the DataFrame with original sentiment and the numeric encoding:")
    # Label Encoding
    label_encoder = LabelEncoder()
    df_reviews_all_proses['sentiment_numeric'] = label_encoder.fit_transform(df_reviews_all_proses['sentiment'])
    st.dataframe(df_reviews_all_proses.head(20), selection_mode="multi-row", use_container_width=True)

    st.title("Sentiment Analysis with Label Encoding")
    st.write("Below is the DataFrame with original sentiment and the numeric encoding:")
    # @title membagi data latih dan data test
    documents = df_reviews_all_proses['content_proses_stemming_nlp_id']
    labels = df_reviews_all_proses['sentiment_numeric']
    X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.3, random_state=42)
    
    # Assuming df_reviews_all_proses is already defined and contains the column 'content_proses_stemming_nlp_id'
    documents = df_reviews_all_proses['content_proses_stemming_nlp_id']

    max_features = 1000  # Adjust this as needed

    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, binary=True)

    # Fit and transform training documents into TF-IDF matrix
    X_train_tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)

    # Transform testing documents into TF-IDF matrix
    X_test_tfidf_matrix = tfidf_vectorizer.transform(X_test)

    # Convert TF-IDF train matrix to DataFrame
    df_tfidf_train = pd.DataFrame(X_train_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    # Convert TF-IDF testing matrix to DataFrame
    df_tfidf_test  = pd.DataFrame(X_train_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Assume tfidf_vectorizer and X_train_tfidf_matrix are already created and fit
    st.title("tf idf data latih")
    st.write("Below is the DataFrame with original sentiment and the numeric encoding:")
    terms = tfidf_vectorizer.get_feature_names_out()
    sums = X_train_tfidf_matrix.sum(axis=0)

    # Connecting term to its sums frequency
    data_tfidf = [(term, sums[0, col]) for col, term in enumerate(terms)]

    # Create a DataFrame with terms and their corresponding ranks
    ranking = pd.DataFrame(data_tfidf, columns=['term', 'rank'])
    ranking = ranking.sort_values('rank', ascending=False)

    # Streamlit display
    st.write("TF-IDF Terms Ranking")

    st.dataframe(ranking.head(20), selection_mode="multi-row", use_container_width=True)

    # Plotting the top 20 terms by TF-IDF rank
    st.write("### Top 20 Terms by TF-IDF Rank")
    top_terms = ranking.head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_terms['term'], top_terms['rank'], color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel('TF-IDF Train Rank')
    ax.set_title('Top 20 Terms by TF-IDF Rank')
    st.pyplot(fig)

    st.title("tf idf data uji")
    terms = tfidf_vectorizer.get_feature_names_out()
    sums = X_test_tfidf_matrix.sum(axis=0)

    # Connecting term to its sums frequency
    data_tfidf = [(term, sums[0, col]) for col, term in enumerate(terms)]

    # Create a DataFrame with terms and their corresponding ranks
    ranking = pd.DataFrame(data_tfidf, columns=['term', 'rank'])
    ranking = ranking.sort_values('rank', ascending=False)

    # Streamlit display
    st.title("TF-IDF Terms Ranking (Test Data)")
    st.write("Below is the DataFrame showing the top terms ranked by TF-IDF for the test data:")

    st.dataframe(ranking.head(10))

    # Plotting the top 20 terms by TF-IDF rank
    st.write("### Top 20 Terms by TF-IDF Rank (Test Data)")
    top_terms = ranking.head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_terms['term'], top_terms['rank'], color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel('TF-IDF Testing Rank')
    ax.set_title('Top 20 Terms by TF-IDF Rank (Test Data)')
    st.pyplot(fig)

    #@title imbalanced SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_tfidf_matrix, y_train)


    # Create a histogram with 3 bins
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(y_resampled, bins=3, edgecolor='black', alpha=0.7)

    # Define bin labels and colors
    bin_labels = ['Negative', 'Neutral', 'Positive']
    colors = ['red', 'yellow', 'green']

    # Add labels and title
    ax.set_xlabel('Sentiment Categories')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Sentiment')

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

    # Display the plot in Streamlit
    st.title("Histogram of Sentiment Categories")
    st.pyplot(fig)

    st.title("Find the optimal number of neighbors for KNN")
    # Find the optimal number of neighbors for KNN
    k_range = range(1, 31)
    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_resampled, y_resampled, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())

    # Plot the cross-validation scores to find the elbow point
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, k_scores, marker='o')
    ax.set_title('Cross-Validation Scores for Different k Values')
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Cross-Validation Accuracy')
    ax.grid(True)

    # Display the plot in Streamlit
    st.title("Optimal Number of Neighbors for KNN")
    st.pyplot(fig)

    # Find the optimal k (elbow point)
    optimal_k = k_range[np.argmax(k_scores)]
    st.write(f"The optimal number of neighbors is {optimal_k} with a cross-validation accuracy of {max(k_scores):.2f}")

    # Initialize the KNN classifier with the optimal k
    knn = KNeighborsClassifier(n_neighbors=optimal_k)

    # Train the KNN model
    knn.fit(X_resampled, y_resampled)

    # Make predictions on the test data
    y_pred = knn.predict(X_test_tfidf_matrix)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.dataframe(conf_matrix)
    st.write("Classification Report:")
    st.text(class_report)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(conf_matrix, cmap='Blues')
    fig.colorbar(cax)
    tick_marks = np.arange(len(np.unique(y_test)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(np.unique(y_test), rotation=45)
    ax.set_yticklabels(np.unique(y_test))

    # Annotate the matrix
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        ax.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Create a DataFrame with the predictions
    df_predictions = pd.DataFrame({
        'Original Text': X_test,  # Assuming X_test is a list or array of original text
        'True Label': y_test,
        'Predicted Label': y_pred
    })

    st.write("Predictions DataFrame:")
    st.dataframe(df_predictions)
if __name__ == "__main__":
    main()
