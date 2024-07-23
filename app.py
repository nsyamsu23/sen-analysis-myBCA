import streamlit as st
from streamlit_option_menu import option_menu
import global_data 
import pandas as pd
from function import *
from tqdm import tqdm

tqdm.pandas()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Set the page configuration
st.set_page_config(
    page_title="myBCA | Analisis Sentimen",
    page_icon="https://play-lh.googleusercontent.com/ckrnc0pzN0oZgSaMQMnOYrICdBLwFTuI17MlTUp9ftyZPJ-m4K1pA3_Dz1B-1dCFVZbv",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Define the pages and styles for the navigation bar
pages = ["Dashboard", "Data Preparation", "Modeling dan Evaluasi", "About"]
with st.sidebar:
    page = option_menu("Main Menu", pages, styles={ "container": {"padding": "5!important", "background-color": "#fafafa"}, "icon": {"color": "orange", "font-size": "25px"}, "nav-link": { "font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee", }, "nav-link-selected": {"background-color": "#02ab21"}, },
        icons=['house', 'gear', 'activity', 'info-circle'], menu_icon="cast", default_index=0)
# Load and process data

reviews_df = global_data.reviews_all_app()
reviews_df["at"] = pd.to_datetime(reviews_df["at"])

kamus_tidak_baku = global_data.data_dict()
chat_words_mapping = global_data.chat_words_mapping()

if 'df_reviews_all_proses' not in st.session_state:
    st.session_state.df_reviews_all_proses = prepare_data(reviews_df, kamus_tidak_baku, chat_words_mapping)
df_reviews_all_proses = st.session_state.df_reviews_all_proses

# Modeling
if "df_reviews_all_modeling" not in st.session_state:
    st.session_state.df_reviews_all_modeling = df_reviews_all_proses
df_reviews_all_modeling = st.session_state.df_reviews_all_modeling


label_encoder = LabelEncoder()
df_reviews_all_modeling["sentiment_numeric"] = label_encoder.fit_transform(df_reviews_all_modeling["sentiment"])
X_train_tfidf_matrix, X_test_tfidf_matrix, y_train, y_test, id_train, id_test, tfidf_vectorizer = process_and_vectorize_reviews(
    df_reviews_all_modeling,
    label_col='sentiment_numeric',
    text_col='content_proses_stemming_nlp_id',
    id_col='reviewId'
)
X_resampled, y_resampled = apply_smote(X_train_tfidf_matrix, y_train)
optimal_k = 1
y_pred = st.session_state.y_pred = train_and_predict_knn(X_resampled, y_resampled, X_test_tfidf_matrix, optimal_k)
st.session_state.results_df = pd.DataFrame({
    'reviewId': id_test,
    'predicted_label': y_pred,
    'true_label': y_test
})
results_df = st.session_state.get('results_df', None)
df_combined =combine_dataframes(st.session_state.results_df,df_reviews_all_modeling)

# Page: about
if page == "About":
    st.header("About")
    st.write("Ini adalah halaman about.")

elif page == "Data Preparation":
    st.header("Data Preparation")
    st.write("Ini adalah halaman persiapan data.")
    with st.expander("Data preview"):
        st.subheader("Data preview")
        st.write("Ini adalah halaman persiapan data.")
        st.dataframe(reviews_df, selection_mode="multi-row", use_container_width=True)

    with st.expander("Case Folding"):
        st.subheader("Case Folding")
        st.write("Mengubah huruf besar ke kecil, mengubah emotikon ke teks, dan menghapus kode HTML, URL, dan simbol-simbol.")
        df_reviews_all_proses_cf = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_cleaning']]
        st.dataframe(df_reviews_all_proses_cf, selection_mode="multi-row", use_container_width=True)

    with st.expander("Normalisasi Kata"):
        st.subheader("Normalisasi Kata")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_nk = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_cleaning_normalized']]
        st.dataframe(df_reviews_all_proses_nk, selection_mode="multi-row", use_container_width=True)

    with st.expander("Tokenisasi Teks"):
        st.subheader("Tokenisasi Teks")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_token = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_tokenizing']]
        st.dataframe(df_reviews_all_proses_token, selection_mode="multi-row", use_container_width=True)

    with st.expander("Part of Speech (POS)"):
        st.subheader("Part of Speech (POS)")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_pos = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_part_of_speech']]
        st.dataframe(df_reviews_all_proses_pos, selection_mode="multi-row", use_container_width=True)

    with st.expander("Stemming dan Lemmatisasi"):
        st.subheader("Stemming dan Lemmatisasi")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_stem = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_proses_stemming_nlp_id']]
        st.dataframe(df_reviews_all_proses_stem, selection_mode="multi-row", use_container_width=True)
    with st.expander("Labeling"):
        st.subheader("lebeling ")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_label = df_reviews_all_proses.loc[:, ['reviewId','content', 'sentiment','sentiment_numeric']]
        st.dataframe(df_reviews_all_proses_label, selection_mode="multi-row", use_container_width=True)

# Page: Modeling dan Evaluasi
elif page == "Modeling dan Evaluasi":
    st.subheader("Sentimen lexicon")
    plot_sentiment_histogram(df_reviews_all_modeling["sentiment"])
    st.subheader("TF IDF")
    col1tfidf,col2tfidf = st.columns(2)
    with col1tfidf:
        display_tfidf_ranking(tfidf_vectorizer, X_train_tfidf_matrix)
    with col2tfidf:
        display_tfidf_ranking(tfidf_vectorizer, X_test_tfidf_matrix)
    st.write(f"The optimal number of neighbors is {optimal_k}")
    st.subheader("SMOTE")
    plot_sentiment_histogram(y_resampled)
    st.subheader("KNN and Evaluasi conflusion Matrix")
    evaluate_and_plot_knn(y_test, y_pred)
    st.dataframe(df_combined, selection_mode="multi-row", use_container_width=True)
# Page: Dashboard
elif page == "Dashboard":
    st.image(image='https://play-lh.googleusercontent.com/ckrnc0pzN0oZgSaMQMnOYrICdBLwFTuI17MlTUp9ftyZPJ-m4K1pA3_Dz1B-1dCFVZbv', width=150, use_column_width=150)
    st.title('Analisis Sentimen w/ K-NN - Aplikasi myBCA')

    col1, col2 = st.columns(2)

    with col1:
        st.header("Total Score")
        # Plot Total Score
        score_counts = reviews_df['score'].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        score_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('Total Rating')
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Jumlah Komentar')
        ax1.set_xticklabels([1, 2, 3, 4, 5])  # Ensure x-ticks are labeled correctly
        st.pyplot(fig1)

    with col2:
        st.header("Distribution of 'at' (review times)")
        # Plot Distribution of 'at'
        fig2, ax2 = plt.subplots()
        reviews_df['at'].hist(bins=50, edgecolor='k', alpha=0.7, ax=ax2)
        ax2.set_title('Grafik Tahun ke Tahun Banyaknya Ulasan oleh Pengguna')
        ax2.set_xlabel('Tahun-Bulan')
        ax2.set_ylabel('Banyaknya')
        ax2.grid(True)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.header("Distribusi Score menurut Versi Aplikasi")
        # Plot Distribusi Score menurut Versi Aplikasi
        app_version_scores = reviews_df.groupby(['appVersion', 'score']).size().unstack().fillna(0)
        fig3, ax3 = plt.subplots(figsize=(14, 8))
        app_version_scores.plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title('Grafik Banyaknya Rating yang diberikan Berdasarkan Update Versi Aplikasi')
        ax3.set_xlabel('Versi Aplikasi')
        ax3.set_ylabel('Banyaknya Rating')
        ax3.legend(title='Rating')
        ax3.grid(True)
        ax3.set_xticklabels(app_version_scores.index, rotation=90)
        st.pyplot(fig3)

    with col4:
        st.header("Trend Ulasan dari Waktu ke Waktu")
        # Plot Trend Ulasan dari Waktu ke Waktu
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        reviews_df.set_index('at').resample('M').size().plot(marker='o', ax=ax4)
        ax4.set_title('Trend Ulasan dari Waktu ke Waktu')
        ax4.set_xlabel('Waktu')
        ax4.set_ylabel('Banyaknya Ulasan')
        ax4.grid(True)
        st.pyplot(fig4)

    st.write("This is a simple example of a dashboard created using Streamlit.")
    df_positif, df_netral, df_negatif = load_data(df_combined)

    # Pisahkan konten berdasarkan prediksi yang dihasilkan
    content_positif = ' '.join(df_positif['content_proses_stemming_nlp_id'])
    content_netral = ' '.join(df_netral['content_proses_stemming_nlp_id'])
    content_negatif = ' '.join(df_negatif['content_proses_stemming_nlp_id'])

    # Hitung rata-rata skor untuk setiap klasifikasi
    mean_score_positif = df_positif['score'].mean()
    mean_score_netral = df_netral['score'].mean()
    mean_score_negatif = df_negatif['score'].mean()

    # Display in Streamlit
    st.subheader("WordClouds and DataFrames for Predictions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Positive")
        fig_pos = create_wordcloud(content_positif, 'Greens')
        st.pyplot(fig_pos)
        st.write(f"Average Score: {mean_score_positif:.2f}")
        st.dataframe(df_positif[['content', 'predicted_label']])

    with col2:
        st.header("Neutral")
        fig_netral = create_wordcloud(content_netral, 'Greys')
        st.pyplot(fig_netral)
        st.write(f"Average Score: {mean_score_netral:.2f}")
        st.dataframe(df_netral[['content', 'predicted_label']])

    with col3:
        st.header("Negative")
        fig_neg = create_wordcloud(content_negatif, 'Reds')
        st.pyplot(fig_neg)
        st.write(f"Average Score: {mean_score_negatif:.2f}")
        st.dataframe(df_negatif[['content', 'predicted_label']])
    evaluate_and_plot_knn(y_test, y_pred)
st.write("End.")