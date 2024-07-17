import streamlit as st
import global_data
import pandas as pd
import io
import matplotlib.pyplot as plt

def main():
    df_reviews_all = global_data.get_data("reviews")
    st.title("Data Understanding")
    st.write("Ini adalah halaman Data Understanding.")
    st.write("Original Data")
    st.dataframe(df_reviews_all.head(5), selection_mode="multi-row", use_container_width=True)
    # Data information
    st.subheader("Descriptive Statistics")
    st.write(df_reviews_all.describe(), selection_mode="multi-row", use_container_width=True)
    
    # Data information
    st.subheader("Data Information")
    buffer = io.StringIO()
    df_reviews_all.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    #date
    # Convert 'at' and 'repliedAt' to datetime
    df_reviews_all['at'] = pd.to_datetime(df_reviews_all['at'])
    df_reviews_all['repliedAt'] = pd.to_datetime(df_reviews_all['repliedAt'])

    # Total Score Plot
    st.subheader("Total Score")
    if 'score' in df_reviews_all.columns:
        df_reviews_all['score'] = df_reviews_all['score'].astype(int)
        score_counts = df_reviews_all['score'].value_counts().sort_index()

        fig, ax = plt.subplots()
        score_counts.plot(kind='bar', ax=ax)
        ax.set_title('Total Rating')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Jumlah Komentar')
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels([1, 2, 3, 4, 5])
        st.pyplot(fig)
    else:
        st.write("The dataframe does not have a 'score' column.")
    
    # Distribution of 'at' (review times)
    st.subheader("Distribution of 'at' (Review Times)")
    if 'at' in df_reviews_all.columns:
        # Ensure 'at' is in datetime format
        df_reviews_all['at'] = pd.to_datetime(df_reviews_all['at'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df_reviews_all['at'].hist(bins=50, edgecolor='k', alpha=0.7, ax=ax)
        ax.set_title('Grafik Tahun ke Tahun Banyaknya Ulasan oleh Pengguna')
        ax.set_xlabel('Tahun-Bulan')
        ax.set_ylabel('Banyaknya')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write("The dataframe does not have an 'at' column.")
    # Trend Ulasan dari Waktu ke Waktu
    st.subheader("Trend Ulasan dari Waktu ke Waktu")
    if 'at' in df_reviews_all.columns:
        df_reviews_all['at'] = pd.to_datetime(df_reviews_all['at'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df_reviews_all.set_index('at').resample('M').size().plot(marker='o', ax=ax)
        ax.set_title('Trend Ulasan dari Waktu ke Waktu')
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Banyaknya Ulasan')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write("The dataframe does not have an 'at' column.")
    # Distribusi Score menurut Versi Aplikasi
    st.subheader("Distribusi Score menurut Versi Aplikasi")
    if 'appVersion' in df_reviews_all.columns and 'score' in df_reviews_all.columns:
        app_version_scores = df_reviews_all.groupby(['appVersion', 'score']).size().unstack().fillna(0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        app_version_scores.plot(kind='bar', stacked=True, ax=ax)
        
        ax.set_title('Grafik Banyaknya Rating yang diberikan Berdasarkan Update Versi Aplikasi')
        ax.set_xlabel('Versi Aplikasi')
        ax.set_ylabel('Banyaknya Rating')
        ax.set_xticklabels(app_version_scores.index, rotation=90)
        ax.legend(title='Rating')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write("The dataframe does not have 'appVersion' or 'score' columns.")

if __name__ == "__main__":
    main()
