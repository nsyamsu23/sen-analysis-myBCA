import streamlit as st
import global_data
import re
import string
import pandas as pd
import  openpyxl
from collections import Counter

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
    
if __name__ == "__main__":
    main()
