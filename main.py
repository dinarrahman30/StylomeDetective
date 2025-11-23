import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "authorship_pipeline.joblib"


# ============== Stylometric ==============
def extract_stylometric_features(series: pd.Series) -> pd.DataFrame:
    function_words = [
        'the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was',
        'he', 'for', 'it', 'with', 'as', 'his', 'on', 'be', 'at',
        'by', 'i', 'this', 'had', 'not', 'are', 'but', 'from',
        'or', 'have', 'an', 'they', 'which', 'one', 'you', 'were',
        'her', 'all', 'she', 'there', 'would', 'their'
    ]

    def compute(text: str) -> pd.Series:
        if not isinstance(text, str):
            text = ""

        features = {}

        # Basic statistics
        features['text_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['vocab_richness'] = len(set(words)) / len(words) if words else 0

        # Character frequencies
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['whitespace_ratio'] = sum(1 for c in text if c.isspace()) / len(text) if text else 0

        # Punctuation patterns
        features['comma_freq'] = text.count(',') / len(text) if text else 0
        features['semicolon_freq'] = text.count(';') / len(text) if text else 0
        features['colon_freq'] = text.count(':') / len(text) if text else 0
        features['exclamation_freq'] = text.count('!') / len(text) if text else 0
        features['question_freq'] = text.count('?') / len(text) if text else 0
        features['period_freq'] = text.count('.') / len(text) if text else 0
        features['dash_freq'] = text.count('--') / len(text) if text else 0
        features['quote_freq'] = text.count('"') / len(text) if text else 0

        # Sentence statistics
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        # Function words (top 20)
        text_lower = text.lower()
        for fw in function_words[:20]:
            features[f'fw_{fw}'] = text_lower.split().count(fw) / len(words) if words else 0

        return pd.Series(features)

    return series.apply(compute)

# ============== Load model ==============
@st.cache_resource
def load_pipeline():
    saved = joblib.load(MODEL_PATH)  # dict: {"tfidf_char", "scaler", "svm"}
    return saved["tfidf_char"], saved["scaler"], saved["svm"]


tfidf_char, scaler, svm_best = load_pipeline()


def predict_from_text_series(text_series: pd.Series):
    X_char = tfidf_char.transform(text_series)
    X_num = extract_stylometric_features(text_series)
    X_num_scaled = scaler.transform(X_num)
    X_combined = hstack([X_char, X_num_scaled])
    y_pred = svm_best.predict(X_combined)
    return y_pred


# ============== UI ==============
st.set_page_config(page_title="Authorship Attribution", layout="wide")
st.title("📝 Authorship Attribution Prediction")
st.caption("Prediksi penulis teks menggunakan SVM + char TF-IDF + stylometric features.")

tab_single, tab_batch = st.tabs(["🔤 Prediksi Satu Teks", "📊 Prediksi dari CSV"])


# ---------- Tab 1: single text ----------
with tab_single:
    st.subheader("Prediksi satu teks")


    text_input = st.text_area("Masukkan teks:", value="", height=200)

    if st.button("Prediksi penulis"):
        if not text_input.strip():
            st.warning("Teks kosong.")
        else:
            y_pred = predict_from_text_series(pd.Series([text_input]))
            st.success(f"Penulis yang diprediksi: **{y_pred[0]}**")


# ---------- Tab 2: batch CSV ----------
with tab_batch:
    st.subheader("Prediksi dari file CSV")

    file = st.file_uploader("Upload CSV (harus ada kolom 'text'; opsional 'author')", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Shape:", df.shape)
        st.dataframe(df.head())

        if "text" not in df.columns:
            st.error("Kolom 'text' tidak ditemukan.")
        else:
            if st.button("Prediksi semua baris"):
                with st.spinner("Memproses..."):
                    y_pred = predict_from_text_series(df["text"])
                    df["predicted_author"] = y_pred

                    st.success("Selesai.")
                    st.dataframe(df.head())

                    if "author" in df.columns:
                        y_true = df["author"]
                        acc = accuracy_score(y_true, y_pred)
                        st.metric("Accuracy", f"{acc:.4f}")

                        report = classification_report(y_true, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.write("Classification report:")
                        st.dataframe(report_df)

                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Download CSV dengan prediksi",
                        data=csv_out,
                        file_name="external_predictions.csv",
                        mime="text/csv",
                    )
