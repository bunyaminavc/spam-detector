"""
Streamlit web arayüzü.
Kullanıcıdan e-posta metni alır ve spam/normal sınıflandırması yapar.
"""

import random
import pandas as pd
import streamlit as st
from predict import SpamPredictor

st.set_page_config(
    page_title="Spam E-posta Tespit Sistemi",
    page_icon="📧",
    layout="centered",
)

st.title("📧 Spam E-posta Tespit Sistemi")
st.markdown("Bir e-posta metni girin veya dosya yükleyin — yapay zeka analiz etsin.")
st.divider()


@st.cache_resource
def load_predictor():
    """Modeli bir kez yükler ve önbelleğe alır."""
    return SpamPredictor()


@st.cache_data
def load_samples():
    """CSV'den tüm mesajları yükler."""
    df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    return df["text"].dropna().tolist()


try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Session state başlat
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

# Örnek veri butonları
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎲 Rastgele Veri", use_container_width=True):
        samples = load_samples()
        st.session_state["text_input"] = random.choice(samples)

with col2:
    if st.button("🚨 Spam Örneği", use_container_width=True):
        df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=[0, 1])
        df.columns = ["label", "text"]
        st.session_state["text_input"] = random.choice(df[df["label"] == "spam"]["text"].dropna().tolist())

with col3:
    if st.button("✅ Normal Örneği", use_container_width=True):
        df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=[0, 1])
        df.columns = ["label", "text"]
        st.session_state["text_input"] = random.choice(df[df["label"] == "ham"]["text"].dropna().tolist())

# Metin alanı
text_input = st.text_area(
    "E-posta metnini buraya yapıştırın:",
    value=st.session_state["text_input"],
    height=200,
    placeholder="E-posta içeriğini buraya girin...",
)

st.divider()

# Analiz butonu
if st.button("Analiz Et", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Lütfen bir metin girin veya dosya yükleyin.")
    else:
        with st.spinner("Analiz ediliyor..."):
            result = predictor.predict(text_input)

        if result["is_spam"]:
            st.error(f"### Sonuç: {result['label']}")
            st.metric("Güven Skoru", f"%{result['confidence']}")
            st.markdown(
                "> Bu e-posta yapay zeka tarafından **spam** olarak sınıflandırıldı. "
                "Dikkatli olun, şüpheli bağlantılara tıklamayın."
            )
        else:
            st.success(f"### Sonuç: {result['label']}")
            st.metric("Güven Skoru", f"%{result['confidence']}")
            st.markdown(
                "> Bu e-posta yapay zeka tarafından **normal** olarak sınıflandırıldı."
            )

st.divider()
st.caption("Yapay Zeka Destekli Spam E-posta Tespit Sistemi | TF-IDF + Logistic Regression")
