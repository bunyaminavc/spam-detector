"""
Metin ön işleme modülü.
E-posta metinlerini temizleyip model için hazır hale getirir.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def download_nltk_resources():
    """Gerekli NLTK kaynaklarını indirir."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)


class TextPreprocessor:
    """
    E-posta metinlerini temizleyen ve normalleştiren sınıf.

    Uygulanan adımlar:
    - Küçük harfe çevirme
    - URL, sayı ve noktalama temizleme
    - Stopword kaldırma
    - Stemming (isteğe bağlı)
    """

    def __init__(self, use_stemming: bool = False, language: str = "english"):
        """
        Args:
            use_stemming: Stemming uygulanıp uygulanmayacağı.
            language: Stopword dili.
        """
        download_nltk_resources()
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer() if use_stemming else None
        self.use_stemming = use_stemming

    def clean(self, text: str) -> str:
        """
        Tek bir metni temizler.

        Args:
            text: Ham e-posta metni.

        Returns:
            Temizlenmiş metin.

        Time Complexity: O(n) — n metnin karakter sayısı.
        """
        if not isinstance(text, str):
            return ""

        # Küçük harfe çevir
        text = text.lower()

        # URL kaldır
        text = re.sub(r"http\S+|www\S+", "", text)

        # Sayıları kaldır
        text = re.sub(r"\d+", "", text)

        # Noktalama kaldır
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Fazla boşlukları temizle
        text = re.sub(r"\s+", " ", text).strip()

        # Stopword kaldır ve isteğe bağlı stemming uygula
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]

        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return " ".join(tokens)

    def clean_batch(self, texts: list) -> list:
        """
        Metin listesini toplu olarak temizler.

        Args:
            texts: Ham metin listesi.

        Returns:
            Temizlenmiş metin listesi.

        Time Complexity: O(n * m) — n metin sayısı, m ortalama metin uzunluğu.
        """
        return [self.clean(text) for text in texts]
