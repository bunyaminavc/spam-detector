"""
Tahmin modülü.
Eğitilmiş modeli yükleyip yeni e-posta metinleri üzerinde tahmin yapar.
"""

import os
import pickle

from preprocess import TextPreprocessor


class SpamPredictor:
    """
    Eğitilmiş modeli kullanarak e-posta metinlerini sınıflandıran sınıf.

    model/ dizinindeki model.pkl ve vectorizer.pkl dosyalarını yükler.
    """

    LABELS = {0: "Normal Mail (Ham)", 1: "Spam"}

    def __init__(self, model_dir: str = "model"):
        """
        Args:
            model_dir: Model dosyalarının bulunduğu dizin.
        """
        model_path = os.path.join(model_dir, "model.pkl")
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                "Model dosyaları bulunamadı. Lütfen önce train.py ile modeli eğitin."
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        self.preprocessor = TextPreprocessor()

    def predict(self, text: str) -> dict:
        """
        Tek bir e-posta metni için tahmin üretir.

        Args:
            text: Ham e-posta metni.

        Returns:
            label: Tahmin sonucu ("Normal Mail" / "Spam")
            is_spam: Spam ise True, değilse False
            confidence: Modelin güven skoru (0-100 arası)
        """
        cleaned = self.preprocessor.clean(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        confidence = self.model.predict_proba(vectorized)[0][prediction]

        return {
            "label": self.LABELS[prediction],
            "is_spam": prediction == 1,
            "confidence": round(float(confidence) * 100, 2),
        }
