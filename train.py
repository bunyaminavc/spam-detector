"""
Model eğitim modülü.
TF-IDF vektörizasyonu ve Logistic Regression ile spam e-posta tespit modeli eğitir.
"""

import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from preprocess import TextPreprocessor


class SpamTrainer:
    """
    Spam e-posta tespit modeli eğiten sınıf.

    TF-IDF vektörizasyonu ile Logistic Regression modelini birleştirir.
    Eğitilmiş model ve vektörizer disk'e kaydedilir.
    """

    def __init__(self, model_dir: str = "model"):
        """
        Args:
            model_dir: Model dosyalarının kaydedileceği dizin.
        """
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs", C=5.0, class_weight="balanced")
        os.makedirs(model_dir, exist_ok=True)

    def load_dataset(self, path: str) -> pd.DataFrame:
        """
        SMS Spam Collection veri setini yükler.

        Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
        Beklenen CSV formatı: 'v1' (ham/spam etiketi), 'v2' (mesaj metni)

        Args:
            path: spam.csv dosyasının yolu.

        Returns:
            DataFrame (text, label sütunları).
            label: 0 = Normal (ham), 1 = Spam
        """
        df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
        df.columns = ["label", "text"]

        # ham -> 0, spam -> 1
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        df = df.dropna(subset=["label", "text"])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Toplam örnek : {len(df)}")
        print(f"Normal (ham) : {(df['label'] == 0).sum()}")
        print(f"Spam         : {(df['label'] == 1).sum()}")
        return df

    def train(self, data_path: str, test_size: float = 0.2):
        """
        Modeli eğitir ve değerlendirme metriklerini yazdırır.

        Args:
            data_path: spam.csv dosyasının yolu.
            test_size: Test seti oranı (varsayılan: 0.2).
        """
        df = self.load_dataset(data_path)

        print("Ön işleme uygulanıyor...")
        df["clean_text"] = self.preprocessor.clean_batch(df["text"].tolist())

        X = df["clean_text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print("TF-IDF vektörizasyonu...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        print("Model eğitimi başlıyor...")
        self.model.fit(X_train_vec, y_train)

        y_pred = self.model.predict(X_test_vec)
        self._print_metrics(y_test, y_pred)
        self._save()

    def _print_metrics(self, y_true, y_pred):
        """Değerlendirme metriklerini ekrana yazdırır."""
        print("\n========== Model Performansı ==========")
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
        print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")
        print("\nDetaylı Rapor:")
        print(classification_report(y_true, y_pred, target_names=["Normal (Ham)", "Spam"]))

    def _save(self):
        """Eğitilmiş model ve vektörizeri disk'e kaydeder."""
        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model '{self.model_dir}/' dizinine kaydedildi.")


if __name__ == "__main__":
    trainer = SpamTrainer()
    trainer.train(data_path="data/spam.csv")
