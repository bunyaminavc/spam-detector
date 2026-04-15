"""
Görselleştirme modülü.
Proposal ve dokümantasyon için model performans görselleri üretir.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from preprocess import TextPreprocessor


def load_model(model_dir="model"):
    """Model ve vektörizeri yükler."""
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def plot_confusion_matrix(model, vectorizer):
    """Confusion matrix görseli üretir."""
    preprocessor = TextPreprocessor()

    df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna()

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    X_test_clean = preprocessor.clean_batch(X_test.tolist())
    X_test_vec = vectorizer.transform(X_test_clean)
    y_pred = model.predict(X_test_vec)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (Ham)", "Spam"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("confusion_matrix.png kaydedildi ✓")


def plot_top_spam_words(model, vectorizer, n=20):
    """Spam için en etkili kelimeleri gösteren grafik üretir."""
    feature_names = vectorizer.get_feature_names_out()
    # Sınıf 1 (spam) için katsayılar
    coefs = model.coef_[0]
    top_spam_idx = np.argsort(coefs)[-n:][::-1]
    top_ham_idx = np.argsort(coefs)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Spam kelimeleri
    axes[0].barh(
        [feature_names[i] for i in top_spam_idx][::-1],
        [coefs[i] for i in top_spam_idx][::-1],
        color="#e74c3c"
    )
    axes[0].set_title("Spam İçin En Etkili Kelimeler", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Model Katsayısı")

    # Normal (ham) kelimeleri
    axes[1].barh(
        [feature_names[i] for i in top_ham_idx][::-1],
        [abs(coefs[i]) for i in top_ham_idx][::-1],
        color="#2ecc71"
    )
    axes[1].set_title("Normal Mail İçin En Etkili Kelimeler", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Model Katsayısı")

    plt.suptitle("TF-IDF + Logistic Regression — Özellik Ağırlıkları", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("top_words.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("top_words.png kaydedildi ✓")


def plot_pipeline():
    """Proje pipeline'ını gösteren akış diyagramı üretir."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        ("Ham Metin\n(E-posta)", 0.8, "#3498db"),
        ("Ön İşleme\n(Temizleme,\nStopword)", 2.5, "#9b59b6"),
        ("TF-IDF\nVektörizasyon", 4.4, "#e67e22"),
        ("Logistic\nRegression", 6.3, "#e74c3c"),
        ("Tahmin\n(Spam / Normal)", 8.5, "#27ae60"),
    ]

    for label, x, color in steps:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.75, 0.8), 1.5, 1.4,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.9
        ))
        ax.text(x, 1.5, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")

    for i in range(len(steps) - 1):
        x_start = steps[i][1] + 0.75
        x_end = steps[i + 1][1] - 0.75
        ax.annotate("", xy=(x_end, 1.5), xytext=(x_start, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))

    ax.set_title("Spam Tespit Sistemi — İşlem Pipeline'ı",
                 fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig("pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("pipeline.png kaydedildi ✓")


if __name__ == "__main__":
    print("Görseller üretiliyor...")
    model, vectorizer = load_model()
    plot_confusion_matrix(model, vectorizer)
    plot_top_spam_words(model, vectorizer)
    plot_pipeline()
    print("\nTüm görseller proje klasörüne kaydedildi!")
