# Proje Teklifi: Yapay Zeka Destekli Spam E-posta Tespit Sistemi

**Öğrenci:** Bünyamin AVCI
**Tarih:** Nisan 2026

---

## Hedef

<!-- Kendi cümlelerinle yaz:
     - Ne yapıyorsun? (e-posta metni alıp spam/normal sınıflandırıyorsun)
     - Neden önemli? (spam maillerin güvenlik tehdidi, kimlik avı, dolandırıcılık)
     - Nasıl sunulacak? (Streamlit web arayüzü ile gerçek zamanlı)
-->

---

## Veri Seti

<!-- Kendi cümlelerinle yaz:
     - SMS Spam Collection Dataset (Kaggle, UCI ML Repository)
     - ~5.574 mesaj: 4.827 normal (ham), 747 spam
     - Tek CSV: spam.csv — v1 (etiket: ham/spam), v2 (mesaj metni)
     - Etiketleme: 0 = Normal (ham), 1 = Spam
     - Veri kaynağı: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
-->

---

## Ön İşleme ve Vektörizasyon

<!-- Kendi cümlelerinle yaz:
     - Küçük harfe çevirme, URL/sayı/noktalama kaldırma, stopword temizleme
     - TF-IDF ile sayısal vektöre dönüştürme (unigram + bigram, max 10.000 özellik)
     - Neden TF-IDF? → kelime sıklığı + nadirlik dengesini yakalar, spam için etkili
-->

---

## Metodoloji

<!-- Kendi cümlelerinle yaz:
     - Sınıflandırıcı: Logistic Regression
     - Neden? → yorumlanabilir, hızlı, TF-IDF ile iyi çalışır
     - Değerlendirme metrikleri: Accuracy, Precision, Recall, F1-Score
     - Train/test split: %80 / %20, stratify=True
-->

---

## Beklenen Sonuç

<!-- Kendi cümlelerinle yaz:
     - TF-IDF + LR bu veri setinde ~%98 accuracy bekleniyor
     - Kullanıcı e-posta metnini yapıştırıp anında sonuç alabilecek
     - Güven skoru ile birlikte spam/normal tahmini gösterilecek
-->

---

## Teknoloji Yığını

| Katman             | Araç / Kütüphane       |
|--------------------|------------------------|
| Programlama dili   | Python 3.10+           |
| NLP / ML           | scikit-learn, NLTK     |
| Arayüz             | Streamlit              |
| Versiyon kontrolü  | Git / GitHub           |

---

## Referanslar

- Almeida, T.A. et al. (2011). *Contributions to the Study of SMS Spam Filtering.*
- Kaggle – SMS Spam Collection: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- scikit-learn Documentation: https://scikit-learn.org
