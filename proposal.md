# Proje Teklifi: Yapay Zeka Destekli Spam E-posta Tespit Sistemi

**Öğrenci:** Bünyamin AVCI
**Tarih:** Nisan 2026

---

## Hedef

Bu projede amacım spam ve normal olan mailleri sınıflandırmak. Modeller eldeki veri setiyle eğitildi. Python ile yazıldı, kullanıcı webde metin girip sonuç görebiliyor.

---

## Veri Seti

Veri seti için Kaggle'den https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset bağlantısındaki veri setinden faydalandım. Toplamda 5572 mesaj içeriği var. 747'si spam. Etiket olarak ham=0, spam=1 olarak etiketlendi.

---

## Yöntem Bölümü

Metinler önce belli başlı ön işleme adımlarından geçiyor, bunlar; küçük harfe çevirme, noktalamaları temizleme ve cümlede anlam taşımayan ifadeler yani stopwords (the, is, a, and vb.) kaldırma.

TF-IDF ile metinler sayıya dönüştürüldü ve bilgisayarın anlayacağı hale getirildi, bu sayede hızlandı. Unigram ve bigram ile "free, won, you won, click here" gibi spam olabilecek mesajlar değerlendirilir.

---

## Metodoloji

Logistic Regression kullandım çünkü kolay, TF-IDF ile de uyumlu, bu sayede hızlı. Veriyi %80 eğitim ve %20 test olarak böldüm. Değerlendirme için kullandığım metrikler ise bunlar: Accuracy, Precision, Recall, F1-Score.

---

## Beklenen Sonuç

Accuracy : %98.03

Precision : %93.79

Recall : %91.28

F1-Score : %92.52

---

## Referanslar

- Almeida, T.A. et al. (2011). *Contributions to the Study of SMS Spam Filtering.*
- Kaggle – SMS Spam Collection Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- scikit-learn Documentation: https://scikit-learn.org

---

## Vibe Coding İçin Kullandığım AI

Claude

---

## Github

https://github.com/bunyaminavc/spam-detector
