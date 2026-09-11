# Sunum Planı: CNN ile Akciğer Röntgeninden Zatürre Tespiti

## Slayt 1 - Başlık ve Amaç

- Proje adı: CNN ile Akciğer Röntgeninden Zatürre Tespiti
- Amaç: Röntgen görüntüsünü `NORMAL` veya `PNEUMONIA` olarak sınıflandırmak
- Uyarı: Proje eğitim amaçlıdır, tıbbi teşhis aracı değildir

## Slayt 2 - Problem ve Motivasyon

- Zatürre akciğerleri etkileyen ciddi bir enfeksiyondur
- Göğüs röntgenleri değerlendirme sürecinde sık kullanılır
- Derin öğrenme, karar destek ve akademik analiz için incelenebilir

## Slayt 3 - Veri Seti

- Chest X-Ray Images (Pneumonia) veri seti kullanıldı
- Toplam görüntü sayısı: 5856
- Ham splitler: train 5216, validation 16, test 624
- Eğitim manifestleri: train 4434, stratified validation 782, test 624
- Sınıflar: `NORMAL` ve `PNEUMONIA`

## Slayt 4 - Veri Analizi ve Ön İşleme

- Bozuk görüntü sayısı: 0
- Genel sınıf dağılımı dengesiz: 1583 `NORMAL`, 4273 `PNEUMONIA`
- Görüntüler `256 x 256` boyutuna getirildi
- Train için sınırlı augmentasyon, test için deterministik dönüşümler kullanıldı

## Slayt 5 - Modelleme Yaklaşımı

- Baseline: `SimpleCNN`
- Final model: transfer learning ile `resnet18`
- ImageNet ön eğitimli ağırlıklar kullanıldı
- Son katman iki sınıflı sınıflandırmaya göre değiştirildi

## Slayt 6 - Eğitim Düzeneği

- Batch size: 24
- Learning rate: 0.0003
- Fine-tuning learning rate: 0.00001
- Weighted loss kullanıldı
- Eğitim iki aşamalı yapıldı: classifier eğitimi ve fine-tuning

## Slayt 7 - Test Sonuçları

- Accuracy: 0.8574
- Precision: 0.8168
- Recall: 0.9949
- F1-score: 0.8971
- ROC-AUC: 0.9599

## Slayt 8 - Confusion Matrix ve Hata Analizi

- True negative: 147
- False positive: 87
- False negative: 2
- True positive: 388
- Yorum: Recall yüksek, false negative düşük; ancak false positive sayısı azaltılabilir

## Slayt 9 - Grad-CAM Açıklanabilirlik

- Doğru sınıflandırılmış `NORMAL` ve `PNEUMONIA` örnekleri görselleştirildi
- Hatalı sınıflandırılmış örnekler için de Grad-CAM üretildi
- Amaç: modelin hangi görüntü bölgelerine odaklandığını yaklaşık olarak incelemek
- Sınırlılık: Grad-CAM klinik lokalizasyon kanıtı değildir

## Slayt 10 - Genel Değerlendirme ve Sınırlılıklar

- Proje uçtan uca veri hazırlama, eğitim, değerlendirme, API ve UI içeriyor
- Test performansı akademik proje için güçlü görünmektedir
- Validation seti küçüktür ve veri seti sınıf dengesizliği içerir
- Klinik kullanım için uzman doğrulaması, farklı veri kaynakları ve regülasyon süreci gerekir
