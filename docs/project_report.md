# CNN ile Akciğer Röntgeninden Zatürre Tespiti: Kısa Proje Raporu

## Problem Tanımı

Bu projede amaç, tek bir akciğer röntgeni görüntüsünü `NORMAL` veya `PNEUMONIA` sınıfına ayıran CNN tabanlı bir sınıflandırma sistemi geliştirmektir. Model, göğüs röntgeni görüntüsünden zatürre bulgusu olasılığını tahmin eder ve sonucu iki sınıflı bir çıktı olarak sunar.

Bu çalışma eğitim ve akademik proje amacıyla hazırlanmıştır. Üretilen tahminler tıbbi teşhis amacıyla kullanılmamalıdır.

## Problemin Önemi

Zatürre, akciğerleri etkileyen ve erken fark edilmediğinde ciddi sonuçlar doğurabilen bir enfeksiyon hastalığıdır. Göğüs röntgenleri zatürre değerlendirmesinde sık kullanılan görüntüleme kaynaklarından biridir. Derin öğrenme tabanlı sınıflandırma modelleri, bu görüntüler üzerinde karar destek sistemi olarak incelenebilir; ancak klinik kullanım için çok daha geniş doğrulama, uzman değerlendirmesi, hasta güvenliği analizi ve regülasyon süreçleri gerekir.

## Veri Seti Açıklaması

Projede Chest X-Ray Images (Pneumonia) veri seti kullanılmıştır. Veri seti manuel olarak `data/raw/chest_xray` klasörüne yerleştirilmiş ve `train`, `val`, `test` kırılımlarında `NORMAL` ve `PNEUMONIA` sınıflarıyla düzenlenmiştir.

Gerçek veri özeti `reports/metrics/dataset_summary.json` dosyasından alınmıştır:

| Split | NORMAL | PNEUMONIA | Toplam |
| --- | ---: | ---: | ---: |
| Train | 1341 | 3875 | 5216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| Toplam | 1583 | 4273 | 5856 |

Bozuk veya okunamayan görüntü sayısı `0` olarak raporlanmıştır. Genel sınıf dağılımında `PNEUMONIA` sınıfı 4273 görüntüyle veri setinin yaklaşık %72.97'sini, `NORMAL` sınıfı 1583 görüntüyle yaklaşık %27.03'ünü oluşturmaktadır. Bu nedenle eğitim aşamasında sınıf dengesizliği dikkate alınmıştır.

## Kullanılan Yapay Zekâ Yöntemi

Proje iki modelleme yaklaşımı içerir:

- Baseline model: Basit CNN mimarisi (`SimpleCNN`)
- Final model: Transfer learning tabanlı `resnet18`

Final değerlendirme `models/best_model.pt` checkpoint dosyasındaki `resnet18` modeliyle yapılmıştır. Model, ImageNet ön eğitimli ağırlıklarla başlatılmış, son sınıflandırıcı katmanı iki sınıflı probleme göre değiştirilmiş ve eğitim iki aşamalı yürütülmüştür: önce sınıflandırıcı katman eğitilmiş, ardından fine-tuning aşamasında son katmanlar düşük öğrenme oranıyla güncellenmiştir.

## Veri Ön İşleme

Görüntüler eğitim ve inference sürecinde aynı boyuta getirilmiştir. Kullanılan görüntü boyutu `256 x 256` olarak yapılandırılmıştır. Normalizasyon için checkpoint içinde saklanan değerler kullanılmıştır:

- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

Eğitim verisinde kontrollü veri artırma uygulanmıştır. Dönüşümler röntgen görüntülerinin tıbbi anlamını bozmayacak şekilde sınırlı tutulmuştur; yeniden boyutlandırma, kontrollü yatay çevirme, küçük rotasyon ve normalizasyon kullanılmıştır. Validation ve test aşamalarında deterministik dönüşümler uygulanmıştır.

## Model Mimarisi

Final model `resnet18` mimarisidir. ResNet mimarisi artık bağlantılar kullanan evrişimsel bloklardan oluşur. Bu bağlantılar, daha derin ağlarda gradyan akışını iyileştirerek eğitimi daha kararlı hale getirir. Projede son sınıflandırıcı katman `NORMAL` ve `PNEUMONIA` sınıfları için iki çıkış verecek şekilde değiştirilmiştir.

Baseline `SimpleCNN` modeli evrişim blokları, batch normalization, ReLU aktivasyonu, max pooling, dropout ve fully connected sınıflandırıcı katmanlarından oluşur. Baseline model referans performans üretmek için kullanılmıştır; final rapor metrikleri transfer learning modeli üzerinden verilmiştir.

## Deney Düzeneği

Deneyler `configs/config.yaml` yapılandırmasıyla yürütülmüştür:

- Model: `resnet18`
- Image size: `256`
- Batch size: `24`
- Başlangıç learning rate: `0.0003`
- Fine-tuning learning rate: `0.00001`
- Seed: `42`
- Device: `cuda`
- Weighted loss: etkin
- Weighted sampler: kapalı

Transfer learning eğitim geçmişi `reports/metrics/transfer_learning_history.json` dosyasına kaydedilmiştir. Eğitim toplam 20 epoch sürmüştür. İlk 8 epoch sınıflandırıcı aşaması, sonraki 12 epoch fine-tuning aşamasıdır. Eğitim RTX 3060 6 GB GPU üzerinde CUDA ve mixed precision ile çalıştırılmıştır. Ham veri setindeki 16 görüntülük küçük validation split eğitim zamanı model seçimi için kullanılmamış; raw `train` klasöründen sınıf oranları korunarak 782 görüntülük stratified validation manifesti oluşturulmuştur. En iyi model epoch 18'de seçilmiştir; validation accuracy `0.9795`, validation precision `0.9896`, validation recall `0.9828` ve validation F1-score `0.9862` olarak kaydedilmiştir.

Baseline eğitim geçmişi `reports/metrics/baseline_history.json` dosyasında bulunur. Baseline CNN 8 epoch çalıştırılmıştır. En iyi baseline validation accuracy `0.9066`, validation precision `0.9504`, validation recall `0.9225` ve validation F1 skoru `0.9362` olarak kaydedilmiştir.

## Sonuçlar ve Metrikler

Test seti sonuçları `reports/metrics/test_metrics.json` dosyasından alınmıştır. Final model `models/best_model.pt` checkpoint dosyasındaki `resnet18` modelidir.

| Metrik | Değer |
| --- | ---: |
| Test örnek sayısı | 624 |
| Accuracy | 0.8574 |
| Precision | 0.8168 |
| Recall | 0.9949 |
| F1-score | 0.8971 |
| ROC-AUC | 0.9599 |

Sınıf bazlı rapora göre `PNEUMONIA` sınıfında recall `0.9949`, precision `0.8168` ve F1-score `0.8971` olarak ölçülmüştür. `NORMAL` sınıfında precision `0.9866`, recall `0.6282` ve F1-score `0.7676` olarak raporlanmıştır.

Bu sonuçlar modelin zatürre sınıfını yakalama oranının çok yüksek olduğunu, ancak daha fazla normal görüntüyü yanlış şekilde zatürre olarak işaretlediğini göstermektedir. Tıbbi bağlamda recall yüksekliği önemlidir; çünkü false negative sayısının düşük olması zatürre içeren görüntülerin kaçırılma riskini azaltır. Buna karşılık false positive artışı, modelin karar eşiği ve precision-recall dengesi açısından ayrıca ayarlanması gerektiğini gösterir. Yine de bu sistem klinik karar için yeterli değildir.

## Confusion Matrix Yorumu

Confusion matrix değerleri şöyledir:

| Gerçek / Tahmin | NORMAL | PNEUMONIA |
| --- | ---: | ---: |
| NORMAL | 147 | 87 |
| PNEUMONIA | 2 | 388 |

Bu matrise göre 147 normal görüntü doğru şekilde `NORMAL`, 388 zatürre görüntüsü doğru şekilde `PNEUMONIA` olarak sınıflandırılmıştır. 87 normal görüntü false positive olarak `PNEUMONIA` sınıfına atanmıştır. 2 zatürre görüntüsü false negative olarak `NORMAL` sınıfına atanmıştır.

False negative sayısının 2'ye düşmesi olumlu bir sonuçtur; çünkü pozitif sınıf olan zatürre örneklerini kaçırma oranı çok düşüktür. Buna karşılık false positive sayısının 87 olması, modelin bazı normal görüntülerde fazla alarm ürettiğini gösterir. Sunumda bu durum precision ve recall dengesi üzerinden açıklanmalıdır.

## Grad-CAM Açıklanabilirlik Yorumu

Grad-CAM çıktıları `reports/figures/gradcam_*.png` dosyalarında üretilmiştir. Özet dosyası `docs/gradcam_summary.md` içinde bulunmaktadır. Üretilen görseller:

- 4 doğru sınıflandırılmış `NORMAL` örneği
- 4 doğru sınıflandırılmış `PNEUMONIA` örneği
- 6 hatalı sınıflandırılmış örnek

Grad-CAM, modelin tahmin sırasında son evrişimsel katmanlarda hangi bölgeleri daha etkili kullandığını yaklaşık olarak gösterir. Bu proje kapsamında overlay görselleri, modelin özellikle akciğer bölgelerine mi yoksa görüntü kenarları, işaretler ve akciğer dışı alanlar gibi ilgisiz bölgelere mi odaklandığını incelemek için kullanılmıştır.

Grad-CAM çıktıları nedensel açıklama değildir ve patoloji lokalizasyonu olarak değerlendirilmemelidir. Bu görseller yalnızca model davranışını daha anlaşılır hale getiren yardımcı analiz çıktılarıdır.

## Sınırlılıklar

Bu çalışmanın başlıca sınırlılıkları şunlardır:

- Proje eğitim amaçlıdır; klinik teşhis aracı değildir.
- Yüksek recall elde edilmiştir, ancak false positive sayısı yüksektir; karar eşiği optimizasyonu ayrıca yapılmalıdır.
- Veri setinde sınıf dengesizliği vardır; `PNEUMONIA` sınıfı genel veri setinde belirgin çoğunluktadır.
- Veri seti tek kaynaklı ve sınırlı dağılıma sahip olabilir; farklı cihaz, hastane ve popülasyonlarda genelleme ayrıca test edilmelidir.
- Model yalnızca iki sınıf üretir; farklı akciğer hastalıklarını, görüntü kalitesini veya klinik bulguları ayrı ayrı değerlendirmez.
- Grad-CAM görselleri yaklaşık açıklanabilirlik sağlar; tıbbi uzman yorumu yerine geçmez.

## Genel Değerlendirme

Proje, veri doğrulama, EDA, PyTorch tabanlı eğitim, transfer learning, test değerlendirmesi, Grad-CAM açıklanabilirlik, FastAPI servis katmanı ve Streamlit arayüzünü içeren uçtan uca bir derin öğrenme uygulaması olarak tamamlanmıştır. Test sonuçlarına göre `resnet18` modeli `0.8574` accuracy, `0.9949` recall, `0.8971` F1-score ve `0.9599` ROC-AUC değerlerine ulaşmıştır.

Sonuçlar akademik proje düzeyi için güçlü görünmektedir; özellikle zatürre sınıfındaki çok yüksek recall dikkat çekicidir. Bununla birlikte false positive sayısı, sınıf dengesizliği ve klinik doğrulama eksikliği nedeniyle model gerçek sağlık hizmeti süreçlerinde kullanıma uygun değildir.

## Kaynakça

- Kaggle Chest X-Ray Images Pneumonia Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Mendeley Data Chest X-Ray Dataset: https://data.mendeley.com/datasets/rscbjbr9sj/2
- Rajpurkar, P. ve diğerleri. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. https://arxiv.org/abs/1711.05225
- He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
- Selvaraju, R. R. ve diğerleri. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. https://arxiv.org/abs/1610.02391
