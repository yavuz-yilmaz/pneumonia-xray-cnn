# EDA Özeti

Bu doküman `python -m src.eda.run_eda --config configs/config.yaml` komutu ile otomatik oluşturulmuştur.

## Veri Seti Özeti

- Okunabilir görüntü sayısı: 5856
- Okunamayan görüntü sayısı: 0

## Sınıf Dağılımı

| Split | NORMAL | PNEUMONIA | Toplam |
| --- | ---: | ---: | ---: |
| train | 1341 | 3875 | 5216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |

## Sayısal Özet

- Çoğunluk/azınlık sınıf oranı: 2.699
- Ortalama genişlik: 1327.881 piksel
- Ortalama yükseklik: 970.689 piksel
- Ortalama aspect ratio: 1.443
- Ortalama piksel yoğunluğu: 122.786

## Gözlenen Olası Problemler

- Sınıf dağılımı dengesiz görünüyor; eğitimde class weight veya sampler kullanılması önerilir.
- Aspect ratio dağılımı geniş; resize işleminde oran bozulmasının etkisi izlenmeli.
- EDA sırasında okunamayan görüntüyle karşılaşılmadı.

## Modelleme İçin Öneriler

- PNEUMONIA ve NORMAL sınıfları arasındaki dengesizlik için weighted loss veya WeightedRandomSampler denenmelidir.
- Görüntüler sabit boyuta getirilirken eğitim ve inference aşamalarında aynı normalizasyon kullanılmalıdır.
- Tıbbi anlamı bozabilecek agresif augmentasyonlardan kaçınılmalıdır.
- Model seçiminde accuracy ile birlikte recall, precision, F1-score ve confusion matrix mutlaka raporlanmalıdır.

## Üretilen Görseller

- `reports/figures/eda_class_distribution.png`
- `reports/figures/eda_sample_grid_NORMAL.png`
- `reports/figures/eda_sample_grid_PNEUMONIA.png`
- `reports/figures/eda_image_size_distribution.png`
- `reports/figures/eda_aspect_ratio_distribution.png`
- `reports/figures/eda_pixel_intensity_histogram.png`
- `reports/figures/eda_average_pixel_maps.png`
