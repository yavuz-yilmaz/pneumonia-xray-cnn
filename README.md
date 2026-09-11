# CNN ile Akciğer Röntgeninden Zatürre Tespiti

## Proje Amacı

Bu proje, akciğer röntgeni görüntülerinden `NORMAL` ve `PNEUMONIA` sınıflarını ayırt eden CNN tabanlı bir sınıflandırma sistemi geliştirmeyi amaçlar. Proje; veri hazırlama, analiz, model eğitimi, değerlendirme, açıklanabilirlik, API ve basit kullanıcı arayüzü adımlarını kapsayacak şekilde yapılandırılmıştır.

## Problem Tanımı

Zatürre, akciğer dokusunu etkileyen ciddi bir enfeksiyondur. Röntgen görüntülerinde zatürre bulgularının otomatik olarak sınıflandırılması, eğitim ve karar destek senaryoları için yararlı bir bilgisayarlı görü problemidir. Bu projede hedef, tek bir göğüs röntgeni görüntüsünün `NORMAL` veya `PNEUMONIA` sınıfına ait olduğunu tahmin etmektir.

## Kullanılacak Veri Seti

Veri seti kullanıcı tarafından manuel olarak indirilecektir. Proje veri setini otomatik indirmez ve Kaggle API bilgisi istemez.

Beklenen ham veri yapısı:

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Kullanılacak veri seti: Chest X-Ray Images (Pneumonia).

## Kullanılacak Yöntemler

- Görüntü ön işleme ve standartlaştırma
- Veri artırma
- Baseline CNN modeli
- Transfer learning ile ResNet18, EfficientNet-B0 veya MobileNetV3
- Accuracy, precision, recall, F1-score ve ROC-AUC metrikleri
- Confusion matrix ve hata analizi
- Grad-CAM ile görsel açıklanabilirlik
- FastAPI tabanlı tahmin servisi
- Basit web arayüzü

## Kurulum

Python 3.10 veya daha yeni bir sürüm kullanılması gerekir.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell için sanal ortam etkinleştirme:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Makefile kullanan ortamlarda aynı kurulum şu komutla yapılabilir:

```bash
make install
```

## Çalıştırma Komutları

Projedeki ana işler `Makefile` üzerinden çalıştırılır:

```bash
make install
make lint
make test
make eda
make train
make evaluate
make api
make ui
```

Merkezi proje ayarları `configs/config.yaml` dosyasındadır. Bu dosyada veri yolları,
görüntü boyutu, batch size, epoch sayısı, learning rate, model adı, seed ve cihaz
seçimi tanımlanır.

Doğrudan Python modül komutları:

```bash
python -m pip install --upgrade pip; python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
python -m src.eda.run_eda --config configs/config.yaml
python -m src.training.train_baseline --config configs/config.yaml
python -m src.training.train --config configs/config.yaml
python -m src.evaluation.evaluate --config configs/config.yaml
python scripts/generate_gradcam_examples.py --config configs/config.yaml
python scripts/predict_image.py --image data/raw/chest_xray/test/NORMAL/IM-0003-0001.jpeg --model models/best_model.pt
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
python -m streamlit run app/ui/streamlit_app.py
```

## Veri Setini Manuel Yerleştirme ve Hazırlama

Bu proje veri setini otomatik indirmez. Kaggle API, Hugging Face veya başka bir
otomatik indirme akışı kullanılmaz. Veri seti kullanıcı tarafından manuel olarak
yerleştirilmelidir.

1. Chest X-Ray Images (Pneumonia) veri setini indir.
2. Zip dosyasını aç.
3. Açılan `chest_xray` klasörünü proje içinde `data/raw/chest_xray` konumuna koy.
4. Klasör yapısını kontrol et:

```bash
python scripts/check_dataset_ready.py
```

5. Bozuk görsel ve sınıf dağılımı kontrolünü çalıştır:

```bash
python -m src.data.validate_dataset --config configs/config.yaml
```

6. Eğitim için standart manifest dosyalarını üret:

```bash
python -m src.data.standardize_dataset --config configs/config.yaml
```

Bu komutlar başarılı olduğunda şu çıktılar oluşur:

- `reports/metrics/dataset_summary.json`
- `reports/metrics/corrupted_images.json`
- `reports/metrics/stratified_validation_split_summary.json`
- `data/processed/train_manifest.csv`
- `data/processed/val_manifest.csv`
- `data/processed/test_manifest.csv`
- `data/processed/raw_val_manifest.csv`

Varsayılan yapılandırmada `configs/config.yaml` içindeki
`data.use_stratified_validation_split: true` ayarı kullanılır. Bu ayar, ham veri
setindeki çok küçük `val` klasörünü eğitim sırasında model seçimi için kullanmaz.
Bunun yerine ham `train` klasörü sınıf oranları korunarak yeniden bölünür:

- `data/processed/train_manifest.csv`: raw `train` içinden ayrılan eğitim örnekleri
- `data/processed/val_manifest.csv`: raw `train` içinden ayrılan stratified validation örnekleri
- `data/processed/raw_val_manifest.csv`: orijinal küçük raw `val` splitinin izlenebilir kopyası
- `data/processed/test_manifest.csv`: raw `test` splitinin değişmeden kullanılan manifesti

Bu projedeki mevcut ayarda validation oranı `0.15`, seed değeri `42` olarak
tanımlıdır. Son üretilen manifestlerde eğitim seti `4,434`, validation seti
`782`, test seti `624` görüntü içerir.

## Eğitim

Veri seti hazırlandıktan sonra baseline CNN modeli şu komutla eğitilir:

```bash
python -m src.training.train_baseline --config configs/config.yaml
```

Bu komut `SimpleCNN` mimarisini eğitir ve şu çıktıları üretir:

- `models/baseline_cnn.pt`
- `reports/metrics/baseline_history.json`
- `reports/figures/baseline_training_curves.png`

Genel eğitim giriş noktası da aynı baseline modeli varsayılan olarak çalıştırır:

```bash
python -m src.training.train --config configs/config.yaml
```

Makefile kullanan ortamlarda eğitim şu şekilde çalıştırılabilir:

```bash
make train
```

## Değerlendirme

Eğitilen en iyi modelin test seti üzerindeki performansı şu komutla ölçülür:

```bash
make evaluate
```

Doğrudan Python komutu:

```bash
python -m src.evaluation.evaluate --config configs/config.yaml
```

Değerlendirme çıktıları:

- `reports/metrics/test_metrics.json`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`
- `reports/figures/precision_recall_curve.png`
- `reports/figures/misclassified_examples.png`
- `docs/evaluation_summary.md`

## Grad-CAM Açıklanabilirlik

Eğitilen en iyi modelin kararlarını görselleştirmek için Grad-CAM örnekleri şu komutla üretilir:

```bash
python scripts/generate_gradcam_examples.py --config configs/config.yaml
```

Grad-CAM çıktıları:

- `reports/figures/gradcam_correct_normal_*.png`
- `reports/figures/gradcam_correct_pneumonia_*.png`
- `reports/figures/gradcam_misclassified_*.png`
- `docs/gradcam_summary.md`

## Tek Görüntü İçin Tahmin

Eğitilmiş checkpoint ile tek bir röntgen görüntüsü için JSON formatında tahmin üretilebilir:

```bash
python scripts/predict_image.py --image data/raw/chest_xray/test/NORMAL/IM-0003-0001.jpeg --model models/best_model.pt
```

Çıktı alanları:

- `predicted_label`
- `normal_probability`
- `pneumonia_probability`
- `confidence`
- `model_version`

Desteklenen görüntü formatları `.jpeg`, `.jpg` ve `.png` uzantılarıdır.

## API ile Tahmin

FastAPI tabanlı servis `app/api/main.py` içinde yer alır. Servis başlangıçta
`configs/config.yaml` içindeki `paths.best_model_path` değerinden modeli bir kez
yükler ve her requestte aynı modeli kullanır.

```bash
make api
```

Doğrudan çalıştırma komutu:

```bash
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Varsayılan checkpoint dışında bir model kullanmak için:

```bash
$env:PNEUMONIA_MODEL_PATH = "models/best_model.pt"
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Endpointler:

- `GET /health`
- `GET /model-info`
- `POST /predict`

Örnek tahmin isteği:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/raw/chest_xray/test/NORMAL/IM-0003-0001.jpeg"
```

`/predict` yanıtı şu alanları döndürür:

- `predicted_label`
- `normal_probability`
- `pneumonia_probability`
- `confidence`
- `warning`

API yalnızca `.jpeg`, `.jpg` ve `.png` görüntü yüklemelerini kabul eder. Yanıt
her zaman şu uyarıyı içerir: “Bu çıktı tıbbi teşhis amacıyla kullanılmamalıdır.”

## Web Arayüzü

Sunum demosu için Streamlit tabanlı arayüz `app/ui/streamlit_app.py` içinde yer alır.
Arayüz görüntü yükleme, önizleme, tahmin sonucu, `NORMAL` ve `PNEUMONIA`
olasılıkları ile güven skorunu gösterir. Tahminler FastAPI servisine istek atılarak
alınır.

Önce API servisini başlat:

```bash
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Ayrı bir terminalde Streamlit arayüzünü başlat:

```bash
python -m streamlit run app/ui/streamlit_app.py
```

Arayüz varsayılan olarak `http://127.0.0.1:8000` API adresini kullanır. Farklı bir
API adresi kullanmak için `PNEUMONIA_API_URL` ortam değişkenini ayarla:

```powershell
$env:PNEUMONIA_API_URL = "http://127.0.0.1:8000"
python -m streamlit run app/ui/streamlit_app.py
```

Kullanım akışı:

1. API ve Streamlit süreçlerini başlat.
2. Streamlit ekranında JPEG, JPG veya PNG röntgen görüntüsü yükle.
3. Görüntü önizlemesini kontrol et.
4. `Tahmin Et` düğmesine bas.
5. Tahmin sonucunu, olasılıkları ve güven skorunu görüntüle.

## Rapor ve Sunum

Proje raporu ve sunum planı `docs` klasöründe
hazırlanmıştır:

- Rapor: `docs/project_report.md`
- Sunum planı: `docs/presentation_outline.md`

Rapor proje içinde gerçekten üretilen metrik dosyalarındaki değerlere dayanır.
Ana kaynak dosyalar:

- `reports/metrics/dataset_summary.json`
- `reports/metrics/eda_summary.json`
- `reports/metrics/baseline_history.json`
- `reports/metrics/transfer_learning_history.json`
- `reports/metrics/test_metrics.json`
- `docs/gradcam_summary.md`

## Uyarı

Bu proje tıbbi teşhis amacıyla kullanılmaz. Üretilen tahminler yalnızca eğitim, araştırma ve akademik proje bağlamında değerlendirilmelidir. Klinik kararlar için uzman hekim değerlendirmesi gereklidir.
