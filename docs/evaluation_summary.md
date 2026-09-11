# Test Seti Değerlendirme Özeti

## Test Sonuçları

- Model: `resnet18`
- Checkpoint: `models\best_model.pt`
- Test örnek sayısı: `624`
- Accuracy: `0.8574`
- Precision: `0.8168`
- Recall: `0.9949`
- F1-score: `0.8971`
- ROC-AUC: `0.9599`

## En Güçlü Metrikler

- `recall`: `0.9949`
- `roc_auc`: `0.9599`

## Zayıf Noktalar

- `precision`: `0.8168`
- `accuracy`: `0.8574`

## False Positive / False Negative Yorumu

- False positive sayısı: `87`. Bu durumda model NORMAL bir görüntüyü PNEUMONIA olarak işaretlemiştir.
- False negative sayısı: `2`. Bu durumda model PNEUMONIA bir görüntüyü NORMAL olarak işaretlemiştir.
- Hatalı sınıflandırılan örnekler: `reports\figures\misclassified_examples.png`

## Tıbbi Bağlamda Recall Neden Önemli?

Recall, gerçekten zatürre olan örneklerin ne kadarının yakalandığını gösterir. Bu projede PNEUMONIA pozitif sınıf olarak ele alındığı için düşük recall, zatürre bulgusu taşıyan bazı görüntülerin NORMAL olarak kaçırılması anlamına gelir. Klinik bağlamda bu hata tipi daha risklidir; ancak bu proje yalnızca eğitim ve akademik çalışma amacıyla değerlendirilmelidir.

## Üretilen Çıktılar

- Metrikler: `reports\metrics\test_metrics.json`
- Confusion matrix: `reports\figures\confusion_matrix.png`
- ROC curve: `reports\figures\roc_curve.png`
- Precision-recall curve: `reports\figures\precision_recall_curve.png`
- Hatalı örnekler: `reports\figures\misclassified_examples.png`

## Uyarı

Bu proje eğitim amaçlıdır; çıktılar tıbbi teşhis amacıyla kullanılmamalıdır.
