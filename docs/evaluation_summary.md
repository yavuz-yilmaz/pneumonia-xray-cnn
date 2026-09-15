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

## Karar Eşiği Optimizasyonu

Varsayılan `0.50` sınıflandırma eşiği, validation setinde tarandı. Recall değerinin
`0.98` üzerinde tutulduğu seçenekler arasında en yüksek özgüllüğü veren eşik
`0.70` olarak seçildi ve inference/API akışında varsayılan yapıldı.

| Metrik | Eşik 0.50 | Eşik 0.70 |
|---|---:|---:|
| Accuracy | 0.8574 | **0.8830** |
| Precision | 0.8168 | 0.8468 |
| Recall | 0.9949 | **0.9923** |
| F1-score | 0.8971 | **0.9138** |
| Özgüllük | 0.6282 | **0.7009** |
| False positive | 87 | **70** |
| False negative | 2 | 3 |

Eşik `0.70` validation setinde seçilmiş, ardından test setinde tek seferlik
doğrulanmıştır. Bu ayar false positive sayısını azaltırken zatürre recall değerini
`%99` seviyesinde korur.

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
