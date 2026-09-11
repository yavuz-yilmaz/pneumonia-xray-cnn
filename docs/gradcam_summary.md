# Grad-CAM Açıklanabilirlik Özeti

## Grad-CAM Nedir?

Grad-CAM, bir CNN modelinin belirli bir sınıf kararını verirken son evrişimsel özellik
haritalarında hangi bölgelerin daha etkili olduğunu yaklaşık olarak görselleştiren bir
açıklanabilirlik yöntemidir. Üretilen ısı haritası, modelin karar skoruna katkısı yüksek
olan bölgeleri sıcak renklerle gösterir.

## Bu Projede Nasıl Kullanıldı?

- Açıklanan model: `resnet18`
- Çalıştırma cihazı: `cuda`
- Hedef katman: seçilen mimarinin son evrişimsel özellik katmanı
- Açıklanan sınıf: modelin tahmin ettiği sınıf
- Görsel formatı: orijinal röntgen ve Grad-CAM heatmap overlay

Seçilen örnekler:

- Doğru sınıflandırılmış NORMAL: `4`
- Doğru sınıflandırılmış PNEUMONIA: `4`
- Hatalı sınıflandırılmış örnek: `6`

## Model Hangi Alanlara Odaklanıyor Gibi Görünüyor?

Üretilen overlay görselleri, modelin kararını görüntünün belirli akciğer bölgelerinde
yoğunlaşan aktivasyonlarla ilişkilendirdiğini incelemek için kullanılabilir. Özellikle
PNEUMONIA tahminlerinde sıcak bölgelerin akciğer alanları üzerinde kalıp kalmadığı kontrol
edilmelidir. Eğer ısı haritası görüntü kenarları, yazılar veya akciğer dışı alanlara
yoğunlaşıyorsa bu durum modelin klinik olarak anlamlı olmayan ipuçlarını öğrenmiş
olabileceğine işaret eder.

## Sınırlılıklar

Grad-CAM nedensel bir açıklama değildir; yalnızca modelin son evrişimsel özellikleri
üzerinden yaklaşık bir görsel yorum sağlar. Isı haritası yüksek çözünürlüklü patoloji
lokalizasyonu olarak değerlendirilmemelidir. Bu proje eğitim amaçlıdır ve üretilen
tahminler veya açıklamalar tıbbi teşhis amacıyla kullanılmamalıdır.

## Üretilen Görseller

- `reports\figures\gradcam_correct_normal_01.png`
- `reports\figures\gradcam_correct_normal_02.png`
- `reports\figures\gradcam_correct_normal_03.png`
- `reports\figures\gradcam_correct_normal_04.png`
- `reports\figures\gradcam_correct_pneumonia_01.png`
- `reports\figures\gradcam_correct_pneumonia_02.png`
- `reports\figures\gradcam_correct_pneumonia_03.png`
- `reports\figures\gradcam_correct_pneumonia_04.png`
- `reports\figures\gradcam_misclassified_01.png`
- `reports\figures\gradcam_misclassified_02.png`
- `reports\figures\gradcam_misclassified_03.png`
- `reports\figures\gradcam_misclassified_04.png`
- `reports\figures\gradcam_misclassified_05.png`
- `reports\figures\gradcam_misclassified_06.png`

## Not

6 hatalı sınıflandırılmış örnek için görsel üretildi.

## Uyarı

Bu proje eğitim amaçlıdır; çıktılar tıbbi teşhis amacıyla kullanılmamalıdır.
