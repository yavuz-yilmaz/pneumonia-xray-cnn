# Modeli iyileştirme ve projeyi kurtarma planı — 8 Ekim 2026

Amaç, farklı veri kaynaklarında nerede başarısız olduğunu ölçebildiğimiz,
deneyleri tekrarlanabilen bir pnömoni sınıflandırma araştırması oluşturmak.
İyileşme iddiası, aynı örnekler üzerinde sabit baselinelarla karşılaştırmaya
ve gerçekten bağımsız bir son teste dayanacak. Bu plan klinik kullanım için
onay veya başarı garantisi değildir.

## Başlangıç durumu

- DINOv3, Kermany geliştirme verisinde eğitildi. Seçilen model 4. epoch;
  karar eşiği `0.6127732396125793`. Kermany doğrulama doğruluğu %98,32.
- Montgomery'deki 78 normal görüntünün 54'ünde yanlış alarm verdi:
  özgüllük %30,77. Yanlış alarmların 43'ünde pnömoni skoru 0,90'ın üstünde.
  Bağımsız pnömoni duyarlılığı henüz ölçülmedi.
- SSMU 2026 part 1'in 50 frontal görüntüsü önceki SSMU görüntüleriyle
  birebir aynı. Yeni bir dış test olarak kullanılamaz.
- Önceki röntgen ön eğitimli DenseNet121/SSMU denemesi kendi kaynağında
  %86,26 doğruluk verdi; OpenI'de %47,61 doğruluk ve 1.505 yanlış alarm
  üretti. Aynı denemeyi tekrarlamak genelleme problemini çözmüş sayılmaz.
- RSUA bulut indirmesi HTTP 403 verdi. VinDr/CheXpert için onaylı erişim yok.
- Yerel GPU: RTX 3060 Laptop, 6 GB. Küçük modeller ve mevcut özelliklerle
  sınıflandırma katmanı denemeleri yerelde yapılabilir.
- Modal defteri $23,9713 rezervasyon içeriyor; $24 çalışma sınırından
  $0,0287 kalıyor. Ayrı $4 pay korunuyor. Bu rakam gerçek fatura değildir.

Kanıtlar: [dış test raporu](dinov3_external_results.md),
[SSMU sonuçları](ssmu_model_results.md),
[veri erişimi](dataset_access_inventory.md),
[NIH/RSNA etiket incelemesi](nih_rsna_target_review.md).

## Çalışma sırası

| Aşama | Yapılacak iş | Somut çıktı | İlerleme / durma koşulu |
|---|---|---|---|
| 1. Kapsam ve bütçe | Hedef etiketleri, veri rollerini ve gerçek Modal harcamasını kaydet | Kapsam belgesi, bütçe durumu, korunacak manifest/hash listesi | Ek bulut işi ancak finanse edilmiş bir sınırla çalışabilir |
| 2. Hata incelemesi | Görüntü işleme ve kaynak farklarını sistematik kontrol et | Denetim raporu, örnek görseller, doğrulanmış hata varsa odaklı düzeltme | Hata bulunursa aynı kaynak sürümü ve modelle düzeltmeyi doğrula; sonuç yoksa neden uydurma |
| 3. Veri ve değerlendirme protokolü | Mevcut geliştirme bölümlerini koru, yeni dış kaynağı denetle | Kaynak/etiket tablosu, grup kontrolü, dondurulmuş protokol | Bağımsız pnömoni kohortu yoksa genel başarı iddiası ve pahalı denemeler bekler |
| 4. Ucuz yerel deneyler | İki sınırlı aday ailesini aynı geliştirme verisinde karşılaştır | Kaynak başına doğrulama tablosu, süre/bellek kaydı, en fazla bir finalist | Önceden belirlenen ölçütler karşılanmıyorsa deneme sayısını artırmadan başarısızlığı kaydet |
| 5. Bir defalık son test | Finalistin modelini, eşiğini ve karşılaştırma kuralını kilitle | Örnek bazlı tahminler, hata matrisi, belirsizlik aralıkları | Son test sonucundan sonra eşik/model seçme; yeni iyileştirme döngüsü yeni bağımsız test gerektirir |
| 6. Proje çıktısı | Deneyleri ve sınırları anlaşılır biçimde sun | Tekrarlama rehberi, sonuç raporu, araştırma demosu | İyileşme yoksa sonuç negatif deney olarak sunulur |

## 1. Kapsamı ve bütçeyi kilitle

Birincil araştırma hedefi frontal röntgenlerde yayıncı etiketlerine göre
NORMAL/PNEUMONIA ayrımıdır. Bu, diğer bütün hastalıklar arasında klinik
pnömoni tanısı koymakla aynı görev değildir. Diğer akciğer hastalıklarını
içeren genişletilmiş negatif sınıf ancak uyumlu pnömoni etiketleri ve
güvenilir kaynak bilgisiyle ayrı bir protokol olarak eklenebilir.

Mevcut DINOv3 checkpoint'i ve eşiği referans olarak korunacak. Eski
Kermany/SSMU/NIH testleri, OpenI ve Montgomery gözlemlenmiş kohortlar
olarak kaydedilecek. Bunları yeniden bölmek, süzmek veya yeniden adlandırmak
yeni bağımsız test oluşturmaz. Eski test üyeleri yeni eğitim verisine alınmaz.

Modal'da gerçek ücret, kredi düşümü öncesindeki kullanım ve devam eden
maliyetler doğrulanacak. Geçmiş rezervasyonlar silinmeyecek veya sıfırlanmayacak;
düşük fatura tek başına mevcut defterde harcama izni oluşturmaz. Yeni bulut
işleri için toplam $28 sınırı ve ayrı $4 pay korunarak, doğrulanmış faturaya
dayanan ve eski kayıtlarla ilişkisi açık bir bütçe muhasebesi gerekir.
Bu koşul sağlanmazsa çalışma yerel inceleme/denemelerle sınırlı kalır.

## 2. Önce hatanın kaynağını araştır

İlk inceleme mevcut tahminler ve geliştirme verileriyle yapılacak:

- Eğitim, doğrulama ve dış testte renk kanalları, görüntü yönü, piksel aralığı,
  yeniden boyutlandırma, EXIF ve normalizasyonun tutarlılığını kontrol et.
  Montgomery'de 8-bit dönüşümün doğrulandığını koru; tekrar bir dönüşüm
  hatası varmış gibi varsayma.
- Pnömoni skorunu gerçek doğruluk olasılığı gibi yorumlama. Yanlış pozitif
  ve yanlış negatif dağılımlarını kaynak başına göster.
- Geliştirme görüntülerinde çerçeve, yazı/işaretler, çekim biçimi ve parlaklık
  farklarını incele. Yaşı yalnızca güvenilir metadata varsa kullan.
- Grad-CAM/attention görsellerini yardımcı inceleme olarak kullan;
  bunları tek başına hastane işaretlerine dayanıldığının kanıtı sayma.
- Ön işlem değişikliklerini veya kırpma deneylerini küçük, sabit geliştirme
  örneklerinde dene. Tanısal alanı bozabilecek dönüşümleri otomatik benimseme.

Somut hata bulunursa en küçük düzeltme yapılır ve odaklı kontrolle doğrulanır.
Hata bulunmazsa gözlenen kaynak farkı raporlanır; kesin neden belirlenmiş
gibi sunulmaz. Yeni büyük görüntü/model indirmesi yerel incelemenin varsayılanı
değildir. Gerekli bulut örneklemeleri önce bütçelendirilir, küçük raporlar alınır.

## 3. Veriyi ve son testi güvenilir hale getir

| Kaynak | Bu plandaki rolü |
|---|---|
| Kermany'nin mevcut train/val bölümleri | Geliştirme; mevcut grupları ve test ayrımını koru |
| SSMU'nun mevcut 665 train / 133 val bölümleri | Etiket ve kimlik sınırları açıkça belirtilen ikinci geliştirme kaynağı; 131 görüntülük gözlemlenmiş test korunur |
| NIH / RSNA | Etiketleri farklı yardımcı görevler; opacity'yi pnömoni, No Finding'i doğrulanmış sağlıklılık diye dönüştürme |
| Montgomery / OpenI / eski testler | Gözlemlenmiş tanısal/stres kohortları; tekrar kullanım varsa keşif amaçlı olduğunu belirt |
| SSMU 2026 part 1 | Birebir tekrarlar nedeniyle dış testten hariç |
| RSUA veya başka uygun yeni kaynak | Erişim, lisans, etiket, görüntü türü ve tekrar denetiminden geçerse son test adayı |

Yeni kaynak için yayıncı/sürüm/checksum, pnömoni etiketinin nasıl üretildiği,
görüntü-hasta ilişkisi, normal/diğer hastalık kapsamı ve lisans kaydedilecek.
Orijinal röntgenler maskelerden, üstüne çizilmiş görüntülerden ve tekrar
temsillerden ayrılacak. Dosya/piksel hashleri ve yakın görüntü kontrolü bütün
bilinen kaynaklara karşı uygulanacak; şüpheli yakın eşleşmeler incelenecek.

Kaynak gerçekten hasta kimliği veriyorsa hasta bazında ayrılacak. Vermezse
korumacı vekil gruplar kullanılabilir, fakat hasta bağımsızlığı iddia edilemez.
Her geliştirme kaynağında her iki sınıf bulunması sağlanacak; sınıf ile kaynak
birbirini otomatik olarak tanımlamamalı.

Tercih edilen yeni son test en az 100 pnömoni ve 100 uygun negatif örnek
içeren, belgelenmiş kimlik/etiket ilişkisine sahip bir kaynaktır. Bu bir erişim
ve örneklem hedefidir, klinik yeterlilik eşiği değildir. RSUA'nın 53/32 örneği
erişilebilir olursa daha küçük, belirsizliği yüksek bir pilot sağlayabilir.
Kaynak adı değişmesi tek başına hastane veya hasta bağımsızlığını kanıtlamaz.
Temel modelin ön eğitim verileriyle olası örtüşme de ayrıca belirtilir.

SSMU/RSUA kimlik ve etiket açıklaması gerekiyorsa soru taslağı hazırlanabilir;
bu plan sağlayıcıya mesaj gönderme talimatı içermez. Onaysız erişim gerektiren
kaynaklar, lisansı belirsiz aynalar ve daha önce reddedilen veriler kullanılmaz.

## 4. En fazla iki ucuz aday ailesi

İlk bütçe dostu karşılaştırma yerel RTX 3060 üzerinde, mevcut yerel veri ve
ağırlıklarla yapılır. DINOv3'ün buluttaki yaklaşık 1,21 GB checkpoint'i sırf
bu karşılaştırma için yerel bilgisayara indirilmez.

1. **Mevcut röntgen ön eğitimli DenseNet121 temsili + yeni sınıflandırma katmanı.**
   Kermany ve SSMU'nun izin verilen geliştirme bölümleri için kaynak başına
   uygun ön işlemle özellikler çıkarılır ve önbelleğe alınır. Karışık kaynakla
   yeni katman eğitilir; geçmiş SSMU'ya özgü deneyin tekrarı diye sunulmaz.
2. **ResNet18 ile aynı çok kaynaklı geliştirme karşılaştırması.**
   224px ve sınırlı batch ile pilot bellek/süre ölçülür. Aynı train/val
   örnekleri, kaynak ağırlıkları, model seçme kuralı ve eşik politikası kullanılır.

Ön eğitim/ön işlem farkları kaydedilir; böylece sonuç mimari farkından ibaret
diye yorumlanmaz. Mevcut ResNet18 ve SSMU modeli sabit referanslardır. DINOv3
referansı aynı yeni kohortta ancak bütçeli bulut değerlendirmesi yapılabilirse
karşılaştırma tablosuna girer. Farklı kohortlardaki eski skorlar eşit koşullu
karşılaştırmanın yerine geçmez.

Kaynak/sınıf dengesini yalnızca eğitim bölümünde düzenle. Aynı hastanın veya
bilinen tekrar grubunun bütün görüntülerini aynı rolde tut. SSMU vekil
gruplarının sınırlarını koru; metadata eksikliğini model başarısıyla gizleme.

Pilot tarifleri, seed, epoch üst sınırı, erken durma ve en fazla üç katman
regularizasyon seçeneği başlamadan kaydedilecek. Deneme sayısı sonuca bakarak
genişletilmeyecek. Pahalı DINOv3 adaptasyonu ancak yerel denemeler ilerleme
gösterir, bağımsız son test erişimi açılır ve bütçe kapısı geçilirse düşünülür.

## 5. Başarı ölçümü ve bir defalık son test

İlk prototip için önerilen araştırma hedefi, uygun ikili doğrulama
kaynaklarının **her birinde duyarlılık ≥%90 ve özgüllük ≥%80**.
Bu klinik bir kullanım standardı değildir. Hedefler, etiket görevi ve seçim
kuralı ilk yeni tahminlerden önce protokolde kesinleştirilecek; başarısız
sonuçtan sonra gevşetilmeyecek.

Bir tane ortak karar eşiği doğrulama verisinde seçilecek. İki hedefi sağlayan
eşikler arasında en düşük kaynak özgüllüğünü en yükseğe çıkaran eşik seçilir;
eşitlikte daha yüksek eşik tercih edilir. Uygun eşik yoksa aday başarısızdır.
Adaylar bu en düşük kaynak özgüllüğüne göre seçilir; eşitlikte daha düşük
çalıştırma maliyeti, sonra önceden belirlenmiş aday sırası kullanılır.
Kaynağa ayrı eşik verilerek kötü transfer görünmez hale getirilmeyecek.

Kermany + SSMU karışık eğitimindeki doğrulama, eğitimde görülen kaynakların
doğrulamasıdır. Hastane dışı başarı diye sunulmaz. Kaynak dışarıda bırakılarak
yapılan kontroller ayrı geliştirme deneyleri olarak kaydedilir; yaş/çekim/etiket
farkları korunur ve geçmiş test sonuçlarıyla model seçilmez.

Son testten önce finalist, checkpoint hash, eşik, ön işlem, örnek listesi ve
baseline karşılaştırma kuralı kilitlenir. Sabit baselineların özgün eşikleri
ve varsa ortak yeni doğrulama politikası ayrı raporlanır; kalibrasyon kazancı
mimari kazancı diye sunulmaz.

Her kaynak için duyarlılık, özgüllük, yanlış alarm/kaçırma sayısı, precision,
F1, ROC-AUC ve PR-AUC raporlanır. Tek sınıflı veri yalnız ilgili sınıf oranını
destekler. Hasta bazlı belirsizlik veya kimlik yoksa açıkça vekil/image düzeyinde
belirsizlik verilir. Precision/F1'in prevalansa bağlı olduğu belirtilir;
farklı hastanelerin olumlu/olumsuz sınıfları yapay bir ikili teste birleştirilmez.

Son testte hedeflerin nokta tahminleri, güven aralıkları ve baseline farkları
birlikte sunulur. Küçük örneklemde nokta kazanımı tek başına sağlam iyileşme
kanıtı sayılmaz. Başarısız son test, eşik ayarlama veya başka finalisti seçme
gerekçesi olamaz; yeni döngü için gerçekten yeni son test gerekir.

## Takvim, bütçe ve teslim

Erişim beklemesi hariç yaklaşık **3–5 çalışma oturumu** hedeflenir:

- İlk oturum: kapsam/bütçe kaydı ve mevcut hata incelemesi.
- İkinci oturum: kaynak rolleri, etiket/tekrar denetimi ve deney protokolü.
- Sonraki bir veya iki oturum: sınırlı yerel pilotlar ve finalist seçimi.
- Son oturum: erişim/bütçe varsa kilitli son test, ardından rapor ve demo.

Yerel işler için ek Modal harcaması yoktur; çalışma süresi pilotta ölçülür.
Yeni veri bulma veya güvenilir hasta bilgisi alma süresi bu takvime dahil
değildir. Erişim olmadan yerel keşif yapılabilir, ancak dış genelleme iddiası
ve pahalı eğitime geçiş kapısı kapalı kalır.

İlk teslim: hata incelemesi, kaynak/rol tablosu ve kayıtlı iki aday protokolü.
Son teslim: başarılı ve başarısız bütün adaylar, kaynak bazında metrikler,
ön işlem/etiket/kimlik sınırları ve tekrarlama komutlarını içeren rapor.
Demo araştırma kapsamını ve model skorlarının sınırlarını gösterir;
modelin hazır bir tıbbi karar sistemi olduğu izlenimi verilmez.

Bu plan yazıldığı sırada yeni eğitim, GPU işi, büyük veri/model transferi,
checkpoint değişimi veya sağlayıcı iletişimi başlatılmadı.
