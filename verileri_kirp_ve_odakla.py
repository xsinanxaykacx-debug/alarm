import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================
# AYARLAR
# ============================================================
# Senin bilgisayarındaki gerçek yollar (Terminal çıktısına göre)
ANA_KLASOR = Path(r"C:\Users\bzdye\Downloads\platform-tools-latest-windows\platform-tools")
CIKTI_KLASOR = ANA_KLASOR / "polis_dataset_v3"

# Görsellerdeki klasör isimlerine göre:
# Polis: polis_dataset_v2/train/polis
# Normal: polis_dataset/train/normal
POLIS_KLASOR = ANA_KLASOR / "polis_dataset_v2" / "train" / "polis"
NORMAL_KLASOR = ANA_KLASOR / "polis_dataset" / "train" / "normal"

# YOLO'nun araç sınıfları (car:2, motorcycle:3, bus:5, truck:7)
ARAC_SINIFLARI = [2, 3, 5, 7]

print("Model yükleniyor...")
model = YOLO("yolo11n.pt")

# Kırpılmış resimlerin kaydedileceği klasörleri oluştur
(CIKTI_KLASOR / "train" / "polis").mkdir(parents=True, exist_ok=True)
(CIKTI_KLASOR / "train" / "normal").mkdir(parents=True, exist_ok=True)

def isle_ve_kirp(kaynak_klasor, hedef_sinif_adi):
    if not kaynak_klasor.exists():
        print(f"❌ HATA: Klasör bulunamadı: {kaynak_klasor}")
        return
        
    # Sadece resim dosyalarını al (jpg, jpeg, png)
    uzantilar = {".jpg", ".jpeg", ".png"}
    resimler = [p for p in kaynak_klasor.rglob("*") if p.suffix.lower() in uzantilar]
    
    print(f"\n{hedef_sinif_adi.upper()} sınıfı işleniyor ({len(resimler)} resim)...")
    
    basarili = 0
    araç_bulunamayan = 0
    
    for i, resim_yolu in enumerate(tqdm(resimler)):
        img = cv2.imread(str(resim_yolu))
        if img is None:
            continue
            
        # Görselde araç ara
        results = model.predict(img, conf=0.3, classes=ARAC_SINIFLARI, verbose=False)
        
        if len(results[0].boxes) > 0:
            # En büyük (kameraya en yakın) aracı bul
            en_buyuk_alan = 0
            en_iyi_kutu = None
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                alan = (x2 - x1) * (y2 - y1)
                if alan > en_buyuk_alan:
                    en_buyuk_alan = alan
                    en_iyi_kutu = (x1, y1, x2, y2)
            
            # Kırpma işlemi (kenarlardan %10 pay bırakalım)
            x1, y1, x2, y2 = en_iyi_kutu
            h, w = img.shape[:2]
            
            pay_x = int((x2 - x1) * 0.1)
            pay_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - pay_x)
            y1 = max(0, y1 - pay_y)
            x2 = min(w, x2 + pay_x)
            y2 = min(h, y2 + pay_y)
            
            kirpilmis_img = img[y1:y2, x1:x2]
            
            hedef_dosya = CIKTI_KLASOR / "train" / hedef_sinif_adi / f"{hedef_sinif_adi}_{basarili:04d}.jpg"
            cv2.imwrite(str(hedef_dosya), kirpilmis_img)
            basarili += 1
        else:
            # Araç bulunamazsa orijinalini kaydet
            hedef_dosya = CIKTI_KLASOR / "train" / hedef_sinif_adi / f"ORIJINAL_{hedef_sinif_adi}_{araç_bulunamayan:04d}.jpg"
            cv2.imwrite(str(hedef_dosya), img)
            araç_bulunamayan += 1

    print(f"✅ {hedef_sinif_adi.upper()}: {basarili} araç başarıyla kırpıldı, {araç_bulunamayan} resimde araç bulunamadı.")

# Çalıştır
isle_ve_kirp(POLIS_KLASOR, "polis")
isle_ve_kirp(NORMAL_KLASOR, "normal")

print(f"\nİşlem tamamlandı! Yeni veri seti: {CIKTI_KLASOR}")