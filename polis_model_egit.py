from pathlib import Path
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
import hashlib
import random
import shutil
import sys

# ============================================================
# AYARLAR
# ============================================================

# Kırptığımız yeni veri setinin yolu (Senin bilgisayarına göre)
KAYNAK = Path(r"C:\Users\bzdye\Downloads\platform-tools-latest-windows\platform-tools\polis_dataset_v3\train")

# Yeni oluşturulacak veri seti klasörü
CIKTI = Path(r"C:\Users\bzdye\Downloads\platform-tools-latest-windows\platform-tools\polis_dataset_egitim")

MODEL = "yolo11n-cls.pt"

TOHUM = 42
EGITIM_EPOCH = 50
GORUNTU_BOYUTU = 224
BATCH = 16

SINIFLAR = {
    "polis": "polis",
    "normal": "normal",
}

random.seed(TOHUM)

def dosya_hashi(dosya):
    sha = hashlib.sha256()
    with dosya.open("rb") as f:
        for parca in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(parca)
    return sha.hexdigest()

def gorsel_gecerli_mi(dosya):
    try:
        with Image.open(dosya) as img:
            img.verify()
        with Image.open(dosya) as img:
            return img.width >= 64 and img.height >= 64
    except Exception:
        return False

def ana():
    if not KAYNAK.is_dir():
        sys.exit(f"HATA: Kaynak klasör bulunamadı: {KAYNAK.resolve()}")

    kaynaklar = {}
    for klasor, sinif in SINIFLAR.items():
        yol = KAYNAK / klasor
        if not yol.is_dir():
            sys.exit(f"HATA: Sınıf klasörü bulunamadı: {yol.resolve()}")
        kaynaklar[sinif] = yol

    hash_siniflari = defaultdict(set)
    hash_dosyalari = defaultdict(list)
    bozuk = []
    taranan = 0

    print("\n[1/5] Fotoğraflar taranıyor...")

    for sinif, klasor in kaynaklar.items():
        dosyalar = [p for p in klasor.rglob("*") if p.is_file()]
        for dosya in dosyalar:
            taranan += 1
            if not gorsel_gecerli_mi(dosya):
                bozuk.append(str(dosya))
                continue
            try:
                h = dosya_hashi(dosya)
            except OSError:
                bozuk.append(str(dosya))
                continue
            hash_siniflari[h].add(sinif)
            hash_dosyalari[h].append((sinif, dosya))

    celiskili_hashler = {
        h for h, siniflar in hash_siniflari.items()
        if len(siniflar) > 1
    }

    benzersiz = {"polis": [], "normal": []}
    sinif_ici_tekrar = 0

    for h, kayitlar in hash_dosyalari.items():
        if h in celiskili_hashler:
            continue
        sinif, dosya = kayitlar[0]
        benzersiz[sinif].append(dosya)
        sinif_ici_tekrar += len(kayitlar) - 1

    print(f"Taranan dosya             : {taranan}")
    print(f"Bozuk/geçersiz görüntü     : {len(bozuk)}")
    print(f"Birebir tekrar            : {sinif_ici_tekrar}")
    print(f"Etiket çelişkili görüntü   : {len(celiskili_hashler)}")

    for sinif, dosyalar in benzersiz.items():
        print(f"Benzersiz {sinif:6s} görüntü     : {len(dosyalar)}")

    if min(len(benzersiz["polis"]), len(benzersiz["normal"])) < 10:
        sys.exit("HATA: Eğitim için sınıflardan birinde 10'dan az görüntü var.")

    print("\n[2/5] Yeni veri seti hazırlanıyor...")

    if CIKTI.exists():
        sys.exit(
            f"HATA: {CIKTI} zaten var. Eski dosyaları korumak için "
            "otomatik silinmedi. Çıktı klasörünü yeniden adlandır "
            "veya farklı bir CIKTI adı belirle."
        )

    for bolum in ("train", "val", "test"):
        for sinif in ("polis", "normal"):
            (CIKTI / bolum / sinif).mkdir(parents=True, exist_ok=True)

    sayilar = {}

    for sinif, dosyalar in benzersiz.items():
        dosyalar = dosyalar.copy()
        random.shuffle(dosyalar)

        toplam = len(dosyalar)
        n_train = int(toplam * 0.80)
        n_val = int(toplam * 0.10)

        parcalar = {
            "train": dosyalar[:n_train],
            "val": dosyalar[n_train:n_train + n_val],
            "test": dosyalar[n_train + n_val:],
        }

        for bolum, liste in parcalar.items():
            hedef_klasor = CIKTI / bolum / sinif
            for i, kaynak in enumerate(liste, start=1):
                hedef = hedef_klasor / f"{sinif}_{i:05d}{kaynak.suffix.lower()}"
                shutil.copy2(kaynak, hedef)
            sayilar[(bolum, sinif)] = len(liste)

    print("\n[3/5] Veri dağılımı:")
    for bolum in ("train", "val", "test"):
        for sinif in ("polis", "normal"):
            print(f"{bolum:5s} | {sinif:6s} | {sayilar[(bolum, sinif)]} foto")

    print(f"\nYeni veri seti: {CIKTI.resolve()}")

    # Modeli eğit.
    print("\n[4/5] Model yükleniyor ve eğitim başlıyor...")
    model = YOLO(MODEL)

    sonuc = model.train(
        data=str(CIKTI.resolve()),
        epochs=EGITIM_EPOCH,
        imgsz=GORUNTU_BOYUTU,
        batch=BATCH,
        patience=10,
        seed=TOHUM,
        workers=0,
        project="polis_model_sonuclari",
        name="kirpilmis_model_v1", # Yeni isim verdik ki eski modeli ezmesin
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\nEğitim tamamlandı.")
    print(f"Sonuç klasörü: {sonuc.save_dir}")

    # Test kümesiyle son değerlendirme.
    print("\n[5/5] Ayrı test kümesi değerlendiriliyor...")
    en_iyi_model = Path(sonuc.save_dir) / "weights" / "best.pt"

    if not en_iyi_model.exists():
        print(f"UYARI: En iyi model dosyası bulunamadı: {en_iyi_model}")
        return

    test_modeli = YOLO(str(en_iyi_model))
    test_sonuc = test_modeli.val(
        data=str(CIKTI.resolve()),
        split="test",
        imgsz=GORUNTU_BOYUTU,
        batch=BATCH,
        workers=0,
        plots=True,
    )

    print("\nTEST TAMAMLANDI")
    print(f"En iyi model: {en_iyi_model.resolve()}")
    print(f"Test sonuçları: {test_sonuc.save_dir}")
    print("Not: Sonuçlar, gerçek sokak performansını garanti etmez.")

if __name__ == "__main__":
    ana()