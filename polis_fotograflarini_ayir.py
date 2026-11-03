from pathlib import Path
from shutil import copy2
from collections import Counter
from ultralytics import YOLO

MODEL = Path(r"runs\classify\polis_model_sonuclari\siniflandirma_v1\weights\best.pt")
KAYNAK = Path("polis_test_fotograflari")
HEDEF = Path("polis_test_fotograflari_ayrilmis")

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model bulunamadi: {MODEL.resolve()}")
    if not KAYNAK.exists():
        raise FileNotFoundError(f"Fotograf klasoru bulunamadi: {KAYNAK.resolve()}")

    fotograflar = sorted(
        p for p in KAYNAK.rglob("*")
        if p.is_file()
        and p.suffix.lower() in UZANTILAR
        and HEDEF.resolve() not in p.resolve().parents
    )

    if not fotograflar:
        print("Fotograf bulunamadi.")
        return

    model = YOLO(str(MODEL))
    sayac = Counter()
    HEDEF.mkdir(parents=True, exist_ok=True)
    (HEDEF / "P").mkdir(exist_ok=True)
    (HEDEF / "N").mkdir(exist_ok=True)

    sonuclar = model.predict(
        source=[str(p) for p in fotograflar],
        imgsz=224,
        batch=16,
        device="cpu",
        verbose=False
    )

    for dosya, sonuc in zip(fotograflar, sonuclar):
        sinif = str(sonuc.names[sonuc.probs.top1]).strip().lower()
        guven = float(sonuc.probs.top1conf)

        if "polis" in sinif:
            klasor = HEDEF / "P"
            etiket = "POLIS"
        elif "normal" in sinif:
            klasor = HEDEF / "N"
            etiket = "NORMAL"
        else:
            print(f"ATLANDI: {dosya.name} | Bilinmeyen sinif: {sinif}")
            continue

        # Ayni isimli dosyalar varsa uzerine yazma.
        hedef_dosya = klasor / dosya.name
        if hedef_dosya.exists():
            hedef_dosya = klasor / f"{dosya.stem}_{sayac[etiket]+1}{dosya.suffix}"

        copy2(dosya, hedef_dosya)
        sayac[etiket] += 1
        print(f"{dosya.name}\t{etiket}\t%{guven * 100:.1f}")

    print("\n--- TAMAMLANDI ---")
    print(f"Modelin POLIS dedigi: {sayac['POLIS']}")
    print(f"Modelin NORMAL dedigi: {sayac['NORMAL']}")
    print(f"Ayrilan fotograflar: {HEDEF.resolve()}")
    print("Orijinal fotograflar degistirilmedi.")

if __name__ == "__main__":
    main()
