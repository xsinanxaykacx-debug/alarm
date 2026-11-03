
from pathlib import Path
from ultralytics import YOLO

MODEL = Path(
    r"runs\classify\polis_model_sonuclari\siniflandirma_v1\weights\best.pt"
)
KAYNAK = Path("polis_test_fotograflari")

if not MODEL.exists():
    raise FileNotFoundError(f"Model bulunamadı: {MODEL.resolve()}")

if not KAYNAK.is_dir():
    raise FileNotFoundError(
        f"Test klasörü yok: {KAYNAK.resolve()}\n"
        "Bu klasörü oluşturup içine test fotoğrafları koy."
    )

uzantilar = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
fotograflar = [
    p for p in KAYNAK.rglob("*")
    if p.is_file() and p.suffix.lower() in uzantilar
]

if not fotograflar:
    raise RuntimeError(f"Test klasöründe fotoğraf yok: {KAYNAK.resolve()}")

model = YOLO(str(MODEL))

sonuclar = model.predict(
    source=[str(p) for p in fotograflar],
    imgsz=224,
    save=True,
    project="polis_model_sonuclari",
    name="gercek_fotograf_testi",
    exist_ok=True,
    verbose=False,
)

print("\nFOTOĞRAF TEST SONUÇLARI")
for dosya, sonuc in zip(fotograflar, sonuclar):
    en_iyi = sonuc.probs.top1
    sinif = sonuc.names[en_iyi]
    guven = float(sonuc.probs.top1conf)
    print(f"{dosya.name}\t{sinif}\t%{guven * 100:.1f}")

print("\nİşlenmiş sonuçlar:")
print(Path("polis_model_sonuclari/gercek_fotograf_testi").resolve())
