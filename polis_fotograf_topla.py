import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# Gerekli paket
if importlib.util.find_spec("icrawler") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "icrawler"
    ])

from icrawler.builtin import BingImageCrawler

ROOT = Path("polis_model_verisi")
ARAMALAR = {
    "POLIS": [
        "Türkiye polis otosu",
        "Türk polisi ekip aracı",
        "Türkiye trafik polisi aracı"
    ],
    "NORMAL": [
        "Türkiye binek otomobil sokakta",
        "Türkiye trafikte normal otomobil",
        "Türkiye sedan otomobil"
    ]
}

for sinif, sorgular in ARAMALAR.items():
    hedef = ROOT / "incele" / sinif
    hedef.mkdir(parents=True, exist_ok=True)

    mevcut = len([
        p for p in hedef.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    for sorgu_no, sorgu in enumerate(sorgular, 1):
        kalan = max(0, 30 - mevcut)
        if kalan == 0:
            break

        print(f"\nAraniyor: {sorgu}")
        crawler = BingImageCrawler(
            storage={"root_dir": str(hedef)}
        )
        crawler.crawl(
            keyword=sorgu,
            max_num=kalan,
            min_size=(200, 150),
            file_idx_offset=mevcut
        )

        mevcut = len([
            p for p in hedef.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])

print("\nIslem tamamlandi.")
print("Fotoğrafları kontrol et:", (ROOT / "incele").resolve())
print("POLIS klasorundeki yanlis fotograflari sil.")
print("NORMAL klasorundeki polis araclarini sil.")
print("BELIRSIZ fotograflari egitimde kullanma.")
