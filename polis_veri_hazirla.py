from pathlib import Path
from collections import defaultdict
import random
import shutil
import hashlib

KAYNAK = Path("polis_model_verisi/incele")
HEDEF = Path("polis_dataset")

SINIFLAR = {
    "POLIS": "polis",
    "NORMAL": "normal",
}
UZANTILAR = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
random.seed(42)

def hash_dosya(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for parca in iter(lambda: f.read(1024 * 1024), b""):
            h.update(parca)
    return h.hexdigest()

def main():
    if not KAYNAK.exists():
        raise FileNotFoundError(
            f"Kaynak klasoru bulunamadi: {KAYNAK.resolve()}"
        )

    HEDEF.mkdir(parents=True, exist_ok=True)
    bulunan = defaultdict(list)
    hashler = {}

    for klasor, sinif in SINIFLAR.items():
        klasor_yolu = KAYNAK / klasor
        if not klasor_yolu.exists():
            raise FileNotFoundError(
                f"Sinif klasoru bulunamadi: {klasor_yolu.resolve()}"
            )

        for dosya in klasor_yolu.rglob("*"):
            if not dosya.is_file() or dosya.suffix.lower() not in UZANTILAR:
                continue

            try:
                h = hash_dosya(dosya)
            except OSError as e:
                print("Okunamadi:", dosya, e)
                continue

            if h in hashler:
                print("Ayni dosyanin kopyasi atlandi:", dosya.name)
                continue

            hashler[h] = dosya
            bulunan[sinif].append(dosya)

    for sinif, dosyalar in bulunan.items():
        random.shuffle(dosyalar)
        n = len(dosyalar)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        bolumler = {
            "train": dosyalar[:n_train],
            "val": dosyalar[n_train:n_train + n_val],
            "test": dosyalar[n_train + n_val:],
        }

        for bolum, liste in bolumler.items():
            hedef_klasor = HEDEF / bolum / sinif
            hedef_klasor.mkdir(parents=True, exist_ok=True)

            for i, kaynak in enumerate(liste, 1):
                hedef_dosya = hedef_klasor / kaynak.name

                # Ayni isimli farkli dosyalarin ustune yazma.
                if hedef_dosya.exists():
                    hedef_dosya = hedef_klasor / f"{i}_{kaynak.name}"

                shutil.copy2(kaynak, hedef_dosya)

            print(f"{bolum:5} | {sinif:6} | {len(liste):4} foto")

    print("\nVeri hazir:", HEDEF.resolve())
    print("Orijinal fotoğraflar korunmustur.")
    print("Not: Birebir ayni dosyalar ayiklandi; benzer goruntuler")
    print("ve ayni aracin farkli kareleri ayrica kontrol edilmelidir.")

if __name__ == "__main__":
    main()
