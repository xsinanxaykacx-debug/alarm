
import cv2
import re
import time
import json
import os
import uuid
import unicodedata
import winsound
import threading
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# YOLLAR VE AYARLAR
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

KAMERA_NO = 1
ALGILAMA_MODELI = "yolo11n.pt"
SINIFLANDIRMA_MODELI = str(
    SCRIPT_DIR / "runs" / "classify" / "polis_model_sonuclari"
    / "siniflandirma_v1" / "weights" / "best.pt"
)

KAMERA_GENISLIGI = 1920
KAMERA_YUKSEKLIGI = 1080
ALGILAMA_GUVEN_ESIGI = 0.35
ALGILAMA_ARALIGI = 0.35
GORUNTU_GENISLIGI = 640
SINIFLANDIRMA_ESIGI = 0.65

P_STREAK_REQUIRED = 4
P_STREAK_K_TOLERANCE = 1
P_STREAK_MAX_WINDOW = 6.0
P_STREAK_HARD_TIME_CAP = 10.0

OCR_ARALIGI = 1.0
OCR_MIN_GUVEN = 0.45
OCR_MIN_KELIME_UZUNLUGU = 4
OCR_CONFIRM_COUNT = 2
OCR_CONFIRM_WINDOW = 3.5
OCR_MISS_TOLERANCE = 1
OCR_TEXT_MATCH_RATIO = 0.50

TRACK_REUSE_MIN_GAP = 2.0
TRACK_REUSE_MAX_IOU = 0.15
TRACK_REUSE_SHORT_GAP = 0.4
TRACK_REUSE_SHORT_IOU = 0.02
TRACK_REUSE_MAX_SPEED = 2000
TRACK_REUSE_SIZE_MIN = 0.30
TRACK_REUSE_SIZE_MAX = 3.00

CLASSIFY_INTERVAL = 0.7
TEKRAR_HAZIRLANMA = 3.0
TRACK_TIMEOUT = 5.0
ALARM_OTOMATIK_KAPATMA = 10.0
TRACK_CLEANUP_ARALIGI = 2.0
ALARM_SESI_TEKRAR = 3

DETECTION_STALE_THRESHOLD = 2.0

CAMERA_MAX_FAIL = 15
CAMERA_REOPEN_MAX_ATTEMPTS = 5
CAMERA_REOPEN_COOLDOWN = 30.0

REARM_COOLDOWN = 15.0
BATCH_APPROVE_CONFIRM_WINDOW = 3.0

CIKTI_KLASORU = SCRIPT_DIR / "arac_fotograflari"
CIKTI_KLASORU.mkdir(parents=True, exist_ok=True)

LOG_DOSYASI = CIKTI_KLASORU / "polis_gelenler.log"
ALARM_GECMIS_JSONL = CIKTI_KLASORU / "alarm_gecmisi.jsonl"
LOG_MAX_BYTES = 5_000_000
LOG_BACKUP = 3

ARACLAR = {
    2: "ARABA",
    3: "MOTOSIKLET",
    5: "OTOBUS",
    7: "KAMYON",
}

kamera = None

# ============================================================
# LOG
# ============================================================

_logger = logging.getLogger("polis_kapi")
_logger.setLevel(logging.INFO)
_logger.propagate = False

if not _logger.handlers:
    _handler = RotatingFileHandler(
        str(LOG_DOSYASI),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP,
        encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _logger.addHandler(_handler)

_jsonl_hata = {
    "count": 0,
    "consecutive": 0,
    "last_t": 0.0,
}

_kamera_hata = {
    "okuma_fail": 0,
    "reopen_attempts": 0,
    "son_reopen_t": 0.0,
    "durum": "OK",
    "mesaj": "",
}


# ============================================================
# OCR SÖZLÜKLERİ
# ============================================================

POLIS_TEK_KELIMELER = {
    "POLIS", "POLISI", "POLISLER", "POLISLERI", "POLISARACI",
    "TRAFIKPOLISI", "TRAFIKPOLIS", "TRAFIKPOLISLERI",
    "ASAYIS", "ASAYII", "ASAYISI", "ASAYILLER",
}

POLIS_KOMBINASYONLAR = [
    {"TRAFIK", "POLISI"},
    {"TRAFIK", "POLIS"},
    {"TRAFIK", "POLISLER"},
    {"TRAFIK", "POLISLERI"},
    {"TRAFIK", "ARACI"},
    {"EMNIYET", "POLISI"},
    {"EMNIYET", "POLIS"},
    {"EMNIYET", "ARACI"},
    {"EMNIYET", "MUDURLUGU"},
    {"EMNIYET", "MUDURLUK"},
    {"EMNIYET", "GENEL"},
    {"ASAYIS", "ARACI"},
    {"ASAYIS", "SUBESI"},
    {"ASAYIS", "MUDURLUGU"},
    {"ASAYIS", "EKIBI"},
    {"ASAYII", "ARACI"},
    {"POLIS", "ARACI"},
]


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def metni_normalize_et(metin):
    metin = str(metin).upper()
    metin = metin.replace("İ", "I").replace("ı", "I")
    metin = unicodedata.normalize("NFKD", metin)
    metin = "".join(
        c for c in metin
        if not unicodedata.combining(c)
    )
    metin = re.sub(r"[^A-Z0-9 ]", " ", metin)
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


def gece_iyilestir(goruntu):
    if goruntu is None or goruntu.size == 0:
        return goruntu

    lab = cv2.cvtColor(goruntu, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    l = clahe.apply(l)
    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR,
    )


def _iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    alan_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    alan_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    birlesim = alan_a + alan_b - inter

    return inter / birlesim if birlesim > 0 else 0.0


def _merkez(box):
    return (
        (box[0] + box[2]) / 2,
        (box[1] + box[3]) / 2,
    )


def _alan(box):
    return (
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )


def _aid_kisa(aid):
    return "A" + str(aid)[:8]


def yeni_alarm_id():
    return uuid.uuid4().hex[:8]


def metin_benzerlik(m1, m2):
    if not m1 or not m2:
        return 0.0

    w1 = set(str(m1).split())
    w2 = set(str(m2).split())

    if not w1 or not w2:
        return 0.0

    birlesim = len(w1 | w2)
    return len(w1 & w2) / birlesim if birlesim else 0.0


def jsonl_yaz(event_dict):
    """JSONL olayını güvenli biçimde ekler ve başarı durumunu döndürür."""
    try:
        with open(ALARM_GECMIS_JSONL, "a", encoding="utf-8") as dosya:
            dosya.write(
                json.dumps(event_dict, ensure_ascii=False, default=str)
                + "\n"
            )
            dosya.flush()
            try:
                os.fsync(dosya.fileno())
            except OSError:
                pass

        _jsonl_hata["consecutive"] = 0
        return True

    except Exception as hata:
        _jsonl_hata["count"] += 1
        _jsonl_hata["consecutive"] += 1
        _jsonl_hata["last_t"] = time.monotonic()

        mesaj = f"JSONL YAZMA HATASI: {hata}"
        print(mesaj)
        _logger.error(mesaj)
        return False


def olay_yaz(event, **alanlar):
    kayit = {
        "event": event,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    kayit.update(alanlar)
    return jsonl_yaz(kayit)


# ============================================================
# KAMERA YÖNETİMİ
# ============================================================

def kamera_ac():
    """Kamerayı açar, ayarları yapar ve ilk kareyi doğrular."""
    global kamera

    if kamera is not None:
        try:
            kamera.release()
        except Exception:
            pass
        kamera = None

    yeni_kamera = None

    try:
        yeni_kamera = cv2.VideoCapture(
            KAMERA_NO,
            cv2.CAP_DSHOW,
        )

        if yeni_kamera is None or not yeni_kamera.isOpened():
            _kamera_hata["mesaj"] = (
                "Kamera açılamadı (isOpened=False)"
            )
            if yeni_kamera is not None:
                yeni_kamera.release()
            return False

        yeni_kamera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        yeni_kamera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            KAMERA_GENISLIGI,
        )
        yeni_kamera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            KAMERA_YUKSEKLIGI,
        )

        ok, kare = yeni_kamera.read()
        if not ok or kare is None:
            _kamera_hata["mesaj"] = "İlk kare alınamadı"
            yeni_kamera.release()
            return False

        kamera = yeni_kamera
        _kamera_hata["mesaj"] = ""
        _kamera_hata["durum"] = "OK"
        return True

    except Exception as hata:
        _kamera_hata["mesaj"] = f"VideoCapture hatası: {hata}"
        try:
            if yeni_kamera is not None:
                yeni_kamera.release()
        except Exception:
            pass
        kamera = None
        return False


# ============================================================
# OCR — 3 DURUMLU SONUÇ: P / N / E
# ============================================================

def polis_yazisi_var_mi(ocr, arac_goruntusu):
    try:
        if arac_goruntusu is None or arac_goruntusu.size == 0:
            return "E", None

        yukseklik, genislik = arac_goruntusu.shape[:2]
        if genislik < 2 or yukseklik < 2:
            return "E", None

        buyutulmus = cv2.resize(
            arac_goruntusu,
            None,
            fx=1.8,
            fy=1.8,
            interpolation=cv2.INTER_CUBIC,
        )
        gri = cv2.cvtColor(buyutulmus, cv2.COLOR_BGR2GRAY)
        iyilestirilmis = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gri)

    except Exception as hata:
        _logger.warning(f"OCR ÖN İŞLEME HATASI: {hata}")
        return "E", None

    try:
        sonuc, _ = ocr(iyilestirilmis)
    except Exception as hata:
        _logger.warning(f"OCR ÇAĞRI HATASI: {hata}")
        return "E", None

    if not sonuc:
        return "N", None

    okunan = []

    for satir in sonuc:
        if not satir or len(satir) < 3:
            continue

        try:
            guven = float(satir[2])
        except (TypeError, ValueError):
            continue

        if guven < OCR_MIN_GUVEN:
            continue

        normalize = metni_normalize_et(str(satir[1]))
        if len(normalize.replace(" ", "")) >= OCR_MIN_KELIME_UZUNLUGU:
            okunan.append(normalize)

    tam_metin = " ".join(okunan)

    if not tam_metin:
        return "N", None

    kelimeler = set(tam_metin.split())

    if kelimeler & POLIS_TEK_KELIMELER:
        return "P", tam_metin

    for kume in POLIS_KOMBINASYONLAR:
        if kume.issubset(kelimeler):
            return "P", tam_metin

    return "N", tam_metin


# ============================================================
# SINIFLANDIRMA
# ============================================================

def siniflandir_arac(model, kare, x1, y1, x2, y2):
    yukseklik, genislik = kare.shape[:2]

    x1 = max(0, min(genislik - 1, int(x1)))
    x2 = max(0, min(genislik, int(x2)))
    y1 = max(0, min(yukseklik - 1, int(y1)))
    y2 = max(0, min(yukseklik, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return "K", 0.0

    kirpilan = kare[y1:y2, x1:x2]
    if kirpilan.size == 0:
        return "K", 0.0

    try:
        sonuclar = model.predict(
            source=kirpilan,
            imgsz=224,
            device="cpu",
            verbose=False,
        )
        if not sonuclar:
            return "K", 0.0

        sonuc = sonuclar[0]
        if sonuc.probs is None:
            return "K", 0.0

        sinif_id = int(sonuc.probs.top1)
        guven = float(sonuc.probs.top1conf.item())
        sinif_adi = str(
            sonuc.names[sinif_id]
        ).strip().lower()

    except Exception as hata:
        _logger.error(f"SINIFLANDIRMA HATASI: {hata}")
        return "K", 0.0

    if guven < SINIFLANDIRMA_ESIGI:
        return "K", guven

    if "polis" in sinif_adi:
        return "P", guven

    if "normal" in sinif_adi:
        return "N", guven

    return "K", guven


# ============================================================
# FOTOĞRAF KAYDI
# ============================================================

def fotograf_kaydet_anotasyonlu(
    goruntu,
    box,
    aid,
    guven,
    ocr_metni,
):
    try:
        kopya = goruntu.copy()
        yukseklik, genislik = kopya.shape[:2]

        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(genislik - 1, x1))
        x2 = max(0, min(genislik - 1, x2))
        y1 = max(0, min(yukseklik - 1, y1))
        y2 = max(0, min(yukseklik - 1, y2))

        cv2.rectangle(
            kopya,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        etiket = f"POLIS {_aid_kisa(aid)} %{guven * 100:.0f}"
        (tw, th), _ = cv2.getTextSize(
            etiket,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            2,
        )

        etiket_y1 = max(0, y1 - th - 15)
        etiket_x2 = min(genislik - 1, x1 + tw + 12)

        cv2.rectangle(
            kopya,
            (x1, etiket_y1),
            (etiket_x2, max(0, y1 - 3)),
            (0, 0, 200),
            -1,
        )

        cv2.putText(
            kopya,
            etiket,
            (x1 + 6, max(th + 8, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )

        if ocr_metni:
            cv2.putText(
                kopya,
                f"OCR: {str(ocr_metni)[:60]}",
                (x1, min(yukseklik - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            kopya,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (15, yukseklik - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        dosya_adi = (
            f"P_alarm_{aid}_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.jpg"
        )
        dosya = CIKTI_KLASORU / dosya_adi

        ok = cv2.imwrite(
            str(dosya),
            gece_iyilestir(kopya),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if ok:
            print(f"FOTOĞRAF: {dosya.name}")
            return dosya.name

        _logger.error(f"FOTOĞRAF YAZILAMADI aid={aid}")
        print("HATA: Fotoğraf yazılamadı; alarm devam ediyor.")
        return None

    except Exception as hata:
        _logger.error(f"FOTOĞRAF HATASI aid={aid}: {hata}")
        print(f"FOTOĞRAF HATASI (alarm devam ediyor): {hata}")
        return None


def fotograf_kaydet(goruntu, etiketler):
    try:
        if not etiketler:
            return None

        ana = etiketler[0].get("kod", "K")
        zaman = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        dosya = CIKTI_KLASORU / f"{ana}_{zaman}.jpg"

        ok = cv2.imwrite(
            str(dosya),
            gece_iyilestir(goruntu),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if ok:
            print(f"FOTOĞRAF: {dosya.name}")
            return dosya.name

        _logger.error(f"FOTOĞRAF YAZILAMADI: {dosya.name}")
        return None

    except Exception as hata:
        _logger.error(f"FOTOĞRAF HATASI: {hata}")
        return None


# ============================================================
# ALARM SESİ
# ============================================================

def polis_alarmi_cal():
    def _cal():
        try:
            for _ in range(ALARM_SESI_TEKRAR):
                winsound.Beep(1500, 250)
                winsound.Beep(1000, 250)
        except Exception as hata:
            _logger.warning(f"ALARM SESİ HATASI: {hata}")

    threading.Thread(target=_cal, daemon=True).start()


# ============================================================
# TRACK DURUMU VE ALARM YENİDEN HAZIRLAMA
# ============================================================

def _reset_p_streak(st):
    st["p_streak"] = 0
    st["first_p_time"] = -999.0
    st["gap_since_p"] = False
    st["k_count_since_p"] = 0


def yeni_track_state():
    return {
        "last_seen": 0.0,
        "last_box": None,
        "last_det_seen_idx": -1,
        "last_classify": -999.0,
        "last_ocr": -999.0,
        "class_code": "K",
        "class_conf": 0.0,
        "p_streak": 0,
        "first_p_time": -999.0,
        "last_p_time": -999.0,
        "gap_since_p": False,
        "k_count_since_p": 0,
        "ocr_polis_streak": 0,
        "ocr_miss_streak": 0,
        "last_ocr_polis_time": -999.0,
        "ocr_found": False,
        "ocr_metni": None,
        "ocr_metni_1": None,
        "ocr_celiski_t": 0.0,
        "ocr_celiski_metni": None,
        "alarm_fired": False,
        "alarm_conf": 0.0,
        "alarm_cooldown": 0.0,
        "alarm_photo_taken": False,
        "alarm_photo_time": 0.0,
        "last_periodic_photo": 0.0,
        "active_alarm_id": None,
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
    }


def _rearm_track(tid, track_state, simdi, expected_aid=None):
    """
    Yalnızca aynı alarma ait mevcut track'i sıfırlar.
    Eski/orphan alarmın aynı ID'yi alan yeni aracı etkilemesini önler.
    """
    if tid not in track_state:
        return False

    st = track_state[tid]

    if expected_aid is not None:
        if st.get("active_alarm_id") != expected_aid:
            return False

    st["alarm_fired"] = False
    st["alarm_cooldown"] = simdi + REARM_COOLDOWN
    st["active_alarm_id"] = None

    _reset_p_streak(st)

    st["ocr_polis_streak"] = 0
    st["ocr_miss_streak"] = 0
    st["ocr_found"] = False
    st["ocr_metni"] = None
    st["ocr_metni_1"] = None
    st["last_ocr_polis_time"] = -999.0
    st["ocr_celiski_t"] = 0.0
    st["ocr_celiski_metni"] = None

    _logger.info(
        f"REARM track={tid} cooldown={REARM_COOLDOWN}s"
    )
    return True


def alarmi_kapat_ve_gerekirse_rearm(
    aid,
    alarm_pending,
    track_state,
    simdi,
    sonuc,
):
    """
    Alarmı kuyruktan çıkarır.
    Orphan alarm, yeni track durumunu hiçbir zaman sıfırlamaz.
    """
    ap = alarm_pending.pop(aid, None)
    if ap is None:
        return None

    if not ap.get("orphaned", False):
        _rearm_track(
            ap["track_id"],
            track_state,
            simdi,
            expected_aid=aid,
        )

    olay_yaz(
        sonuc,
        aid=aid,
        track_id=ap["track_id"],
        orphaned=bool(ap.get("orphaned", False)),
        duration_s=round(
            max(0.0, simdi - ap["first_t"]),
            1,
        ),
    )

    return ap


# ============================================================
# EKRAN ÇİZİMİ
# ============================================================

def alarm_banner_ciz(
    goruntu,
    alarm_pending,
    simdi,
    secili_aid,
    batch_confirm_until,
    detection_stale,
):
    yukseklik, genislik = goruntu.shape[:2]
    ust = max(yukseklik - 195, 0)

    overlay = goruntu.copy()
    cv2.rectangle(
        overlay,
        (0, ust),
        (genislik, yukseklik),
        (0, 0, 200),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.55,
        goruntu,
        0.45,
        0,
        goruntu,
    )

    count = len(alarm_pending)

    if count == 1:
        baslik = "POLIS ARACI ALGILANDI - OPERATOR ONAYI GEREKLI"
    else:
        baslik = (
            f"{count} POLIS ARACI ALGILANDI - "
            "OPERATOR ONAYI GEREKLI"
        )

    cv2.putText(
        goruntu,
        baslik,
        (30, ust + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
    )

    parcalar = []

    for aid, p in alarm_pending.items():
        kalan = max(
            0.0,
            ALARM_OTOMATIK_KAPATMA
            - (simdi - p["last_seen_t"]),
        )

        kalan_etiket = "DURDU" if detection_stale else f"{kalan:.0f}s"
        etiket = (
            f"{_aid_kisa(aid)}(kayip)"
            if p["orphaned"]
            else _aid_kisa(aid)
        )

        if aid == secili_aid:
            etiket = "*" + etiket

        parcalar.append(f"{etiket}:{kalan_etiket}")

    cv2.putText(
        goruntu,
        "  ".join(parcalar)[:180],
        (30, ust + 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        goruntu,
        "Sol/Sag veya A/D: sec | SPACE: onay | X: red | B: toplu onay",
        (30, ust + 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    if secili_aid in alarm_pending:
        p = alarm_pending[secili_aid]
        durum = "KAYIP (onay yok)" if p["orphaned"] else "aktif"
        cv2.putText(
            goruntu,
            (
                f"Secili: {_aid_kisa(secili_aid)} "
                f"guven=%{p['conf'] * 100:.0f} durum={durum}"
            ),
            (30, ust + 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    if detection_stale:
        cv2.putText(
            goruntu,
            "DETECTION STALE: alarm timeout'lari DURDURULDU",
            (30, ust + 156),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
        )

    orphan_count = sum(
        1 for p in alarm_pending.values()
        if p["orphaned"]
    )

    if orphan_count:
        cv2.putText(
            goruntu,
            f"UYARI: {orphan_count} kayip alarm - sadece X ile reddedilir",
            (30, ust + 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            1,
        )


def secili_kutuyu_ciz(goruntu, alarm_pending, secili_aid):
    if secili_aid not in alarm_pending:
        return

    p = alarm_pending[secili_aid]
    if p["orphaned"]:
        return

    box = p.get("last_box")
    if box is None or len(box) != 4:
        return

    x1, y1, x2, y2 = [int(v) for v in box]
    yukseklik, genislik = goruntu.shape[:2]

    x1 = max(0, min(genislik - 1, x1))
    x2 = max(0, min(genislik - 1, x2))
    y1 = max(0, min(yukseklik - 1, y1))
    y2 = max(0, min(yukseklik - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(
        goruntu,
        (max(0, x1 - 4), max(0, y1 - 4)),
        (min(genislik - 1, x2 + 4),
         min(yukseklik - 1, y2 + 4)),
        (0, 255, 255),
        3,
    )

    cv2.putText(
        goruntu,
        "SECILI",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def pending_sayaclarini_ciz(
    goruntu,
    alarm_pending,
    simdi,
    gx,
    gy,
    secili_aid,
    detection_stale,
):
    for aid, p in alarm_pending.items():
        if p["orphaned"]:
            continue

        box = p.get("last_box")
        if box is None or len(box) != 4:
            continue

        kalan = max(
            0.0,
            ALARM_OTOMATIK_KAPATMA
            - (simdi - p["last_seen_t"]),
        )

        bx1, by1 = int(box[0]), int(box[1])
        bx1 = max(5, min(max(5, gx - 130), bx1))
        by1 = max(30, min(max(30, gy - 5), by1))

        etiket = (
            f"{_aid_kisa(aid)}: DURDU"
            if detection_stale
            else f"{_aid_kisa(aid)}: {kalan:.0f}s"
        )

        if aid == secili_aid:
            etiket = "> " + etiket

        (tw, th), _ = cv2.getTextSize(
            etiket,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )

        arka = (
            (0, 200, 200)
            if aid == secili_aid
            else (0, 0, 180)
        )

        y_ust = max(0, by1 - th - 10)
        x_sag = min(gx - 1, bx1 + tw + 10)

        cv2.rectangle(
            goruntu,
            (bx1, y_ust),
            (x_sag, max(0, by1 - 2)),
            arka,
            -1,
        )

        cv2.putText(
            goruntu,
            etiket,
            (bx1 + 5, max(th + 5, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )


# ============================================================
# ANA PROGRAM
# ============================================================

def main():
    global kamera

    algilama_modeli = None
    siniflandirma_modeli = None
    ocr = None

    track_state = {}
    alarm_pending = {}
    secili_aid = None

    batch_confirm_until = 0.0
    batch_bekleyen = []

    son_etiketler = []
    son_algilama = 0.0
    son_cleanup = 0.0

    kare_sayisi = 0
    baslangic = time.monotonic()
    det_index = 0

    detect_time_total = 0.0
    detect_count = 0
    classify_time_total = 0.0
    classify_count = 0
    ocr_time_total = 0.0
    ocr_count = 0

    son_basarili_algilama_t = baslangic
    stale_baslangic = None
    detection_stale = False

    kamera_reopen_sayisi = 0
    normal_kapanis = False

    try:
        print("Algilama modeli yukleniyor...")
        algilama_modeli = YOLO(ALGILAMA_MODELI)

        if not Path(SINIFLANDIRMA_MODELI).is_file():
            raise FileNotFoundError(
                "Siniflandirma modeli bulunamadi:\n"
                f"{SINIFLANDIRMA_MODELI}"
            )

        print("Polis siniflandirma modeli yukleniyor...")
        siniflandirma_modeli = YOLO(SINIFLANDIRMA_MODELI)

        print("OCR modeli yukleniyor...")
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

        print("PlugCam kamerasi aciliyor...")
        if not kamera_ac():
            raise RuntimeError(
                f"Kamera acilamadi: {_kamera_hata['mesaj']}"
            )

        ok, test_kare = kamera.read()
        if not ok or test_kare is None:
            raise RuntimeError("Kameradan goruntu alinamadi.")

        gy, gx = test_kare.shape[:2]
        print(f"Gercek kamera cozunurlugu: {gx}x{gy}")

        _logger.info("SISTEM BASLATILDI")
        olay_yaz("startup", camera=KAMERA_NO)

        while True:
            # ----------------------------------------------------
            # KAMERA OKUMA VE KURTARMA
            # ----------------------------------------------------
            ok, kare = kamera.read() if kamera is not None else (False, None)

            if not ok or kare is None:
                _kamera_hata["okuma_fail"] += 1
                _kamera_hata["durum"] = "HATA"

                if _kamera_hata["okuma_fail"] < CAMERA_MAX_FAIL:
                    time.sleep(0.05)
                    continue

                simdi_kk = time.monotonic()
                kalan_cooldown = (
                    CAMERA_REOPEN_COOLDOWN
                    - (simdi_kk - _kamera_hata["son_reopen_t"])
                )

                if kalan_cooldown > 0:
                    time.sleep(min(0.25, kalan_cooldown))
                    continue

                if kamera_reopen_sayisi >= CAMERA_REOPEN_MAX_ATTEMPTS:
                    _kamera_hata["durum"] = "KAPALI"
                    _kamera_hata["mesaj"] = (
                        f"{kamera_reopen_sayisi} denemede acilamadi. "
                        "Kamera baglantisini kontrol et."
                    )
                    _logger.critical(_kamera_hata["mesaj"])
                    olay_yaz(
                        "camera_fatal",
                        attempts=kamera_reopen_sayisi,
                        message=_kamera_hata["mesaj"],
                    )

                    hata_ekrani = cv2.zeros(
                        (720, 1280, 3),
                        dtype="uint8",
                    )
                    cv2.putText(
                        hata_ekrani,
                        "KAMERA ACILAMADI",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                    )
                    cv2.putText(
                        hata_ekrani,
                        "Q: cikis",
                        (50, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("PlugCam - KAMERA HATASI", hata_ekrani)
                    tus = cv2.waitKey(500) & 0xFF

                    if tus in (ord("q"), ord("Q")):
                        normal_kapanis = True
                        break

                    # Görünmez biçimde sonsuza dek devam etme.
                    print(_kamera_hata["mesaj"])
                    break

                kamera_reopen_sayisi += 1
                _kamera_hata["reopen_attempts"] = kamera_reopen_sayisi
                _kamera_hata["son_reopen_t"] = simdi_kk

                _logger.warning(
                    f"KAMERA YENIDEN ACMA DENEMESI "
                    f"{kamera_reopen_sayisi}/"
                    f"{CAMERA_REOPEN_MAX_ATTEMPTS}"
                )

                if kamera_ac():
                    _kamera_hata["okuma_fail"] = 0
                    _kamera_hata["durum"] = "OK"
                    _logger.info("KAMERA YENIDEN ACILDI")
                    # Sayaç burada sıfırlanmıyor; gerçek bir kare
                    # başarıyla okunduğunda sıfırlanacak.
                else:
                    _kamera_hata["durum"] = "HATA"
                    _logger.error(
                        "KAMERA YENIDEN ACMA BASARISIZ: "
                        f"{_kamera_hata['mesaj']}"
                    )

                time.sleep(0.1)
                continue

            # Gerçek kare alındı: kamera kurtarma sayacı sıfırlanır.
            _kamera_hata["okuma_fail"] = 0
            _kamera_hata["durum"] = "OK"
            kamera_reopen_sayisi = 0
            _kamera_hata["reopen_attempts"] = 0

            yh, gw = kare.shape[:2]
            simdi = time.monotonic()

            # ----------------------------------------------------
            # ALGILAMA
            # ----------------------------------------------------
            if simdi - son_algilama >= ALGILAMA_ARALIGI:
                det_index += 1

                kh = max(
                    1,
                    int(yh * GORUNTU_GENISLIGI / max(1, gw)),
                )
                kk = cv2.resize(
                    kare,
                    (GORUNTU_GENISLIGI, kh),
                )

                t_bas = time.monotonic()
                sonuclar = None
                algilama_hata = False

                try:
                    sonuclar = algilama_modeli.track(
                        source=kk,
                        classes=[0, 2, 3, 5, 7],
                        conf=ALGILAMA_GUVEN_ESIGI,
                        imgsz=640,
                        device="cpu",
                        verbose=False,
                        persist=True,
                        tracker="bytetrack.yaml",
                    )
                except Exception as hata:
                    algilama_hata = True
                    _logger.exception(f"ALGILAMA HATASI: {hata}")
                    print(f"Algilama hatasi: {hata}")

                detect_time_total += time.monotonic() - t_bas
                detect_count += 1

                if not algilama_hata:
                    son_basarili_algilama_t = time.monotonic()

                # Algılama sistemi yeniden çalışmaya başladığında,
                # stale süresini pending alarm zaman aşımından düş.
                yeni_stale = (
                    time.monotonic() - son_basarili_algilama_t
                    > DETECTION_STALE_THRESHOLD
                )

                if detection_stale and not algilama_hata:
                    if stale_baslangic is not None:
                        duraklama = max(
                            0.0,
                            time.monotonic() - stale_baslangic,
                        )
                        for ap in alarm_pending.values():
                            ap["last_seen_t"] += duraklama

                        _logger.info(
                            f"DETECTION RECOVERY; timeout "
                            f"{duraklama:.2f}s uzatildi"
                        )

                    stale_baslangic = None
                    detection_stale = False

                elif algilama_hata and not detection_stale:
                    detection_stale = True
                    stale_baslangic = time.monotonic()
                    _logger.warning("DETECTION STALE BASLADI")

                # Sonuç listesini sadece geçerli algılamada güncelle.
                if not algilama_hata:
                    yeni_etiketler = []
                    sx = gw / GORUNTU_GENISLIGI
                    sy = yh / kh

                    sonuc_boxes = None
                    if sonuclar and sonuclar[0].boxes is not None:
                        sonuc_boxes = sonuclar[0].boxes

                    if sonuc_boxes is not None:
                        for kutu in sonuc_boxes:
                            try:
                                sinif = int(kutu.cls[0].item())
                                kutu_guveni = float(kutu.conf[0].item())

                                xx1, yy1, xx2, yy2 = (
                                    kutu.xyxy[0].tolist()
                                )

                                x1 = max(
                                    0, min(gw - 1, int(xx1 * sx))
                                )
                                x2 = max(
                                    0, min(gw, int(xx2 * sx))
                                )
                                y1 = max(
                                    0, min(yh - 1, int(yy1 * sy))
                                )
                                y2 = max(
                                    0, min(yh, int(yy2 * sy))
                                )

                                takip_id = (
                                    int(kutu.id[0].item())
                                    if kutu.id is not None
                                    else None
                                )
                            except Exception as hata:
                                _logger.warning(
                                    f"KUTU OKUMA HATASI: {hata}"
                                )
                                continue

                            if x2 <= x1 or y2 <= y1:
                                continue

                            if sinif == 0:
                                yeni_etiketler.append({
                                    "kod": "İ",
                                    "aciklama": "İNSAN",
                                    "guven": kutu_guveni,
                                    "kutu_guveni": kutu_guveni,
                                    "sinif": sinif,
                                    "takip_id": takip_id,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "ocr_metni": None,
                                })
                                continue

                            if sinif not in ARACLAR:
                                continue

                            if sinif != 2:
                                yeni_etiketler.append({
                                    "kod": "K",
                                    "aciklama": ARACLAR[sinif] + " - KONTROL",
                                    "guven": kutu_guveni,
                                    "kutu_guveni": kutu_guveni,
                                    "sinif": sinif,
                                    "takip_id": takip_id,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "ocr_metni": None,
                                })
                                continue

                            if takip_id is None:
                                yeni_etiketler.append({
                                    "kod": "K",
                                    "aciklama": "KONTROL (takip yok)",
                                    "guven": kutu_guveni,
                                    "kutu_guveni": kutu_guveni,
                                    "sinif": sinif,
                                    "takip_id": None,
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "ocr_metni": None,
                                })
                                continue

                            st = track_state.setdefault(
                                takip_id,
                                yeni_track_state(),
                            )

                            # ------------------------------------
                            # TRACK REUSE KONTROLÜ
                            # ------------------------------------
                            if st["last_box"] is not None:
                                gap = simdi - st["last_seen"]
                                yeni_box = (x1, y1, x2, y2)

                                iou = _iou(st["last_box"], yeni_box)
                                alan_eski = _alan(st["last_box"])
                                alan_yeni = _alan(yeni_box)

                                size_ratio = (
                                    alan_yeni / alan_eski
                                    if alan_eski > 0
                                    else 1.0
                                )

                                reason = None

                                if (
                                    gap >= TRACK_REUSE_MIN_GAP
                                    and iou < TRACK_REUSE_MAX_IOU
                                ):
                                    reason = "uzun_gap+dusuk_iou"

                                elif (
                                    gap >= TRACK_REUSE_SHORT_GAP
                                    and iou < TRACK_REUSE_SHORT_IOU
                                ):
                                    reason = "kisa_gap+sifir_iou"

                                elif (
                                    gap > 0.05
                                    and (
                                        size_ratio < TRACK_REUSE_SIZE_MIN
                                        or size_ratio > TRACK_REUSE_SIZE_MAX
                                    )
                                ):
                                    reason = f"boyut({size_ratio:.2f})"

                                elif gap > 0.05:
                                    pcx, pcy = _merkez(st["last_box"])
                                    ccx, ccy = _merkez(yeni_box)
                                    uzaklik = (
                                        (ccx - pcx) ** 2
                                        + (ccy - pcy) ** 2
                                    ) ** 0.5

                                    if uzaklik / gap > TRACK_REUSE_MAX_SPEED:
                                        reason = (
                                            f"hiz({uzaklik / gap:.0f}px/s)"
                                        )

                                if reason:
                                    _logger.info(
                                        f"TRACK REUSE id={takip_id} "
                                        f"reason={reason}"
                                    )

                                    for old_aid, ap in list(
                                        alarm_pending.items()
                                    ):
                                        if (
                                            ap["track_id"] == takip_id
                                            and not ap["orphaned"]
                                        ):
                                            ap["orphaned"] = True
                                            ap["orphaned_reason"] = "reuse"

                                            _logger.info(
                                                f"ALARM ORPHANED "
                                                f"aid={old_aid} "
                                                f"track={takip_id}"
                                            )
                                            olay_yaz(
                                                "orphaned",
                                                aid=old_aid,
                                                track_id=takip_id,
                                                reason="reuse",
                                            )

                                    st = yeni_track_state()
                                    track_state[takip_id] = st

                            if (
                                st["last_det_seen_idx"] >= 0
                                and det_index - st["last_det_seen_idx"] > 1
                            ):
                                st["gap_since_p"] = True

                            st["last_det_seen_idx"] = det_index
                            st["last_seen"] = simdi
                            st["last_box"] = (x1, y1, x2, y2)
                            st["x1"], st["y1"] = x1, y1
                            st["x2"], st["y2"] = x2, y2

                            # ------------------------------------
                            # SINIFLANDIRMA
                            # ------------------------------------
                            if (
                                simdi - st["last_classify"]
                                >= CLASSIFY_INTERVAL
                            ):
                                st["last_classify"] = simdi
                                t_bas = time.monotonic()

                                kod, guven = siniflandir_arac(
                                    siniflandirma_modeli,
                                    kare,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                )

                                classify_time_total += (
                                    time.monotonic() - t_bas
                                )
                                classify_count += 1

                                st["class_code"] = kod
                                st["class_conf"] = guven

                                if kod == "P":
                                    pencere_doldu = (
                                        st["first_p_time"] < 0
                                        or simdi - st["first_p_time"]
                                        > P_STREAK_MAX_WINDOW
                                    )

                                    yeni_seri = (
                                        st["gap_since_p"]
                                        or pencere_doldu
                                        or simdi - st["last_p_time"]
                                        > P_STREAK_HARD_TIME_CAP
                                        or st["k_count_since_p"]
                                        > P_STREAK_K_TOLERANCE
                                    )

                                    if yeni_seri:
                                        st["p_streak"] = 1
                                        st["first_p_time"] = simdi
                                    else:
                                        st["p_streak"] += 1

                                    st["last_p_time"] = simdi
                                    st["gap_since_p"] = False
                                    st["k_count_since_p"] = 0

                                elif kod == "N":
                                    _reset_p_streak(st)
                                    st["ocr_polis_streak"] = 0
                                    st["ocr_miss_streak"] = 0
                                    st["ocr_metni_1"] = None

                                elif kod == "K":
                                    st["k_count_since_p"] += 1
                                    if (
                                        st["k_count_since_p"]
                                        > P_STREAK_K_TOLERANCE
                                    ):
                                        _reset_p_streak(st)

                            kod = st["class_code"]
                            nesne_guveni = st["class_conf"]

                            # ------------------------------------
                            # OCR DOĞRULAMA
                            # ------------------------------------
                            if (
                                st["ocr_polis_streak"] > 0
                                and simdi - st["last_ocr_polis_time"]
                                > OCR_CONFIRM_WINDOW
                            ):
                                st["ocr_polis_streak"] = 0
                                st["ocr_miss_streak"] = 0
                                st["ocr_metni_1"] = None
                                _logger.info(
                                    f"OCR WINDOW EXPIRED track={takip_id}"
                                )

                            if (
                                not st["ocr_found"]
                                and simdi - st["last_ocr"] >= OCR_ARALIGI
                            ):
                                st["last_ocr"] = simdi
                                kirpim = kare[y1:y2, x1:x2]
                                t_bas = time.monotonic()

                                durum, ocr_metni = polis_yazisi_var_mi(
                                    ocr,
                                    kirpim,
                                )

                                ocr_time_total += (
                                    time.monotonic() - t_bas
                                )
                                ocr_count += 1

                                if durum == "P":
                                    if st["class_code"] == "N":
                                        st["ocr_polis_streak"] = 0
                                        st["ocr_miss_streak"] = 0
                                        st["ocr_metni_1"] = None
                                        st["ocr_celiski_t"] = simdi
                                        st["ocr_celiski_metni"] = ocr_metni

                                        _logger.info(
                                            f"OCR/CLS CELISKI "
                                            f"cls=N ocr=P track={takip_id}"
                                        )
                                    else:
                                        if st["ocr_metni_1"] is None:
                                            st["ocr_polis_streak"] = 1
                                            st["ocr_metni_1"] = ocr_metni
                                            st["last_ocr_polis_time"] = simdi
                                            st["ocr_metni"] = ocr_metni
                                            st["ocr_miss_streak"] = 0
                                        else:
                                            benzerlik = metin_benzerlik(
                                                st["ocr_metni_1"],
                                                ocr_metni,
                                            )

                                            if (
                                                benzerlik
                                                < OCR_TEXT_MATCH_RATIO
                                            ):
                                                _logger.info(
                                                    f"OCR METIN UYUMSUZ "
                                                    f"track={takip_id} "
                                                    f"sim={benzerlik:.2f}"
                                                )

                                                st["ocr_polis_streak"] = 1
                                                st["ocr_metni_1"] = ocr_metni
                                                st["ocr_metni"] = ocr_metni
                                                st["last_ocr_polis_time"] = simdi
                                                st["ocr_miss_streak"] = 0
                                            else:
                                                st["ocr_polis_streak"] += 1
                                                st["ocr_metni"] = ocr_metni
                                                st["last_ocr_polis_time"] = simdi
                                                st["ocr_miss_streak"] = 0

                                        if (
                                            st["ocr_polis_streak"]
                                            >= OCR_CONFIRM_COUNT
                                        ):
                                            st["ocr_found"] = True
                                            _logger.info(
                                                f"OCR DOGRULANDI "
                                                f"track={takip_id} "
                                                f"streak={st['ocr_polis_streak']} "
                                                f"metin='{st['ocr_metni']}'"
                                            )

                                elif durum == "N":
                                    st["ocr_miss_streak"] += 1

                                    if (
                                        st["ocr_miss_streak"]
                                        > OCR_MISS_TOLERANCE
                                    ):
                                        st["ocr_polis_streak"] = 0
                                        st["ocr_metni_1"] = None
                                        st["ocr_metni"] = None

                                else:
                                    _logger.info(
                                        f"OCR ERROR track={takip_id}; "
                                        "mevcut seri korundu"
                                    )

                            # ------------------------------------
                            # SON SINIF VE ALARM KOŞULU
                            # ------------------------------------
                            if st["ocr_found"]:
                                kod = "P"
                                nesne_guveni = max(nesne_guveni, 0.90)
                                aciklama = "POLIS (YAZI TANINDI)"
                            elif kod == "P":
                                aciklama = "POLIS"
                            elif kod == "N":
                                aciklama = "NORMAL"
                            else:
                                aciklama = "KONTROL"

                            onaylandi = (
                                st["ocr_found"]
                                or st["p_streak"] >= P_STREAK_REQUIRED
                            )

                            if (
                                onaylandi
                                and not st["alarm_fired"]
                                and simdi >= st["alarm_cooldown"]
                            ):
                                st["alarm_fired"] = True
                                st["alarm_conf"] = nesne_guveni

                                aid = yeni_alarm_id()
                                polis_alarmi_cal()

                                alarm_foto = fotograf_kaydet_anotasyonlu(
                                    kare,
                                    (x1, y1, x2, y2),
                                    aid,
                                    nesne_guveni,
                                    st["ocr_metni"],
                                )

                                st["alarm_photo_taken"] = True
                                st["alarm_photo_time"] = simdi

                                alarm_pending[aid] = {
                                    "track_id": takip_id,
                                    "first_t": simdi,
                                    "last_seen_t": simdi,
                                    "conf": nesne_guveni,
                                    "last_box": (x1, y1, x2, y2),
                                    "orphaned": False,
                                    "orphaned_reason": None,
                                }

                                st["active_alarm_id"] = aid

                                if secili_aid is None:
                                    secili_aid = aid

                                # Önce JSONL yazımını tamamla; başarı
                                # bilgisini henüz oluşmamış değişkenden
                                # okumaya çalışma.
                                jsonl_ok = olay_yaz(
                                    "alarm",
                                    aid=aid,
                                    track_id=takip_id,
                                    conf=round(nesne_guveni, 3),
                                    p_streak=st["p_streak"],
                                    ocr_streak=st["ocr_polis_streak"],
                                    ocr_metni=st["ocr_metni"],
                                    foto=alarm_foto,
                                    box=[x1, y1, x2, y2],
                                )

                                _logger.info(
                                    f"ALARM aid={aid} track={takip_id} "
                                    f"guven=%{nesne_guveni * 100:.0f} "
                                    f"p_streak={st['p_streak']} "
                                    f"ocr_streak={st['ocr_polis_streak']} "
                                    f"jsonl_ok={jsonl_ok}"
                                )

                                print(
                                    f"\n*** ALARM {_aid_kisa(aid)} "
                                    f"(track={takip_id}, "
                                    f"guven=%{nesne_guveni * 100:.0f}) ***\n"
                                )

                            # Aktif alarmın son görüldüğü konum/zaman.
                            aid = st.get("active_alarm_id")
                            if (
                                aid is not None
                                and aid in alarm_pending
                                and not alarm_pending[aid]["orphaned"]
                            ):
                                alarm_pending[aid]["last_seen_t"] = simdi
                                alarm_pending[aid]["last_box"] = (
                                    x1, y1, x2, y2
                                )

                                if nesne_guveni > alarm_pending[aid]["conf"]:
                                    alarm_pending[aid]["conf"] = nesne_guveni

                            yeni_etiketler.append({
                                "kod": kod,
                                "aciklama": aciklama,
                                "guven": nesne_guveni,
                                "kutu_guveni": kutu_guveni,
                                "sinif": sinif,
                                "takip_id": takip_id,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "ocr_metni": st["ocr_metni"],
                            })

                    son_etiketler = yeni_etiketler
                    son_algilama = time.monotonic()

                    # --------------------------------------------
                    # ESKİ TRACK TEMİZLİĞİ
                    # --------------------------------------------
                    if (
                        time.monotonic() - son_cleanup
                        >= TRACK_CLEANUP_ARALIGI
                    ):
                        son_cleanup = time.monotonic()

                        bayat = [
                            tid
                            for tid, state in track_state.items()
                            if (
                                state["last_seen"] > 0
                                and simdi - state["last_seen"]
                                > TRACK_TIMEOUT
                            )
                        ]

                        for tid in bayat:
                            eski_state = track_state[tid]
                            eski_aid = eski_state.get("active_alarm_id")

                            if (
                                eski_aid is not None
                                and eski_aid in alarm_pending
                                and not alarm_pending[eski_aid]["orphaned"]
                            ):
                                alarm_pending[eski_aid]["orphaned"] = True
                                alarm_pending[eski_aid]["orphaned_reason"] = (
                                    "timeout"
                                )

                                _logger.info(
                                    f"ALARM ORPHANED aid={eski_aid} "
                                    f"track={tid} reason=timeout"
                                )
                                olay_yaz(
                                    "orphaned",
                                    aid=eski_aid,
                                    track_id=tid,
                                    reason="timeout",
                                )

                            del track_state[tid]

                # Başarılı çıkarım yoksa eski etiketler bir sonraki
                # başarılı çıkarıma kadar korunur; timeout duraklatılır.

            # ----------------------------------------------------
            # ALGILAMA STALE DURUMU VE ALARM TIMEOUT
            # ----------------------------------------------------
            simdi = time.monotonic()

            if (
                not detection_stale
                and simdi - son_basarili_algilama_t
                > DETECTION_STALE_THRESHOLD
            ):
                detection_stale = True
                stale_baslangic = simdi
                _logger.warning("DETECTION STALE BASLADI")

            if not detection_stale:
                suresi_dolanlar = [
                    aid
                    for aid, ap in alarm_pending.items()
                    if (
                        simdi - ap["last_seen_t"]
                        > ALARM_OTOMATIK_KAPATMA
                    )
                ]

                for aid in suresi_dolanlar:
                    ap = alarm_pending.get(aid)
                    if ap is None:
                        continue

                    orphaned = bool(ap["orphaned"])

                    _logger.info(
                        f"ALARM TIMEOUT aid={aid} "
                        f"track={ap['track_id']} "
                        f"orphaned={orphaned} "
                        f"duration={simdi - ap['first_t']:.1f}s"
                    )

                    # Aktif alarm kapanınca ilgili track yeniden
                    # hazırlanır; orphan alarm yeni track'i etkileyemez.
                    alarmi_kapat_ve_gerekirse_rearm(
                        aid,
                        alarm_pending,
                        track_state,
                        simdi,
                        "timeout_orphaned" if orphaned else "timeout",
                    )

            # Seçili alarm artık yoksa yeni bir seçim yap.
            if secili_aid not in alarm_pending:
                secili_aid = (
                    next(iter(alarm_pending))
                    if alarm_pending
                    else None
                )

            # Toplu onay süresi dolduysa isteği iptal et.
            if batch_bekleyen and simdi > batch_confirm_until:
                _logger.info(
                    f"BATCH ONAY ZAMAN ASIMI: {batch_bekleyen}"
                )
                print("Toplu onay iptal edildi (zaman aşımı).")
                batch_bekleyen = []
                batch_confirm_until = 0.0

            # ----------------------------------------------------
            # GÖRÜNTÜ ÇİZİMİ
            # ----------------------------------------------------
            goruntu = kare.copy()
            insan_sayisi = 0
            arac_sayisi = 0
            polis_sayisi = 0

            for etiket in son_etiketler:
                kod = etiket["kod"]
                guven = etiket["guven"]

                x1 = etiket["x1"]
                y1 = etiket["y1"]
                x2 = etiket["x2"]
                y2 = etiket["y2"]

                if kod == "İ":
                    renk = (0, 255, 0)
                    insan_sayisi += 1
                elif kod == "P":
                    renk = (0, 0, 255)
                    polis_sayisi += 1
                    arac_sayisi += 1
                else:
                    renk = (0, 165, 255)
                    if etiket.get("sinif") in ARACLAR:
                        arac_sayisi += 1

                cv2.rectangle(
                    goruntu,
                    (x1, y1),
                    (x2, y2),
                    renk,
                    2,
                )
                cv2.putText(
                    goruntu,
                    f"{kod} %{guven * 100:.0f}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    renk,
                    2,
                )

            # Periyodik fotoğraf kaydı.
            kaydedilecekler = []

            for etiket in son_etiketler:
                tid = etiket.get("takip_id")
                if tid is None or tid not in track_state:
                    continue

                state = track_state[tid]

                if (
                    state["alarm_photo_taken"]
                    and simdi - state["alarm_photo_time"]
                    < TEKRAR_HAZIRLANMA
                ):
                    continue

                if (
                    simdi - state["last_periodic_photo"]
                    >= TEKRAR_HAZIRLANMA
                ):
                    state["last_periodic_photo"] = simdi
                    kaydedilecekler.append(etiket)

            if kaydedilecekler:
                fotograf_kaydet(goruntu, kaydedilecekler)

            if alarm_pending:
                secili_kutuyu_ciz(
                    goruntu,
                    alarm_pending,
                    secili_aid,
                )
                pending_sayaclarini_ciz(
                    goruntu,
                    alarm_pending,
                    simdi,
                    gw,
                    yh,
                    secili_aid,
                    detection_stale,
                )
                alarm_banner_ciz(
                    goruntu,
                    alarm_pending,
                    simdi,
                    secili_aid,
                    batch_confirm_until,
                    detection_stale,
                )

            # ----------------------------------------------------
            # HUD
            # ----------------------------------------------------
            gecen = max(time.monotonic() - baslangic, 0.001)
            fps = kare_sayisi / gecen
            kare_sayisi += 1

            cv2.putText(
                goruntu,
                (
                    f"INSAN:{insan_sayisi} ARAC:{arac_sayisi} "
                    f"POLIS:{polis_sayisi} TRACK:{len(track_state)} "
                    f"PEND:{len(alarm_pending)}"
                ),
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            ort_det = (
                detect_time_total / detect_count * 1000
                if detect_count else 0
            )
            ort_cls = (
                classify_time_total / classify_count * 1000
                if classify_count else 0
            )
            ort_ocr = (
                ocr_time_total / ocr_count * 1000
                if ocr_count else 0
            )

            cv2.putText(
                goruntu,
                (
                    f"FPS:{fps:.1f} | "
                    f"Det:{ort_det:.0f}ms({detect_count}) | "
                    f"Cls:{ort_cls:.0f}ms({classify_count}) | "
                    f"OCR:{ort_ocr:.0f}ms({ocr_count})"
                ),
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

            kamera_renk = (
                (0, 255, 0)
                if _kamera_hata["durum"] == "OK"
                else (0, 0, 255)
            )

            cv2.putText(
                goruntu,
                f"KAMERA: {_kamera_hata['durum']}",
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                kamera_renk,
                1,
            )

            if detection_stale:
                cv2.putText(
                    goruntu,
                    "DETECTION STALE - timeout durdu",
                    (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

            if _jsonl_hata["consecutive"] > 0:
                cv2.putText(
                    goruntu,
                    (
                        f"!!! JSONL HATA: ard arda "
                        f"{_jsonl_hata['consecutive']} "
                        f"(toplam {_jsonl_hata['count']}) !!!"
                    ),
                    (15, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

            if batch_bekleyen:
                kalan_onay = max(
                    0,
                    batch_confirm_until - simdi,
                )
                cv2.putText(
                    goruntu,
                    (
                        f"TOPLU ONAY: {len(batch_bekleyen)} alarm "
                        f"onaylanacak - SPACE onay / X iptal "
                        f"({kalan_onay:.0f}s)"
                    ),
                    (15, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2,
                )

            pencere_adi = (
                "PlugCam - N Normal | P Polis | K Kontrol | "
                "I Insan | Ok/AD Sec | SPACE Onay | X Red | B Toplu"
            )
            cv2.imshow(pencere_adi, goruntu)

            try:
                tus = cv2.waitKeyEx(1)
            except Exception:
                tus = cv2.waitKey(1) & 0xFF

            tus_low = tus & 0xFF

            # ----------------------------------------------------
            # TOPLU ONAY MODU
            # ----------------------------------------------------
            if batch_bekleyen:
                if tus_low == ord(" "):
                    onaylanan = []

                    for aid in list(batch_bekleyen):
                        ap = alarm_pending.get(aid)
                        if ap is None:
                            continue

                        if ap["orphaned"]:
                            continue

                        alarmi_kapat_ve_gerekirse_rearm(
                            aid,
                            alarm_pending,
                            track_state,
                            simdi,
                            "approved",
                        )
                        onaylanan.append(aid)

                    _logger.info(
                        f"BATCH ONAYLANDI aids={onaylanan}"
                    )
                    print(f"Toplu onay: {len(onaylanan)} alarm")

                    batch_bekleyen = []
                    batch_confirm_until = 0.0
                    secili_aid = (
                        next(iter(alarm_pending))
                        if alarm_pending else None
                    )

                elif tus_low in (ord("x"), ord("X")):
                    _logger.info(
                        f"BATCH ONAY IPTAL: {batch_bekleyen}"
                    )
                    print("Toplu onay iptal edildi.")
                    batch_bekleyen = []
                    batch_confirm_until = 0.0

                elif tus_low in (ord("q"), ord("Q")):
                    normal_kapanis = True
                    break

                continue

            # ----------------------------------------------------
            # NORMAL TUŞ YÖNETİMİ
            # ----------------------------------------------------
            if (
                tus in (2424832, 81)
                or tus_low in (ord("a"), ord("A"))
            ):
                if alarm_pending:
                    ids = list(alarm_pending.keys())
                    cur = (
                        ids.index(secili_aid)
                        if secili_aid in ids
                        else 0
                    )
                    secili_aid = ids[(cur - 1) % len(ids)]

            elif (
                tus in (2555904, 83)
                or tus_low in (ord("d"), ord("D"))
            ):
                if alarm_pending:
                    ids = list(alarm_pending.keys())
                    cur = (
                        ids.index(secili_aid)
                        if secili_aid in ids
                        else -1
                    )
                    secili_aid = ids[(cur + 1) % len(ids)]

            elif tus_low in (ord("q"), ord("Q")):
                normal_kapanis = True
                break

            elif tus_low == ord(" "):
                if secili_aid and secili_aid in alarm_pending:
                    ap = alarm_pending[secili_aid]

                    if ap["orphaned"]:
                        print(
                            f"UYARI: {_aid_kisa(secili_aid)} KAYIP. "
                            "Onaylanamaz; X ile reddet."
                        )
                        _logger.info(
                            f"APPROVE REDDEDILDI orphaned aid={secili_aid}"
                        )
                    else:
                        aid = secili_aid
                        alarmi_kapat_ve_gerekirse_rearm(
                            aid,
                            alarm_pending,
                            track_state,
                            simdi,
                            "approved",
                        )

                        _logger.info(
                            f"OPERATOR ONAYLADI aid={aid} "
                            f"track={ap['track_id']}"
                        )
                        print(f"Onaylandi: {_aid_kisa(aid)}")

                        secili_aid = (
                            next(iter(alarm_pending))
                            if alarm_pending else None
                        )

            elif tus_low in (ord("x"), ord("X")):
                if secili_aid and secili_aid in alarm_pending:
                    aid = secili_aid
                    ap = alarmi_kapat_ve_gerekirse_rearm(
                        aid,
                        alarm_pending,
                        track_state,
                        simdi,
                        "rejected",
                    )

                    if ap is not None:
                        _logger.info(
                            f"OPERATOR REDDETTI aid={aid} "
                            f"track={ap['track_id']} "
                            f"orphaned={ap['orphaned']}"
                        )
                        print(f"Reddedildi: {_aid_kisa(aid)}")

                    secili_aid = (
                        next(iter(alarm_pending))
                        if alarm_pending else None
                    )

            elif tus_low in (ord("b"), ord("B")):
                aktif_onaylanabilir = [
                    aid
                    for aid, ap in alarm_pending.items()
                    if not ap["orphaned"]
                ]

                if aktif_onaylanabilir:
                    batch_bekleyen = aktif_onaylanabilir
                    batch_confirm_until = (
                        simdi + BATCH_APPROVE_CONFIRM_WINDOW
                    )

                    _logger.info(
                        f"BATCH ONAY BEKLIYOR: "
                        f"{len(batch_bekleyen)} alarm"
                    )
                    print(
                        f"Toplu onay: {len(batch_bekleyen)} alarm "
                        "onaylanacak. SPACE=onay, X=iptal"
                    )
                else:
                    print("Onaylanabilir aktif alarm yok.")

    except KeyboardInterrupt:
        print("\nKullanici tarafindan durduruldu.")
        _logger.info("KULLANICI CTRL+C ILE DURDURDU")

    except Exception as hata:
        _logger.exception(f"KRITIK PROGRAM HATASI: {hata}")
        print(f"\nKRITIK PROGRAM HATASI: {hata}")

    finally:
        try:
            olay_yaz(
                "shutdown",
                normal=normal_kapanis,
                jsonl_errors=_jsonl_hata["count"],
            )
        except Exception:
            pass

        try:
            if kamera is not None:
                kamera.release()
        except Exception:
            pass

        kamera = None

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        _logger.info("SISTEM KAPATILDI")
        print("Kamera kapatildi.")
        print(
            f"Alarm olay gecmisi: "
            f"{ALARM_GECMIS_JSONL.resolve()}"
        )


if __name__ == "__main__":
    main()
