import cv2
import time
import winsound
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# AYARLAR
# ============================================================
KAMERA_INDEX = 1 
MODEL_YOLU = r"C:\Users\bzdye\Downloads\platform-tools-latest-windows\platform-tools\runs\classify\polis_model_sonuclari\kirpilmis_model_v1\weights\best.pt"

ARAC_SINIFLARI = [2, 3, 5, 7]
GUVEN_ESIGI = 0.85 

# Klasör oluştur
KAYIT_KLASORU = Path("polis_kayitlari")
KAYIT_KLASORU.mkdir(exist_ok=True)

# ============================================================
# ANA DÖNGÜ
# ============================================================
print("Modeller yükleniyor...")
det_model = YOLO("yolo11n.pt")
cls_model = YOLO(MODEL_YOLU)

print("Kamera başlatılıyor (Index: 1)...")
cap = cv2.VideoCapture(KAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı. Plugcam açık mı?")

alarm_aktif = False
son_alarm_zamani = 0
yanip_sonme_durumu = False
susturuldu = False # Manuel susturma durumu
onceki_kare_polisti = False # Aracın kapıdan çıkıp çıkmadığını anlamak için

print("Sistem çalışıyor. Çıkmak için 'q' tuşuna bas.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    
    polis_var_mi = False
    en_yuksek_guven = 0
    arac_kirpilmis_resim = None

    # 1. Araçları Tespit Et
    det_results = det_model.predict(frame, conf=0.4, verbose=False, classes=ARAC_SINIFLARI)
    
    for box in det_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if (x2 - x1) < 40 or (y2 - y1) < 40:
            continue

        arac_resmi = frame[y1:y2, x1:x2]

        # 2. Polis mi Kontrol Et
        cls_results = cls_model.predict(arac_resmi, imgsz=224, verbose=False)
        
        if len(cls_results) > 0:
            probs = cls_results[0].probs
            sinif_adi = cls_results[0].names[probs.top1]
            guven = float(probs.top1conf)

            if "normal" in sinif_adi.lower():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{sinif_adi} %{guven*100:.0f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if "polis" in sinif_adi.lower() and guven >= GUVEN_ESIGI:
                if guven > en_yuksek_guven:
                    en_yuksek_guven = guven
                    polis_var_mi = True
                    
                    # Ekranda kırmızı kutu çiz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"POLIS %{guven*100:.0f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Fotoğraf kaydetmek için kırp
                    arac_kirpilmis_resim = frame[y1:y2, x1:x2].copy()

    su_an = time.time()

    # 3. Alarm Mantığı ve Susturma
    if polis_var_mi:
        # Eğer önceden polis yoksa (yeni araç girdiyse), susturmayı kaldır
        if not onceki_kare_polisti:
            susturuldu = False 
        
        # Eğer alarm çalmıyorsa ve susturulmamışsa alarmı başlat
        if not alarm_aktif and not susturuldu:
            alarm_aktif = True
            print("🚨 POLİS TESPİT EDİLDİ! ALARM ÇALIYOR!")
            
            # --- FOTOĞRAF KAYDETME ---
            if arac_kirpilmis_resim is not None:
                zaman_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                dosya_adi = KAYIT_KLASORU / f"polis_{zaman_str}.jpg"
                cv2.imwrite(str(dosya_adi), arac_kirpilmis_resim)
                print(f"📸 Fotoğraf kaydedildi: {dosya_adi}")

    else:
        # Ekranda polis yoksa alarmı ve susturmayı sıfırla
        alarm_aktif = False
        susturuldu = False
        
    onceki_kare_polisti = polis_var_mi

    # 4. Ekran Efektleri
    if alarm_aktif:
        # Yanıp sönen kırmızı efekt
        yanip_sonme_durumu = not yanip_sonme_durumu
        if yanip_sonme_durumu:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
        # Büyük Uyarı Yazısı
        metin = "!!! POLIS ARACI !!!"
        (w, h), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
        cv2.putText(frame, metin, ((frame.shape[1]-w)//2, (frame.shape[0]+h)//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        
        # Sustur Butonu (Ekranın sağ altına)
        cv2.putText(frame, "[S] SUSTUR", (frame.shape[1]-250, frame.shape[0]-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Bip Sesi
        winsound.Beep(1500, 200)

    # 5. Klavye Kontrolü
    tus = cv2.waitKey(1) & 0xFF
    if tus == ord("q"):
        break
    elif tus == ord("s") or tus == ord("S"):
        if alarm_aktif:
            alarm_aktif = False
            susturuldu = True
            print("🔇 Alarm susturuldu. Araç gidene kadar çalmayacak.")

    cv2.imshow("Otopark Polis Dedektoru", frame)

cap.release()
cv2.destroyAllWindows()