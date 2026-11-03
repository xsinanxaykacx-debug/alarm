import cv2
from pathlib import Path

out = Path("kalite_testi")
out.mkdir(exist_ok=True)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Plugcam acilamadi. Plugcam uygulamasini acik tut.")

try:
    # Kameranin otomatik pozlama ve odak ayarlari otursun.
    for _ in range(30):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Kameradan goruntu alinamadi.")

    cv2.imwrite(str(out / "orijinal.png"), frame)
    cv2.imwrite(
        str(out / "orijinal_yuksek_jpeg.jpg"),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 100]
    )

    print("Goruntu boyutu:", frame.shape[1], "x", frame.shape[0])
    print("Kaydedildi:", out.resolve())
finally:
    cap.release()
