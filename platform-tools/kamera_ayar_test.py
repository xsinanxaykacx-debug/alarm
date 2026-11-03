import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Plugcam acilamadi.")

print("Kamera ayarlari penceresi acilmaya calisiliyor...")
cap.set(cv2.CAP_PROP_SETTINGS, 1)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.imshow("Ham Plugcam Goruntusu - Q ile cik", frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
