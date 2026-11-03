import cv2

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("HATA: Plugcam kamerası açılamadı.")
    raise SystemExit(1)

print("Canli goruntu acildi. Cikmak icin Q tusuna bas.")

while True:
    ok, frame = camera.read()

    if not ok:
        print("Goruntu alinamadi.")
        break

    cv2.imshow("Magaza Kamerasi - Plugcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
