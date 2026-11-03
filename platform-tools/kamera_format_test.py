import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Plugcam kamerasi acilamadi.")

print("Backend:", cap.getBackendName())
print("Genislik:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Yukseklik:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("FourCC:", int(cap.get(cv2.CAP_PROP_FOURCC)))

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print("Format:", "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)))

for i in range(10):
    ok, frame = cap.read()
    if not ok:
        print("Kare okunamadi:", i)
        break
    print(f"Kare {i+1}: {frame.shape}, ortalama parlaklik={frame.mean():.1f}")

cap.release()
