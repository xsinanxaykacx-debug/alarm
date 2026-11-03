import cv2

for backend, name in [
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_MSMF, "MSMF"),
]:
    cap = cv2.VideoCapture(1, backend)
    print(f"\n--- {name} ---")

    if not cap.isOpened():
        print("Kamera acilamadi")
        continue

    print("Baslangic:", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for w, h in [(1920, 1080), (1280, 720), (640, 480)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, frame = cap.read()

        if ok:
            print(f"Istenen {w}x{h} -> Gelen "
                  f"{frame.shape[1]}x{frame.shape[0]}")

    cap.release()
