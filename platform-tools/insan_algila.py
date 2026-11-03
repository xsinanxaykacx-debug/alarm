import cv2
import time
from ultralytics import YOLO

CAMERA_INDEX = 1
MODEL_NAME = "yolo11n.pt"
CONFIDENCE = 0.15
DETECT_EVERY_SECONDS = 0.20
DETECT_WIDTH = 640

def main():
    print("YOLO modeli yukleniyor...")
    model = YOLO(MODEL_NAME)

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("Plugcam kamerası açılamadı.")

    print("Insan algilama basladi. Cikmak icin Q.")

    last_detect = 0.0
    boxes_to_draw = []
    fps_start = time.perf_counter()
    fps_frames = 0
    fps_value = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Kameradan goruntu alinamadi.")
                break

            now = time.perf_counter()

            if now - last_detect >= DETECT_EVERY_SECONDS:
                h, w = frame.shape[:2]
                resized = cv2.resize(
                    frame,
                    (DETECT_WIDTH, int(h * DETECT_WIDTH / w))
                )

                results = model.predict(
                    resized,
                    imgsz=640,
                    conf=CONFIDENCE,
                    classes=[0],
                    verbose=False
                )

                scale_x = w / resized.shape[1]
                scale_y = h / resized.shape[0]
                boxes_to_draw = []

                for result in results:
                    if result.boxes is None:
                        continue

                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        confidence = float(box.conf[0])
                        boxes_to_draw.append((
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(x2 * scale_x),
                            int(y2 * scale_y),
                            confidence
                        ))

                last_detect = now

            for x1, y1, x2, y2, confidence in boxes_to_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    frame,
                    f"INSAN {confidence:.0%}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            fps_frames += 1
            elapsed = now - fps_start
            if elapsed >= 1:
                fps_value = fps_frames / elapsed
                fps_frames = 0
                fps_start = now

            cv2.putText(
                frame,
                f"FPS: {fps_value:.1f} | INSAN: {len(boxes_to_draw)}",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

            cv2.imshow("Plugcam - YOLO Insan Algilama", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
