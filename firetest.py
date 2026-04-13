from ultralytics import YOLO
import cv2

def main():
    # Load model
    model = YOLO("runs/train/helmet-yolov8n/weights/yolov8n.pt")

    # URL stream
    stream_url = "https://testing-apicctv.gresikkab.go.id/uploads/hls/tbe0l9twnxn81xdv08gw4pjt.m3u8"

    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Tidak bisa membuka stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal ambil frame")
            break

        # Ambil ukuran frame
        h, w, _ = frame.shape

        # 🔥 Area deteksi: setengah atas saja
        frame_top = frame[0:h//2, 0:w]

        # Inferensi
        results = model(frame_top, conf=0.5)

        # Ambil bounding box
        boxes = results[0].boxes

        # Gambar ke frame asli
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Gambar rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # (Optional) garis pembatas area deteksi
        cv2.line(frame, (0, h//2), (w, h//2), (255, 0, 0), 2)

        # Tampilkan FULL frame (tidak ke-crop)
        cv2.imshow("Helmet Detection (Top Only)", frame)

        # Keluar dengan tombol q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(model.names)


if __name__ == "__main__":
    main()