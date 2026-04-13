from ultralytics import YOLO


def main():
    # 1. Load model dasar (boleh ganti ke 'yolov8s.pt' kalau mau sedikit lebih berat)
    model = YOLO("yolov8n.pt")

    # 2. Path ke data.yaml untuk dataset fire-smoke
    data_yaml_path = "datasets/helmet/data.yaml"  # sesuaikan kalau beda

    # 3. Training
    model.train(
        data=data_yaml_path,      # file yaml
        epochs=40,                # boleh kamu naik/turunin
        imgsz=640,
        batch=4,                 # sesuaikan VRAM (turunin kalau OOM)
        workers=2,
        device=0,                 # pakai GPU 0 (RTX 4050 kamu)
        name="helmet27-yolov8n",      # nama eksperimen
        project="runs/train"      # folder output
    )

    # 4. (opsional) evaluasi pakai val set dari yaml
    model.val(
        data=data_yaml_path,
        imgsz=640,
        device=0
    )


if __name__ == "__main__":
    main()
