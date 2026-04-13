from ultralytics import YOLO
import cv2
import os

def main():
    # Load model
    model = YOLO("runs/train/helmet-yolov8n/weights/yolov8n.pt")

    # Folder input & output
    input_folder = "images"          # ganti dengan folder gambar kamu
    output_folder = "results1"        # hasil akan disimpan di sini

    os.makedirs(output_folder, exist_ok=True)

    # Loop semua file di folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, filename)

            # Baca gambar
            img = cv2.imread(img_path)
            if img is None:
                print(f"Gagal baca: {filename}")
                continue

            # Inferensi
            results = model(img, conf=0.2)

            # Ambil hasil annotated
            annotated = results[0].plot()

            # Simpan hasil
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, annotated)

            print(f"Processed: {filename}")

    print("Selesai semua!")

if __name__ == "__main__":
    main()