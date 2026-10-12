from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")  # lekka wersja, szybko działa

    model.train(
        data="data.yaml",
        epochs=120,
        imgsz=640,
        batch=16,
        device=0  # GPU
    )