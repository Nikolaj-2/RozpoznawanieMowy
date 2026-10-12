import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

MODEL_PATH = "E:/Studia_Materiały/PŚK/Semestr_7/Mowa_Proj/Final/proj/best.pt"

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.fuse()
model.to(device)

def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    results = model(
        file_path,
        conf=0.4,  #próg epwności
        iou=0.5  #próg NMS
    )[0]

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except:
        font = ImageFont.load_default()

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(y1 - 20, 0)), label, fill="red", font=font)

    tk_img = ImageTk.PhotoImage(img)

    img_label.config(image=tk_img)
    img_label.image = tk_img

    #resize okna tak aby zdjęcie sie zmieściło
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    new_w = min(img.width + 40, screen_w - 100)
    new_h = min(img.height + 100, screen_h - 100)

    root.geometry(f"{new_w}x{new_h}")


root = tk.Tk()
root.title("YOLOv8 Rozpoznawanie Zwierząt")

btn = tk.Button(root, text="Wybierz obraz", command=predict_image)
btn.pack(pady=10)

img_label = tk.Label(root)
img_label.pack()

root.mainloop()
