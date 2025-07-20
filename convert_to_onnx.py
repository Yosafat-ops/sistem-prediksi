from ultralytics import YOLO
import os

# Pastikan folder 'models' ada
os.makedirs('models', exist_ok=True)

# Load model YOLO
model = YOLO('models/best.pt')  # Ganti dengan path model Anda

# Export ke format ONNX
model.export(format='onnx', imgsz=[640, 640])  # Sesuaikan ukuran input
print("Model berhasil dikonversi ke ONNX!")