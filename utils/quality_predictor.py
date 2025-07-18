import cv2
import numpy as np
from ultralytics import YOLO

class QualityPredictor:
    def __init__(self, yolo_model_path):
        self.yolo_model = YOLO(yolo_model_path)
    
    def predict_quality(self, cropped_img, confidence):
        """Prediksi kualitas berdasarkan crop gambar dan confidence YOLO"""
        try:
            # 1. Analisis Warna (Kesegaran)
            hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            color_saturation = np.mean(hsv[:,:,1]) / 255  # Nilai 0-1
            
            # 2. Analisis Tekstur (Kematangan/Keseragaman)
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000  # Normalisasi
            
            # 3. Gabungkan dengan confidence YOLO
            quality_score = (
                0.4 * confidence +          # Kepercayaan deteksi
                0.3 * color_saturation +    # Warna
                0.3 * laplacian_var         # Tekstur
            ) * 5  # Skala 1-5
            
            return round(np.clip(quality_score, 1, 5), 1)  # Batasi range 1-5
            
        except Exception as e:
            print(f"Error in quality prediction: {e}")
            return 5.0  # Nilai default jika error