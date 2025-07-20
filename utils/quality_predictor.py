import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class QualityPredictor:
    def __init__(self, yolo_model_path):
        try:
            # Model bisa digunakan untuk ekstraksi fitur tambahan
            self.yolo_model_path = yolo_model_path
            logger.info("Quality predictor initialized")
        except Exception as e:
            logger.error(f"Quality predictor init error: {str(e)}")
            raise
    
    def predict_quality(self, cropped_img, confidence):
        try:
            # Analisis Warna
            hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            color_saturation = np.mean(hsv[:,:,1]) / 255
            
            # Analisis Tekstur
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
            
            # Gabungkan faktor
            quality_score = (
                0.4 * confidence +
                0.3 * color_saturation +
                0.3 * laplacian_var
            ) * 5
            
            return round(np.clip(quality_score, 1, 5), 1)
        except Exception as e:
            logger.error(f"Quality prediction error: {str(e)}")
            return 3.0  # Nilai default jika error
