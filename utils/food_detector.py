import cv2
from ultralytics import YOLO
import numpy as np

class FoodDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
    
    def detect(self, image):
        results = self.model.predict(image)
        detections = []
        for result in results:
            for box in result.boxes:
                # Pastikan koordinat berupa integer dan dalam format [x1,y1,x2,y2]
                bbox = [int(coord) for coord in box.xyxy[0].tolist()]
                if len(bbox) != 4:  # Validasi panjang bounding box
                    continue
                    
                detections.append({
                    "name": self.class_names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": bbox  # Sudah dalam format yang benar
                })
        return detections
    
    def draw_results(self, image, results):
        output_image = image.copy()
        for item in results:
            try:
                # Pastikan bounding box memiliki 4 nilai dan berupa integer
                if 'bbox' not in item or len(item['bbox']) != 4:
                    continue
                    
                x1, y1, x2, y2 = map(int, item['bbox'])
                
                # Gambar rectangle
                cv2.rectangle(output_image, 
                            (x1, y1), 
                            (x2, y2), 
                            (0, 255, 0), 
                            2)
                
                # Tambahkan teks label
                label = f"{item['name']} ({item.get('quality', '?')}/5)"
                cv2.putText(output_image, 
                           label, 
                           (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, 
                           (0, 255, 0), 
                           2)
            except Exception as e:
                print(f"Error drawing box: {e}")
                continue
                
        return output_image