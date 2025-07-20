import cv2
from ultralytics import YOLO
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FoodDetector:
    def __init__(self, model_path):
        try:
            logger.info(f"Loading model from: {model_path}")
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def detect(self, image):
        try:
            results = self.model.predict(image)
            detections = []
            for result in results:
                for box in result.boxes:
                    bbox = [int(coord) for coord in box.xyxy[0].tolist()]
                    if len(bbox) != 4:
                        continue
                        
                    detections.append({
                        "name": self.class_names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": bbox
                    })
            return detections
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []
    
    def draw_results(self, image, results):
        try:
            output_image = image.copy()
            for item in results:
                if 'bbox' not in item or len(item['bbox']) != 4:
                    continue
                    
                x1, y1, x2, y2 = map(int, item['bbox'])
                
                cv2.rectangle(output_image, 
                            (x1, y1), 
                            (x2, y2), 
                            (0, 255, 0), 
                            2)
                
                label = f"{item['name']} ({item.get('quality', '?')}/5)"
                cv2.putText(output_image, 
                           label, 
                           (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, 
                           (0, 255, 0), 
                           2)
            return output_image
        except Exception as e:
            logger.error(f"Drawing error: {str(e)}")
            return image
