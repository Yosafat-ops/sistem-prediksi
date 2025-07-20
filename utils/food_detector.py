import cv2
import numpy as np
import logging
import onnxruntime as ort
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FoodDetector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        """Initialize ONNX runtime session for YOLO model.
        
        Args:
            model_path: Path to ONNX model file
            device: Inference device ('cpu' or 'cuda')
        """
        try:
            logger.info(f"Loading ONNX model from: {model_path}")
            
            # ONNX Runtime providers
            providers = ['CPUExecutionProvider']
            if device.lower() == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Create inference session
            self.session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"Model loaded. Input shape: {self.input_shape}")
            logger.info(f"Available providers: {ort.get_available_providers()}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for ONNX model."""
        # Resize and normalize
        img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        return img

    def postprocess(self, outputs: List[np.ndarray], conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Postprocess raw model outputs."""
        detections = []
        
        # Assuming outputs[0] has shape [1, num_detections, 6]
        # where last dim is [x1, y1, x2, y2, conf, class_id]
        for detection in outputs[0][0]:
            x1, y1, x2, y2, conf, cls_id = detection
            
            if conf < conf_threshold:
                continue
                
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "class_id": int(cls_id),
                "name": f"class_{int(cls_id)}"  # Replace with your class names
            })
            
        return detections

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run object detection on input image."""
        try:
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
            
            # Postprocess
            detections = self.postprocess(outputs)
            
            logger.debug(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return []

    def draw_results(self, image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection results on image."""
        try:
            output_img = image.copy()
            img_height, img_width = image.shape[:2]
            
            for item in results:
                if 'bbox' not in item or len(item['bbox']) != 4:
                    continue
                    
                # Convert normalized coordinates to pixel values
                x1 = int(item['bbox'][0] * img_width)
                y1 = int(item['bbox'][1] * img_height)
                x2 = int(item['bbox'][2] * img_width)
                y2 = int(item['bbox'][3] * img_height)
                
                # Draw bounding box
                cv2.rectangle(
                    output_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                
                # Draw label
                label = f"{item.get('name', 'object')} {item.get('confidence', 0):.2f}"
                cv2.putText(
                    output_img,
                    label,
                    (x1, max(20, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
            return output_img
            
        except Exception as e:
            logger.error(f"Drawing failed: {str(e)}")
            return image
