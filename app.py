import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import logging
from typing import Dict, List, Any, Optional

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Konfigurasi aplikasi
app.config.update({
    'UPLOAD_FOLDER': '/tmp/images',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'SQLALCHEMY_DATABASE_URI': os.environ.get('DATABASE_URL'),
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key')
})

# Inisialisasi database
db = SQLAlchemy(app)

# Model Database
class DetectionResult(db.Model):
    __tablename__ = 'detection_results'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    detection_time = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    food_name = db.Column(db.String(50), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False)
    quality_score = db.Column(db.Float, nullable=False)
    bbox_coordinates = db.Column(db.String(100), nullable=False)

# Inisialisasi Model ML
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def initialize_models() -> bool:
    """Initialize ML models with proper error handling"""
    try:
        from utils.food_detector import FoodDetector
        from utils.quality_predictor import QualityPredictor
        
        MODEL_URL = os.environ.get('MODEL_URL')
        LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "best.onnx")
        
        if MODEL_URL and not os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Downloading model from {MODEL_URL}")
            import requests
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(LOCAL_MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Model downloaded successfully")
        
        if not os.path.exists(LOCAL_MODEL_PATH):
            logger.error("Model file not found")
            return False
            
        global detector, quality_predictor
        detector = FoodDetector(LOCAL_MODEL_PATH, device='cpu')
        quality_predictor = QualityPredictor(LOCAL_MODEL_PATH, device='cpu')
        return True
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        return False

# Global model instances
detector = None
quality_predictor = None
if not initialize_models():
    logger.warning("Running in degraded mode without ML models")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_tables():
    """Initialize database tables"""
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

@app.route('/')
def home() -> str:
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    try:
        # Data untuk chart
        seven_days_data = db.session.query(
            db.func.date(DetectionResult.detection_time).label('date'),
            db.func.avg(DetectionResult.quality_score).label('avg_quality')
        ).filter(DetectionResult.detection_time >= datetime.now() - timedelta(days=7))\
         .group_by(db.func.date(DetectionResult.detection_time))\
         .order_by(db.func.date(DetectionResult.detection_time))\
         .all()

        dates = [d.date.strftime('%Y-%m-%d') for d in seven_days_data]
        avg_qualities = [float(d.avg_quality) if d.avg_quality else 0 for d in seven_days_data]

        # Statistik
        total_analyses = DetectionResult.query.count()
        avg_quality = db.session.query(db.func.avg(DetectionResult.quality_score)).scalar() or 0
        top_food = db.session.query(
            DetectionResult.food_name,
            db.func.count(DetectionResult.id).label('count')
        ).group_by(DetectionResult.food_name).order_by(db.desc('count')).first()

        return render_template('dashboard.html',
                            dates=dates,
                            avg_qualities=avg_qualities,
                            total_analyses=total_analyses,
                            avg_quality=avg_quality,
                            top_food=top_food)
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return render_template('error.html', error="Gagal memuat dashboard"), 500

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('analyze.html')

@app.route('/predict', methods=['POST'])
def predict():
    if detector is None or quality_predictor is None:
        return jsonify({"error": "Model not available"}), 503

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Baca gambar
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Deteksi makanan
        detected_foods = detector.detect(img)
        if not detected_foods:
            return jsonify({"error": "No food detected"}), 400

        results = []
        for food in detected_foods:
            try:
                x1, y1, x2, y2 = map(int, food['bbox'])
                cropped_img = img[y1:y2, x1:x2]
                
                if cropped_img.size == 0:
                    continue
                
                # Prediksi kualitas
                quality = quality_predictor.predict_quality(cropped_img, food["confidence"])
                
                # Simpan ke database
                new_result = DetectionResult(
                    filename=secure_filename(file.filename),
                    image_path='memory',
                    food_name=food["name"],
                    confidence=float(food["confidence"]),
                    quality_score=float(quality),
                    bbox_coordinates=f"{x1},{y1},{x2},{y2}"
                )
                db.session.add(new_result)
                db.session.commit()
                
                results.append({
                    "name": food["name"],
                    "confidence": float(food["confidence"]),
                    "quality": float(quality),
                    "bbox": [x1, y1, x2, y2]
                })
                
            except Exception as e:
                logger.error(f"Error processing food item: {str(e)}")
                db.session.rollback()
                continue

        if not results:
            return jsonify({"error": "No valid results"}), 400

        # Gambar hasil dengan bounding box
        output_img = detector.draw_results(img.copy(), results)
        _, buffer = cv2.imencode('.jpg', output_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "results": results,
            "image": img_base64
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/history')
def history():
    try:
        food_name = request.args.get('food_name', '').strip()
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        query = DetectionResult.query
        
        if food_name:
            query = query.filter(DetectionResult.food_name.ilike(f'%{food_name}%'))
        
        if start_date:
            query = query.filter(DetectionResult.detection_time >= start_date)
            
        if end_date:
            query = query.filter(DetectionResult.detection_time <= end_date)
        
        results = query.order_by(DetectionResult.detection_time.desc()).limit(50).all()
        
        return render_template('history.html', 
                            results=results,
                            food_name=food_name,
                            start_date=start_date,
                            end_date=end_date)
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/api/detections', methods=['GET'])
def api_detections():
    try:
        limit = request.args.get('limit', default=10, type=int)
        results = DetectionResult.query.order_by(
            DetectionResult.detection_time.desc()
        ).limit(limit).all()
        
        return jsonify([{
            "id": r.id,
            "food_name": r.food_name,
            "confidence": r.confidence,
            "quality_score": r.quality_score,
            "detection_time": r.detection_time.isoformat()
        } for r in results])
    except Exception as e:
        logger.error(f"API detections error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error="Server error"), 500

# Inisialisasi database
create_tables()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)
