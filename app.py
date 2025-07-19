from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
from jinja2 import TemplateNotFound
from utils.food_detector import FoodDetector
from utils.quality_predictor import QualityPredictor
from flask import request, flash, redirect, url_for


app = Flask(__name__)

# Konfigurasi
app.config.update({
    'UPLOAD_FOLDER': 'static/images/',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///results.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False
})

db = SQLAlchemy(app)

# Model Database
class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    detection_time = db.Column(db.DateTime, default=datetime.utcnow)
    food_name = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    quality_score = db.Column(db.Float, nullable=False)
    bbox_coordinates = db.Column(db.String(100), nullable=False)

# Inisialisasi Model
detector = FoodDetector("models/best.pt")
quality_predictor = QualityPredictor("models/best.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Redirect root URL ke dashboard
@app.route('/')
def root():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    try:
        # Data untuk chart
        seven_days_data = db.session.query(
            db.func.strftime('%Y-%m-%d', DetectionResult.detection_time).label('date'),
            db.func.avg(DetectionResult.quality_score).label('avg_quality')
        ).filter(DetectionResult.detection_time >= datetime.now() - timedelta(days=7))\
         .group_by(db.func.strftime('%Y-%m-%d', DetectionResult.detection_time))\
         .order_by(db.func.strftime('%Y-%m-%d', DetectionResult.detection_time))\
         .all()

        # Extract dates and average qualities
        dates = [d.date for d in seven_days_data]  # Already formatted as string
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
        print(f"Dashboard error: {e}")
        return render_template('dashboard/error.html', error="Gagal memuat dashboard")

# Analisis Kualitas (Integrasi dengan route yang ada)
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        # [Kode yang sama dari route /predict sebelumnya]
        return render_template('result.html', foods=results, result_image=output_path)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Proses gambar
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return render_template('result.html', error="Format gambar tidak didukung")

        # Deteksi makanan
        detected_foods = detector.detect(img)
        if not detected_foods:
            return render_template('result.html', error="Tidak terdeteksi makanan India!")

        results = []
        analysis_data = {
            'total_items': 0,
            'avg_quality': 0,
            'quality_distribution': {},
            'details': []
        }

        for food in detected_foods:
            try:
                x1, y1, x2, y2 = map(int, food['bbox'])
                cropped_img = img[y1:y2, x1:x2]
                
                if cropped_img.size == 0:
                    continue
                
                # Analisis kualitas
                quality = quality_predictor.predict_quality(cropped_img, food["confidence"])
                
                # Simpan data untuk analitik
                analysis_data['total_items'] += 1
                analysis_data['avg_quality'] += quality
                analysis_data['details'].append({
                    'name': food["name"],
                    'quality': quality,
                    'confidence': food["confidence"],
                    'image': f'crop_{len(results)}.jpg'
                })
                
                # Simpan crop gambar untuk analisis detail
                cv2.imwrite(f'static/images/crop_{len(results)}.jpg', cropped_img)

                # Simpan ke database
                new_result = DetectionResult(
                    filename=secure_filename(file.filename),
                    image_path='static/images/result.jpg',
                    food_name=food["name"],
                    confidence=float(food["confidence"]),
                    quality_score=float(quality),
                    bbox_coordinates=f"{x1},{y1},{x2},{y2}"
                )
                db.session.add(new_result)
                
                results.append({
                    "name": food["name"],
                    "confidence": float(food["confidence"]),
                    "quality": float(quality),
                    "bbox": [x1, y1, x2, y2],
                    "db_id": new_result.id
                })
                
            except Exception as e:
                print(f"Error processing food item: {e}")
                continue

        # Hitung rata-rata kualitas
        if analysis_data['total_items'] > 0:
            analysis_data['avg_quality'] /= analysis_data['total_items']
        
        db.session.commit()

        if not results:
            return render_template('result.html', error="Tidak ada hasil valid")

        try:
            # Simpan gambar hasil
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
            annotated_img = detector.draw_results(img.copy(), results)
            cv2.imwrite(output_path, annotated_img)
            
            # Update database
            for result in results:
                if 'db_id' in result:
                    record = DetectionResult.query.get(result['db_id'])
                    if record:
                        record.image_path = output_path
            db.session.commit()
            
            # Render hasil dengan data analitik
            return render_template('result.html', 
                                foods=results, 
                                result_image=output_path,
                                analysis=analysis_data)
                
        except Exception as e:
            print(f"Error saving image: {e}")
            return render_template('result.html', 
                                foods=results, 
                                error="Gagal menyimpan gambar")

    except Exception as e:
        db.session.rollback()
        print(f"Unexpected error: {e}")
        return render_template('result.html', 
                            error=f"Terjadi kesalahan: {str(e)}")

# History (Integrasi dengan route yang ada)
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
        
        results = query.order_by(DetectionResult.detection_time.desc()).all()
        
        return render_template('history.html', 
                            results=results,
                            food_name=food_name,
                            start_date=start_date,
                            end_date=end_date)
    except Exception as e:
        return render_template('error.html', error="Gagal memuat history")

@app.route('/delete/<int:id>', methods=['DELETE'])
def delete_result(id):
    try:
        # Cari dan hapus record dari database
        result = DetectionResult.query.get_or_404(id)
        
        # Hapus file gambar jika ada
        if result.image_path:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], result.image_path))
            except Exception as e:
                app.logger.error(f"Gagal menghapus file gambar: {str(e)}")
        
        db.session.delete(result)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Data berhasil dihapus"})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

# Profil Perusahaan
@app.route('/about')
def about():
    return render_template('about.html')

# [Tambahkan route lainnya yang sudah ada...]

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

# API Endpoints
@app.route('/api/dashboard/quality-trend')
def quality_trend_api():
    try:
        days = request.args.get('days', default=30, type=int)
        
        trend_data = (
            db.session.query(
                db.func.date(DetectionResult.detection_time).label('date'),
                db.func.avg(DetectionResult.quality_score).label('avg_quality')
            )
            .filter(DetectionResult.detection_time >= datetime.now() - timedelta(days=days))
            .group_by('date')
            .order_by('date')
            .all()
        )
        
        return jsonify([{
            'date': d.date.strftime('%Y-%m-%d'),
            'quality': float(d.avg_quality) if d.avg_quality else 0
        } for d in trend_data])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main Application Entry
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
