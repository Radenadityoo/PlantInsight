import os

from flask import Blueprint, current_app, render_template, request
from werkzeug.utils import secure_filename

from app.extensions import db
from app.models import PredictionHistory
from app.model.predict import predict_image
from app.services.dataset import build_dataset_summary

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_image(filepath)
        best_model = 'ResNet50' if result['resnet']['confidence'] >= result['mobilenet']['confidence'] else 'MobileNetV2'

        history_entry = PredictionHistory(
            uploaded_image=filename,
            resnet_label=result['resnet']['label'],
            resnet_confidence=result['resnet']['confidence'],
            mobilenet_label=result['mobilenet']['label'],
            mobilenet_confidence=result['mobilenet']['confidence'],
            best_model=best_model,
        )

        try:
            db.session.add(history_entry)
            db.session.commit()
        except Exception:
            current_app.logger.exception('Failed to save prediction history')
            db.session.rollback()

        return render_template('result.html', result=result, image=filename)

    return 'File must be an image'


@main.route('/history')
def history():
    records = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('history.html', records=records)

@main.route('/about')
def about():
    summary = build_dataset_summary()
    return render_template(
        'about.html',
        dataset_distribution=summary['distribution'],
        total=summary['total'],
        dataset_root=summary['root'],
    )