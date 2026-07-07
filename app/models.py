from datetime import datetime

from .extensions import db


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uploaded_image = db.Column(db.String(255), nullable=False)
    resnet_label = db.Column(db.String(100), nullable=False)
    resnet_confidence = db.Column(db.Float, nullable=False)
    mobilenet_label = db.Column(db.String(100), nullable=False)
    mobilenet_confidence = db.Column(db.Float, nullable=False)
    best_model = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)