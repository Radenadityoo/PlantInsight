import os
import random
import time

import numpy as np

from app.model.loader import get_models, mobilenet_metrics, resnet_metrics
from app.model.preprocess import preprocess_image

USE_DUMMY = os.environ.get('PLANTINSIGHT_DEMO_MODE', '0') == '1' or os.environ.get('VERCEL') == '1'

CLASS_NAMES = ['black_spot', 'canker', 'greening', 'healthy', 'other']

def get_label(pred):
    return CLASS_NAMES[pred.argmax()]

# ================= DUMMY =================
def dummy_prediction():
    time.sleep(1)
    return {
        'resnet': {
            'label': random.choice(CLASS_NAMES),
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'time': 0.12,
            'metrics': resnet_metrics
        },
        'mobilenet': {
            'label': random.choice(CLASS_NAMES),
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'time': 0.05,
            'metrics': mobilenet_metrics
        }
    }

# ================= REAL =================
def real_prediction(img_path):
    img = preprocess_image(img_path)
    resnet_model, mobilenet_model = get_models()
    
    # ResNet
    start = time.time()
    resnet_pred = resnet_model.predict(img)
    resnet_time = time.time() - start

    # MobileNet
    start = time.time()
    mobilenet_pred = mobilenet_model.predict(img)
    mobilenet_time = time.time() - start

    return {
        'resnet': {
            'label': get_label(resnet_pred),
            'confidence': float(resnet_pred.max()),
            'time': round(resnet_time, 3),
            'metrics': resnet_metrics
        },
        'mobilenet': {
            'label': get_label(mobilenet_pred),
            'confidence': float(mobilenet_pred.max()),
            'time': round(mobilenet_time, 3),
            'metrics': mobilenet_metrics
        }
    }

def predict_image(img_path):
    return dummy_prediction() if USE_DUMMY else real_prediction(img_path)