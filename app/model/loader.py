import json
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent.parent / 'models'

DEFAULT_METRICS = {
    'accuracy': 'N/A',
    'precision': 'N/A',
    'recall': 'N/A',
    'f1_score': 'N/A',
}

def _load_metrics(filename):
    metrics_path = MODEL_DIR / filename
    if not metrics_path.exists():
        return DEFAULT_METRICS.copy()

    with open(metrics_path, encoding='utf-8') as f:
        return json.load(f)


resnet_metrics = _load_metrics('resnet_metrics.json')
mobilenet_metrics = _load_metrics('mobilenet_metrics.json')


@lru_cache(maxsize=1)
def get_models():
    import tensorflow as tf

    resnet_model = tf.keras.models.load_model(MODEL_DIR / 'resnet50_final.h5')
    mobilenet_model = tf.keras.models.load_model(MODEL_DIR / 'mobilenetv2_final.h5')
    return resnet_model, mobilenet_model