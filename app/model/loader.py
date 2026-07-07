import json
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent.parent / 'models'

DEFAULT_METRICS = {
    'accuracy': 0.9917440414428711,
    'precision': 0.991921459579247,
    'recall': 0.9917440660474717,
    'f1_score': 0.9917462958470399,
}

DEFAULT_MOBILENET_METRICS = {
    'accuracy': 0.978844165802002,
    'precision': 0.9792208062050062,
    'recall': 0.9788441692466461,
    'f1_score': 0.9788552309985733,
}

def _load_metrics(filename):
    metrics_path = MODEL_DIR / filename
    if not metrics_path.exists():
        return DEFAULT_METRICS.copy()

    with open(metrics_path, encoding='utf-8') as f:
        return json.load(f)


resnet_metrics = _load_metrics('resnet_metrics.json')
mobilenet_metrics = _load_metrics('mobilenet_metrics.json')

if resnet_metrics == DEFAULT_METRICS:
    resnet_metrics = DEFAULT_METRICS.copy()

if mobilenet_metrics == DEFAULT_METRICS:
    mobilenet_metrics = DEFAULT_MOBILENET_METRICS.copy()


@lru_cache(maxsize=1)
def get_models():
    import tensorflow as tf

    resnet_model = tf.keras.models.load_model(MODEL_DIR / 'resnet50_final.h5')
    mobilenet_model = tf.keras.models.load_model(MODEL_DIR / 'mobilenetv2_final.h5')
    return resnet_model, mobilenet_model