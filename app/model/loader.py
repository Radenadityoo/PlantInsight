import json
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent.parent / 'models'

with open(MODEL_DIR / 'resnet_metrics.json', encoding='utf-8') as f:
    resnet_metrics = json.load(f)

with open(MODEL_DIR / 'mobilenet_metrics.json', encoding='utf-8') as f:
    mobilenet_metrics = json.load(f)


@lru_cache(maxsize=1)
def get_models():
    import tensorflow as tf

    resnet_model = tf.keras.models.load_model(MODEL_DIR / 'resnet50_final.h5')
    mobilenet_model = tf.keras.models.load_model(MODEL_DIR / 'mobilenetv2_final.h5')
    return resnet_model, mobilenet_model