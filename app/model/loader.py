import json
from pathlib import Path

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent.parent / 'models'

resnet_model = tf.keras.models.load_model(MODEL_DIR / 'resnet50_final.h5')
mobilenet_model = tf.keras.models.load_model(MODEL_DIR / 'mobilenetv2_final.h5')

with open(MODEL_DIR / 'resnet_metrics.json', encoding='utf-8') as f:
    resnet_metrics = json.load(f)

with open(MODEL_DIR / 'mobilenet_metrics.json', encoding='utf-8') as f:
    mobilenet_metrics = json.load(f)

print('ResNet input shape:', resnet_model.input_shape)
print('MobileNet input shape:', mobilenet_model.input_shape)