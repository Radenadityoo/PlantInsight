import numpy as np
from PIL import Image

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((160, 160))
    img_array = np.asarray(img, dtype='float32')
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array