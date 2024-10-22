import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Resmi analiz edecek bir model yükleme
model = MobileNetV2(weights="imagenet")

def resim_analiz_et(image_path):
    # Resmi yükleme
    image = cv2.imread(image_path)
    
    # Resmi uygun forma getirip işleme
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    # Tahmin yapma
    predictions = model.predict(image)
    
    # Tahmin edilen sınıfı ve etiketini alma
    label = decode_predictions(predictions)[0][0]
    return label[1], label[2]  # Nesne etiketi ve olasılık

# Örnek: indirilmiş bir profil resminin analizi
etiket, olasilik = resim_analiz_et('profil_resmi.jpg')
print(f"Bu resim: {etiket} (%{olasilik * 100})")
