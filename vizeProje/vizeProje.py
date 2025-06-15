import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLO v8 modelini yükleyin
model = YOLO("yolov8n.pt")

# Webcam açma
cap = cv2.VideoCapture(0)

# İşleme yöntemleri
def log_transform_with_contrast(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * (np.log(img + 1))
    log_img = np.array(log_img, dtype=np.uint8)

    # Kontrast artırma
    log_img = cv2.normalize(log_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return log_img

def gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

# Performans raporlama fonksiyonu
def evaluate_method(method_name, processed_frame, display_frame):
    start_time = time.time()
    results = model(processed_frame)  # YOLO modeline görüntüyü gönder
    detections = results[0].boxes  # Algılamaları al
    elapsed_time = time.time() - start_time

    confidences = []  # Güven skorlarını sakla

    # Algılamaları işaretle ve güven testi ekle
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Algılanan nesnenin dikdörtgen koordinatları
        confidence = detection.conf[0]  # Algılamanın güven skoru
        confidences.append(confidence)  # Güven skorlarını listeye kaydet
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #Algılanan nesnenin etrafına yeşil bir dikdörtgen çizer.
        cv2.putText(display_frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)#Dikdörtgenin üzerine güven skorunu yazar.

    # Performansı konsolda raporla
    print(f"{method_name}: Algılama süresi = {elapsed_time:.4f} saniye, Algılanan yüz sayısı = {len(detections)}, Ortalama Güven = {np.mean(confidences) if confidences else 0:.2f}")

    return display_frame, confidences

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Orijinal görüntüye güven testi ekle
    original_frame = frame.copy()
    _, original_confidences = evaluate_method("Original Frame", original_frame, original_frame)

    # Her yöntem için işlem yap ve güven skorlarını kıyasla
    log_frame, log_confidences = evaluate_method("Log Dönüşümü", log_transform_with_contrast(frame.copy()), frame.copy())
    gamma_frame, gamma_confidences = evaluate_method("Gamma Dönüşümü", gamma_correction(frame.copy(), gamma=2.0), frame.copy())
    hist_eq_frame, hist_confidences = evaluate_method("Histogram Eşitleme", histogram_equalization(frame.copy()), frame.copy())
    gauss_frame, gauss_confidences = evaluate_method("Gaussian Blur", gaussian_blur(frame.copy()), frame.copy())

    # Güven skorlarını kıyasla
    print("=== Güven Skorları Karşılaştırması ===")
    print(f"Original Frame: Ortalama Güven = {np.mean(original_confidences) if original_confidences else 0:.2f}")
    print(f"Log Dönüşümü: Ortalama Güven = {np.mean(log_confidences) if log_confidences else 0:.2f}")
    print(f"Gamma Dönüşümü: Ortalama Güven = {np.mean(gamma_confidences) if gamma_confidences else 0:.2f}")
    print(f"Histogram Eşitleme: Ortalama Güven = {np.mean(hist_confidences) if hist_confidences else 0:.2f}")
    print(f"Gaussian Blur: Ortalama Güven = {np.mean(gauss_confidences) if gauss_confidences else 0:.2f}")
    print("=====================================")

    # Görüntüleri ayrı pencerelerde göster
    cv2.imshow("Original Frame", original_frame)
    cv2.imshow("Log Transform with Contrast", log_frame)
    cv2.imshow("Gamma Correction", gamma_frame)
    cv2.imshow("Histogram Equalization", hist_eq_frame)
    cv2.imshow("Gaussian Blur", gauss_frame)


    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
