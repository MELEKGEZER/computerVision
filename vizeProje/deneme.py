import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLO v8 modelini yükleyin
model = YOLO("yolov8n.pt")  # Uygun model dosyasını buraya yerleştirin

# Webcam açma
cap = cv2.VideoCapture(0)

# İşleme yöntemleri
def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * (np.log(img + 1))
    log_img = np.array(log_img, dtype=np.uint8)
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
def evaluate_method(method_name, frame):
    start_time = time.time()
    results = model(frame)  # YOLO modeline görüntüyü gönder
    detections = results[0].boxes  # Algılamaları al
    elapsed_time = time.time() - start_time

    # Algılamaları işaretle ve güven testi ekle
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        confidence = detection.conf[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Performansı konsolda raporla
    print(f"{method_name}: Algılama süresi = {elapsed_time:.4f} saniye, Algılanan yüz sayısı = {len(detections)}")

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Her yöntem için işlem yap
    log_frame = evaluate_method("Log Dönüşümü", log_transform(frame.copy()))
    gamma_frame = evaluate_method("Gamma Dönüşümü", gamma_correction(frame.copy(), gamma=2.0))
    hist_eq_frame = evaluate_method("Histogram Eşitleme", histogram_equalization(frame.copy()))
    gauss_frame = evaluate_method("Gaussian Blur", gaussian_blur(frame.copy()))

    # Görüntüleri ayrı pencerelerde göster
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Log Transform", log_frame)
    cv2.imshow("Gamma Correction", gamma_frame)
    cv2.imshow("Histogram Equalization", hist_eq_frame)
    cv2.imshow("Gaussian Blur", gauss_frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
