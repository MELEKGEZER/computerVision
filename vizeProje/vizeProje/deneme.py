import cv2
import numpy as np
from ultralytics import YOLO

# Yolo v8 modelini yükleyin
model = YOLO("yolov8n.pt")  # Uygun model dosyasını buraya yerleştirin (örneğin yolov8n.pt)

# Webcam açma
cap = cv2.VideoCapture(0)

def log_transform(img):
    # Log dönüşümü
    c = 255 / np.log(1 + np.max(img))
    log_img = c * (np.log(img + 1))
    log_img = np.array(log_img, dtype=np.uint8)
    return log_img

def gamma_correction(img, gamma=1.0):
    # Gamma düzeltme
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def histogram_equalization(img):
    # Histogram eşitleme (renkli görüntüye uygulanır)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def gaussian_blur(img):
    # Gaussian filtreleme
    return cv2.GaussianBlur(img, (15, 15), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Yolo v8 ile yüz tespiti
    results = model(frame)
    detections = results[0].boxes

    # Yüzleri kare içine alma ve güven testi ekleme
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Tespit edilen kutunun koordinatları
        confidence = detection.conf[0]  # Güven oranı (0-1 arasında)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil kare çiz
        cv2.putText(
            frame,
            f"{confidence:.2f}",  # Güven oranını yaz
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Görüntü işleme yöntemleri
    log_frame = log_transform(frame)
    gamma_frame = gamma_correction(frame, gamma=2.0)
    hist_eq_frame = histogram_equalization(frame)
    gauss_frame = gaussian_blur(frame)

    # Her dönüşüm ayrı bir pencerede gösteriliyor
    cv2.imshow("Original Frame with Detections", frame)
    cv2.imshow("Log Transform", log_frame)
    cv2.imshow("Gamma Correction", gamma_frame)
    cv2.imshow("Histogram Equalization", hist_eq_frame)
    cv2.imshow("Gaussian Blur", gauss_frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
