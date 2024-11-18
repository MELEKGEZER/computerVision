import cv2
import numpy as np

# 1. Adım: Web Kamerasını Aç
cap = cv2.VideoCapture(0)

while True:
    # 2. Adım: Kameradan Görüntü Al
    ret, frame = cap.read()
    if not ret:
        print("Web kamerasından görüntü alınamadı.")
        break

    # 3. Adım: Görüntüyü Gri Tonlamaya Çevir
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Adım: Ortalama Filtresi Uygula
    mean_filtered = cv2.blur(gray_frame, (5, 5))

    # 5. Adım: Laplace Filtresi Uygula
    laplace_filtered = cv2.Laplacian(mean_filtered, cv2.CV_64F)
    laplace_filtered = np.uint8(np.absolute(laplace_filtered))
 
    # 6. Adım: Orijinal ve Filtrelenmiş Görüntüleri Yanyana Birleştir
    combined = np.hstack((gray_frame, mean_filtered, laplace_filtered))

    # 7. Adım: Sonucu Ekrana Göster
    cv2.imshow('Webcam - Orijinal - Ortalama Filtre - Laplace Filtre', combined)

    # 8. Adım: ESC'ye Basınca Çık
    if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşu
        break

# 9. Adım: Kaynakları Serbest Bırak ve Pencereleri Kapat
cap.release()
cv2.destroyAllWindows()
