import cv2
import numpy as np

# 1. Adım: Görüntüyü Oku
image = cv2.imread('girdi_gorsel.png', cv2.IMREAD_GRAYSCALE)

# 2. Adım: Ortalama Filtresi (Blur)
mean_filtered = cv2.blur(image, (5, 5))

# 3. Adım: Laplace Filtresi
laplace_filtered = cv2.Laplacian(mean_filtered, cv2.CV_64F)
laplace_filtered = np.uint8(np.absolute(laplace_filtered))

# 4. Adım: Orijinal ve Filtrelenmiş Görüntüleri Yanyana Gösterme
combined = np.hstack((image, mean_filtered, laplace_filtered))

# 5. Adım: Sonucu Ekrana Bas
cv2.imshow('Orijinal - Ortalama Filtre - Laplace Filtre', combined)

# 6. Adım: ESC'ye Basınca Pencereyi Kapat
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşu
        break

cv2.destroyAllWindows()
