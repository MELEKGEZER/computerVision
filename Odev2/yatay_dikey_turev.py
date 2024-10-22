

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Yatay türev için filtre [-1, 1]
horizontal_filter = np.array([[-1, 1]])

# Dikey türev için filtre [-1, 1]^T
vertical_filter = np.array([[-1], [1]])

# Fonksiyon: Gri tonlamaya çevirme ve filtreleme
def apply_filters(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Yatay ve dikey türevler
    horizontal_derivative = cv2.filter2D(gray_image, -1, horizontal_filter)
    vertical_derivative = cv2.filter2D(gray_image, -1, vertical_filter)
    
    return horizontal_derivative, vertical_derivative

# Mavi daire ve kare oluşturma
def ciz_mavi_daire():
    image = np.ones((200, 200, 3), dtype="uint8") * 255
    center = (100, 100)
    radius = 50
    color = (255, 0, 0)
    cv2.circle(image, center, radius, color, -1)
    return image

def ciz_mavi_kare():
    image = np.ones((200, 200, 3), dtype="uint8") * 255
    top_left = (75, 75)
    bottom_right = (125, 125)
    color = (255, 0, 0)
    cv2.rectangle(image, top_left, bottom_right, color, -1)
    return image

# Mavi daire ve kareyi çizme
mavi_daire = ciz_mavi_daire()
mavi_kare = ciz_mavi_kare()

# Daire ve kare için türev filtrelerini uygulama
daire_yatay, daire_dikey = apply_filters(mavi_daire)
kare_yatay, kare_dikey = apply_filters(mavi_kare)

# Sonuçları görselleştirme
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Daire türevleri
axs[0][0].imshow(daire_yatay, cmap='gray')
axs[0][0].set_title('Daire Yatay Türev')
axs[0][0].axis('off')

axs[0][1].imshow(daire_dikey, cmap='gray')
axs[0][1].set_title('Daire Dikey Türev')
axs[0][1].axis('off')

# Kare türevleri
axs[1][0].imshow(kare_yatay, cmap='gray')
axs[1][0].set_title('Kare Yatay Türev')
axs[1][0].axis('off')

axs[1][1].imshow(kare_dikey, cmap='gray')
axs[1][1].set_title('Kare Dikey Türev')
axs[1][1].axis('off')

plt.show()
