# cizim_daire_kare.py

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Fonksiyon: Mavi bir daire çizme
def ciz_mavi_daire():
    image = np.ones((200, 200, 3), dtype="uint8") * 255  # Beyaz arka plan
    center = (100, 100)
    radius = 50
    color = (255, 0, 0)  
    thickness = -1  
    cv2.circle(image, center, radius, color, thickness)
    return image

# Fonksiyon: Mavi bir kare çizme
def ciz_mavi_kare():
    image = np.ones((200, 200, 3), dtype="uint8") * 255  # Beyaz arka plan
    top_left = (75, 75)
    bottom_right = (125, 125)
    color = (255, 0, 0)  
    thickness = -1 
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image


mavi_daire = ciz_mavi_daire()
mavi_kare = ciz_mavi_kare()

# Mavi daireyi ve kareyi görselleştirme
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(mavi_daire, cv2.COLOR_BGR2RGB))
axs[0].set_title('Mavi Daire')
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(mavi_kare, cv2.COLOR_BGR2RGB))
axs[1].set_title('Mavi Kare')
axs[1].axis('off')

plt.show()
