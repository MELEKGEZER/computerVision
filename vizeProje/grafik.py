import matplotlib.pyplot as plt
import numpy as np

# Projeden elde edilen örnek veriler
methods = ["Original Frame", "Log Transform", "Gamma Correction", "Histogram Equalization", "Gaussian Blur"]
detection_times = [0.1226, 0.1015, 0.1477, 0.1179, 0.1055]  # Algılama süreleri (saniye)
avg_confidences = [0.49, 0.39, 0.50, 0.51, 0.74]       # Ortalama güven skorları
detected_objects = [1, 1, 1, 1, 1]                     # Algılanan nesne sayısı

# 1. Algılama Süresi Grafiği
plt.figure(figsize=(10, 6))
plt.bar(methods, detection_times, color='skyblue')
plt.xlabel("Methods")
plt.ylabel("Detection Time (s)")
plt.title("Detection Time by Method")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("detection_time.png", dpi=300)
plt.show()

# 2. Ortalama Güven Skoru Grafiği
plt.figure(figsize=(10, 6))
plt.bar(methods, avg_confidences, color='lightgreen')
plt.xlabel("Methods")
plt.ylabel("Average Confidence")
plt.title("Average Confidence by Method")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("average_confidence.png", dpi=300)
plt.show()

# 3. Algılanan Nesne Sayısı Grafiği
plt.figure(figsize=(10, 6))
plt.bar(methods, detected_objects, color='salmon')
plt.xlabel("Methods")
plt.ylabel("Detected Objects")
plt.title("Number of Detected Objects by Method")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("detected_objects.png", dpi=300)
plt.show()
