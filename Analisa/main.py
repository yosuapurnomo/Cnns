import numpy as np
import cv2
import matplotlib.pyplot as plt

imageSenang = cv2.imread("Image/Senang_6.jpg")
imageSenang_1 = cv2.imread("Image/Senang_4.jpg")
imageNormal = cv2.imread("Image/Normal_1.jpg")

GraySenang = cv2.cvtColor(imageSenang, cv2.COLOR_BGR2GRAY)
GraySenang_1 = cv2.cvtColor(imageSenang_1, cv2.COLOR_BGR2GRAY)
GrayNormal = cv2.cvtColor(imageNormal, cv2.COLOR_BGR2GRAY)

# Laplacian Kernel / Deteksi Tepi
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

Senang_Image = cv2.Canny(GraySenang, 100, 200)
Senang_Image_1 = cv2.Canny(GraySenang_1, 100, 200)
Normal_Image = cv2.Canny(GrayNormal, 100, 200)

figure, axes = plt.subplots(2, 3, figsize=(12, 6))

# axes[0][0].imshow(cv2.cvtColor(imageSenang, cv2.COLOR_BGR2RGB))
axes[1][0].imshow(GraySenang, cmap=plt.cm.gray)
axes[1][0].set_title("Senang Image")

axes[0][0].imshow(Senang_Image, cmap=plt.cm.gray)
axes[0][0].set_title("Senang Image")

# axes[1][0].imshow(cv2.cvtColor(imageNormal, cv2.COLOR_BGR2RGB))

axes[1][1].imshow(GraySenang_1, cmap=plt.cm.gray)
axes[1][1].set_title("Senang Image 1")

axes[0][1].imshow(Senang_Image_1, cmap=plt.cm.gray)
axes[0][1].set_title("Senang Image 1")

axes[1][2].imshow(GrayNormal, cmap=plt.cm.gray)
axes[1][2].set_title("Image Normal")

axes[0][2].imshow(Normal_Image, cmap=plt.cm.gray)
axes[0][2].set_title("Image Normal")

print(Senang_Image, "\n")
print(Senang_Image_1, "\n")
print(Normal_Image, "\n")

plt.show()
