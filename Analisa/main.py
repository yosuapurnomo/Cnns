import numpy as np
import cv2
import matplotlib.pyplot as plt

imageSenang = cv2.imread("Image/Senang_2.jpg")
imageNormal = cv2.imread("Image/Normal_1.jpg")

GraySenang = cv2.cvtColor(imageSenang, cv2.COLOR_BGR2GRAY)
GrayNormal = cv2.cvtColor(imageNormal, cv2.COLOR_BGR2GRAY)

# Laplacian Kernel / Deteksi Tepi
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

# Senang_Image = cv2.Laplacian(GraySenang, -1)
# Normal_Image = cv2.Laplacian(GrayNormal, -1)
Senang_Image = cv2.filter2D(GraySenang, -1, kernel)
Normal_Image = cv2.filter2D(GrayNormal, -1, kernel)
# Senang_Image = cv2.Canny(GraySenang, 10, 120)
# Normal_Image = cv2.Canny(GrayNormal, 10, 120)

figure, axes = plt.subplots(2, 2, figsize=(12, 6))

# axes[0][0].imshow(cv2.cvtColor(imageSenang, cv2.COLOR_BGR2RGB))
axes[0][0].imshow(GraySenang, cmap=plt.cm.gray)
axes[0][0].set_title("Image Senang")

axes[0][1].imshow(Senang_Image, cmap=plt.cm.gray)
axes[0][1].set_title("Laplacian Senang")

# axes[1][0].imshow(cv2.cvtColor(imageNormal, cv2.COLOR_BGR2RGB))
axes[1][0].imshow(GrayNormal, cmap=plt.cm.gray)
axes[1][0].set_title("Image Normal")

axes[1][1].imshow(Normal_Image, cmap=plt.cm.gray)
axes[1][1].set_title("Laplacian Normal")

plt.show()
# cv2.imshow("Grayscale Image", gray_image)
# cv2.imshow("Laplacian Image", laplacian_Image)
# cv2.imshow("Result Image", result_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()