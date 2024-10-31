import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image_path = 'D:/PyThon01/TGMT/xp-15.jpg'  # Đường dẫn đến ảnh của bạn
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Kiểm tra xem ảnh có được tải thành công không
if image is None:
    print("Không thể tải ảnh.")
    exit()

# Áp dụng toán tử Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Dò biên theo trục x
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Dò biên theo trục y
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)  # Tính độ lớn của gradient

# Áp dụng Laplacian of Gaussian (LoG)
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)  # Làm mượt ảnh
laplacian = cv2.Laplacian(gaussian_blur, cv2.CV_64F, ksize=3)  # Áp dụng Laplacian

# Hiển thị kết quả
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc'), plt.axis('off')

plt.subplot(1, 3, 2), plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Dò biên Sobel'), plt.axis('off')

plt.subplot(1, 3, 3), plt.imshow(laplacian, cmap='gray')
plt.title('Dò biên LoG'), plt.axis('off')

plt.tight_layout()
plt.show()
