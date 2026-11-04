import cv2
import numpy as np
from pathlib import Path

base_path = Path(__file__).resolve().parent                             # Alamat folder saat ini
image_path = base_path.parent / "Resource 1" / "coin_1.jpg"             # Alamat file gambar

image_bgr = cv2.imread(str(image_path))
image_bgr_h, image_bgr_w, _ = image_bgr.shape
print(f"image height: {image_bgr_h}, image width: {image_bgr_w}")

# y, x = 0 , 0                                                          # Koordinat pixel yang ingin diambil nilainya

# (blue, green, red) = image_bgr[y, x]   
# print(f"Intensitas B: {blue} G: {green} R: {red}")                    # Output nilai intensitas warna pada pixel (0,0)

# color_1 = (255, 255, 255)                                             # Manipulasi pixel dengan mengganti warna pixel (0,0) menjadi putih
# image_bgr[0:50, 0:50] = color_1

image_gry = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
_, image_thr = cv2.threshold(image_gry, 150, 255, cv2.THRESH_TOZERO)    # Pengubahan citra grayscale menjadi citra biner dengan threshold 150, nilai di bawah threshold menjadi 0, di atas threshold tetap sama

kernel_dilation = np.ones((5, 5), np.uint8)                             # Dilation morphology dengan kernel 5x5                             # Erosion morphology dengan kernel 3x3
image_dilated = cv2.dilate(image_thr, kernel_dilation, 2)     

kernel_erosion = np.ones((5, 5), np.uint8)                              # Dilation morphology dengan kernel 5x5                             # Erosion morphology dengan kernel 3x3
image_erosion = cv2.erode(image_thr, kernel_erosion, 2)    

image_edge = cv2.Canny(image_thr, 30, 50)                               # Penggunaan metode Canny untuk mendeteksi tepi pada citra biner hasil thresholding

contours, _ = cv2.findContours(image_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Mendeteksi kontur pada citra hasil edge detection dengan metode Canny
image_copy = image_bgr.copy()                                                          # Membuat salinan citra asli untuk menggambar kontur di atasnya
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)                # Menggambar kontur pada citra asli dengan warna hijau dan ketebalan 2 piksel


cv2.imshow("Image raw", image_bgr)              # Tampilkan citra asli
cv2.imshow("Image gray", image_gry)             # Tampilkan citra grayscale
cv2.imshow("Image thres", image_thr)            # Tampilkan citra biner hasil thresholding
cv2.imshow("Image eroded", image_erosion)       # Tampilkan citra hasil erosi
cv2.imshow("Image edge", image_edge)            # Tampilkan citra hasil edge detection
cv2.imshow("Image contours", image_copy)        # Tampilkan citra asli dengan kontur yang digambar di atasnya

cv2.waitKey(0)
cv2.destroyAllWindows()