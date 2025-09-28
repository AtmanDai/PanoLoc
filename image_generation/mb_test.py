import cv2
import numpy as np

image_path="datasets/MGL/tokyo/images/101572959414079_front.jpg"
image = cv2.imread(image_path)

factor = 0.25
# Convert to HSV color space to manipulate brightness
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv = hsv.astype(np.float32)
# Reduce brightness channel (V)
hsv[:, :, 2] = hsv[:, :, 2] * factor
# Ensure values are in valid range
hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
hsv = hsv.astype(np.uint8)
# Convert back to BGR
overex = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imwrite("image_generation/mgl/101572959414079_front_ux.jpg", overex)