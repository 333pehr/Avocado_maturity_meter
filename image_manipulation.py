import cv2
import numpy as np

# Load the mask and original image
import cv2
import numpy as np

import colorspacious as cs

def rgb_to_lab(rgb):
    lab = cs.cspace_convert(rgb, "sRGB255", "CIELab")
    return lab

# Load the mask and original image
mask = cv2.imread('mask2.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('image2.jpg')

# Verify that the mask and image are loaded correctly
if mask is None or image is None:
    print("Error loading mask or image")
    exit()

# Ensure the mask is binary
_, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

# Mask the original RGB image
masked_rgb_image = cv2.bitwise_and(image, image, mask=binary_mask)

# Get the RGB values of the masked region
rgb_values = image[binary_mask == 255]

# Split the R, G, B channels
R = rgb_values[:, 2]  # OpenCV loads images in BGR format by default
G = rgb_values[:, 1]
B = rgb_values[:, 0]

# Calculate the average R, G, B values
average_R = np.mean(R)
average_G = np.mean(G)
average_B = np.mean(B)

print(f'Average RGB values: R = {average_R}, G = {average_G}, B = {average_B}')

# Convert the average RGB values to CIELab
average_rgb = np.array([average_R, average_G, average_B], dtype=np.uint8)
average_lab = rgb_to_lab(average_rgb)

print(f'Average CIELab values: L = {average_lab[0]}, a = {average_lab[1]}, b = {average_lab[2]}')

# Display the masked RGB image for verification
cv2.imshow('Masked RGB Image', masked_rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
