import cv2
import numpy as np

# Load image
image = cv2.imread("image.jpg")
height, width, _ = image.shape

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# Apply thresholding
_, thresh = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    #Ignore borders
    if 0 <= x <= int(0.1 * width) or int(0.9 * width) <= x <= width:
        continue
    if 0 <= y <= int(0.1 * height) or int(0.9 * height) <= y <= height:
        continue
    #Ignore small boxes (noise)
    if w < 25 or h < 25:
        continue
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    print("Location is", x, y, "\nSize is", w, h)

# Show result
cv2.imshow("Detected IR LED", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
