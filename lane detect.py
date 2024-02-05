import cv2
import numpy as np

# Load the image
image = cv2.imread('IMG_3569.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)

# Perform a dilation and erosion to close gaps in between object edges
dilated = cv2.dilate(edges, None, iterations=2)
eroded = cv2.erode(dilated, None, iterations=1)

# Perform Hough Line Transform
lines = cv2.HoughLinesP(eroded, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green line

# Show the image with lines
cv2.imshow('Image with Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
