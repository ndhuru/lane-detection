import cv2
import numpy as np

def apply_square_detection(frame, max_lines=None):
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # apply gaussianblur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    # use canny edge detector to find edges in the frame> >
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow('canny edges', edges)  # display the edges for visualization
    # use houghlinesp to detect lines in the frame
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=100)
    # draw lines
    line_frame = frame.copy()  # create a copy of the original frame to draw lines on
    if lines is not None:
        if max_lines is not None:
            lines = lines[:max_lines]  # limit the number of detected lines if specified
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)  # calculate the angle of the line
            cv2.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)  # draw the line on the frame
            angles.append(angle)
        # calculate the average angle of detected lines
        avg_angle = np.mean(angles)
        # calculate the midpoint between two parallel lines
        if len(lines) >= 2:
            x1_avg, y1_avg, x2_avg, y2_avg = np.mean(lines[:, 0, :], axis=0).astype(int)
            cv2.line(line_frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (255, 0, 0), 10)
    # merge the original frame with the frame containing drawn lines
    overlay = cv2.addWeighted(frame, 0.8, line_frame, 1, 0)
    return overlay
# create a videocapture object to capture video from the camera
cap = cv2.VideoCapture(0)
while True:
    # read a frame from the camera
    ret, frame = cap.read()
    # apply the square detection function to the frame, specifying max_lines
    overlay = apply_square_detection(frame, max_lines=4)
    # display the resulting frame with the overlay
    cv2.imshow('video overlay', overlay)
    # check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# release the videocapture object and close all windows
cap.release()
cv2.destroyAllWindows()

# sources:
# opencv shape detection:
#    this tutorial explains how to use contour approximation and cv2.approxpolydp to identify shapes in an image.
#    it also shows how to use cv2.arclength and cv2.drawcontours to draw the shapes on the image.
#    citation: rosebrock, a. (2016, january 11). opencv shape detection. pyimagesearch.
#    https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
# square detection in image:
#    this question on stack overflow discusses how to detect squares in an image using cv2.canny,
#    cv2.gaussianblur, cv2.houghlinesp, and cv2.iscontourconvex. it also shows how to calculate the average angle of the
#    lines and draw a vertical or middle line based on it.
#    citation: user366312. (2010, september 29). square detection in image. stack overflow.
#    https://stackoverflow.com/questions/3823621/square-detection-in-image
# how to detect a rectangle and square in an image using opencv python?:
#    this article on tutorialspoint demonstrates how to use cv2.cvtcolor, cv2.threshold, cv2.findcontours, and cv2.boundingrect
#    to detect and extract rectangles and squares in an image.
#    citation: tutorialspoint. (n.d.). how to detect a rectangle and square in an image using opencv python?
#    tutorialspoint. https://www.tutorialspoint.com/how-to-detect-a-rectangle-and-square-in-an-image-using-opencv-python
