import cv2 as cv
import numpy as np


def lineDetection(image):
    # Applies gaussian blur, median blur, and canny edge detection on the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_scale = cv.GaussianBlur(gray, (15, 15), 0)
    median_blur = cv.medianBlur(gray_scale, 5)
    canny_image = cv.Canny(median_blur, 100, 20)

    # Creates a mask around desired area
    roi = np.zeros(image.shape[:2], dtype="uint8")
    # Calculate the center coordinates of the screen
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    # Calculate the width and height of the ROI
    roi_width = 350
    roi_height = 350
    # Calculate the top-left and bottom-right coordinates of the ROI
    roi_tl_x = center_x - roi_width // 2
    roi_tl_y = center_y - roi_height // 2
    roi_br_x = center_x + roi_width // 2
    roi_br_y = center_y + roi_height // 2
    cv.rectangle(roi, (roi_tl_x, roi_tl_y), (roi_br_x, roi_br_y), 1, -1)
    mask = cv.bitwise_and(canny_image, canny_image, mask=roi)
    cv.rectangle(image, (roi_tl_x, roi_tl_y), (roi_br_x, roi_br_y), (255, 0, 255), 5)

    # Creates the hough lines used for the line detection
    lines = cv.HoughLinesP(mask, 1, np.pi / 180, threshold=45, minLineLength=10, maxLineGap=5)

    return lines


def drawLines(image, lines):
    # check if there are any lines detected
    if lines is not None:
        # loop over the lines
        for line in lines:
            # get the coordinates of the line endpoints
            x1, y1, x2, y2 = line[0]
            # Generate a set of x values from the smallest x to the largest x
            x = np.linspace(min(x1, x2), max(x1, x2), 1000)
            # Fit a polynomial of degree 2 to the x and y coordinates
            coefficients = np.polyfit([x1, x2], [y1, y2], 2)
            # Generate the y values from the polynomial
            y = np.polyval(coefficients, x)
            # Draw the curve on the original image
            for i in range(len(x) - 1):
                cv.line(image, (int(x[i]), int(y[i])), (int(x[i + 1]), int(y[i + 1])), (255, 206, 235), 10)


def drawCenterLine(image, lines):
    # Prevents program from crashing if no lines detected
    if lines is not None:
        # Variables needed to find the centerline
        x_arr = []
        y_arr = []
        for line in lines:
            # Creates array of lines
            x1, y1, x2, y2 = line[0]
            x_arr.extend([x1, x2])
            y_arr.extend([y1, y2])

        # Fit a polynomial of degree 2 to the x and y coordinates
        coefficients = np.polyfit(x_arr, y_arr, 2)
        # Generate a set of x values from the smallest x to the largest x
        x = np.linspace(min(x_arr), max(x_arr), 1000)
        # Generate the y values from the polynomial
        y = np.polyval(coefficients, x)
        # Draw the curve on the original image
        for i in range(len(x) - 1):
            cv.line(image, (int(x[i]), int(y[i])), (int(x[i + 1]), int(y[i + 1])), (251, 13, 136), 10)


def showVideo(image):
    # Returns the processed frame
    cv.imshow("Detected Lines", image)
    cv.waitKey(1)


def main():
    try:
        videoIsPlaying = True

        # Starts the video capture
        video = cv.VideoCapture(0)

        # While the video is playing, read the frame, process it & display it
        while videoIsPlaying:
            videoIsPlaying, frame = video.read()
            # Perform line detection
            lines = lineDetection(frame)
            # Draw parallel lines
            drawLines(frame, lines)
            # Draw centerline
            drawCenterLine(frame, lines)
            # Display processed frame
            showVideo(frame)

        # Destroys the program when exiting
        cv.destroyAllWindows()

    # Removes the error message when you stop the program
    except Exception as e:
        print("error detected, cant be parsed so program quits")
        print(e)
    finally:
        exit()


# Runs the program
if __name__ == "__main__":
    main()
