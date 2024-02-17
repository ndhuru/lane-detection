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
    cv.rectangle(image, (roi_tl_x, roi_tl_y), (roi_br_x, roi_br_y), (255, 0, 0), 5)

    # Creates the hough lines used for the line detection
    lines = cv.HoughLinesP(mask, 1, np.pi / 180, threshold=10, minLineLength=30, maxLineGap=5)

    return lines


def drawParallelLines(image, lines):
    # check if there are any lines detected
    if lines is not None:
        # loop over the lines
        for line in lines:
            # get the coordinates of the line endpoints
            x1, y1, x2, y2 = line[0]
            # draw the line on the original image
            cv.line(image, (x1, y1), (x2, y2), (135, 206, 235), 10)  # Changed color to sky-blue


def drawCenterLine(image, lines):
    # Prevents program from crashing if no lines detected
    if lines is not None:
        # Variables needed to find the centerline
        slope_arr = []
        lines_list = []
        for line in lines:
            # Creates array of lines
            x1, y1, x2, y2 = line[0]
            lines_list.append(line[0])

            # Calculates the slopes of the lines
            slope = 0
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            slope_arr.append(slope)

        # Loops through the slope array to calculate the centerline
        for i in range(len(slope_arr)):
            for j in range(len(slope_arr)):
                x1, y1, x2, y2 = lines_list[i]
                x3, y3, x4, y4 = lines_list[j]
                # Calculates and displays the centerline
                cv.line(image, ((x1 + x3) // 2, (y1 + y3) // 2), ((x2 + x4) // 2, (y2 + y4) // 2), (51, 51, 51), 10)


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
            drawParallelLines(frame, lines)
            # Draw centerline
            drawCenterLine(frame, lines)
            # Display processed frame
            showVideo(frame)

        # Destroys the program when exiting
        cv.destroyAllWindows()

    # Removes the error message when you stop the program
    except Exception as e:
        print("Quitting the program")
        print(e)
    finally:
        exit()


# Runs the program
if __name__ == "__main__":
    main()



# sources:
# https://geeksforgeeks.org/python-play-a-video-using-opencv/
# https://github.com/adityagandhamal/road-lane-detection/blob/master/detection_on_vid.py
# https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/
# https://www.geeksforgeeks.org/program-find-slope-line/
# https://www.geeksforgeeks.org/python-nested-loops/
# https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
# https://hackthedeveloper.com/line-detection-opencv-python/
