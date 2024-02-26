import cv2 as cv
import numpy as np

# Define the height and width of the mask
mask_height = 300
mask_width = 300

def processImage(image):
    # Applies gaussian blur, median blur, and canny edge detection on the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_scale = cv.GaussianBlur(gray, (15, 15), 1)
    median_blur = cv.medianBlur(gray_scale, 5)
    canny_image = cv.Canny(median_blur, 100, 20)

    # Creates a black image with the same size as the original image
    mask = np.zeros_like(gray)

    # Calculates the coordinates of the top-left and bottom-right corners of the mask
    height, width = mask.shape
    x1 = (width - mask_width) // 2
    y1 = (height - mask_height) // 2
    x2 = x1 + mask_width
    y2 = y1 + mask_height

    # Fills the mask with white color inside the square region
    mask[y1:y2, x1:x2] = 255

    # Applies the mask to the canny image
    canny_image = cv.bitwise_and(canny_image, mask)

    # Detects the contours
    contours, hierarchy = cv.findContours(canny_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Prevents program from crashing if no contours detected
    if len(contours) > 0:
        # Draws contours
        cv.drawContours(image, contours, -1, (210, 130, 250), 5)

        # Finds minimum length of contours
        min_length = min(len(cnt) for cnt in contours)

        # Calculates the average of the contour points
        midpoint_x_arr = np.mean([contour[:, 0, :][:min_length][:, 0] for contour in contours], axis=0).astype(int)
        midpoint_y_arr = np.mean([contour[:, 0, :][:min_length][:, 1] for contour in contours], axis=0).astype(int)

        # Displays the centerline
        for i in range(len(midpoint_x_arr) - 1):
            cv.line(image, (midpoint_x_arr[i], midpoint_y_arr[i]), (midpoint_x_arr[i + 1], midpoint_y_arr[i + 1]),
                    (0, 0, 255), 5)

    # Draws a pink outline around the mask
    cv.rectangle(image, (x1, y1), (x2, y2), (200, 144, 255), 2) # pink color is (255, 0, 255) in BGR format

def displayImage(image):
    # Displays the processed frame
    cv.imshow("Detected Lines", image)
    cv.waitKey(1)

if __name__ == "__main__":
    try:
        # Variable needed for displaying the video
        videoIsPlaying = True

        # Starts the video capture
        video = cv.VideoCapture(0)

        # While the video is playing, read the frame, process it & display it
        while videoIsPlaying:
            videoIsPlaying, frame = video.read()
            processImage(frame)
            displayImage(frame)

        # Destroys the program when exiting
        cv.destroyAllWindows()

    except Exception as e:
        print("Quitting the program:", e)
