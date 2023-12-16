import cv2
import argparse

# Pixel detection using opencv -- detects color changes in pixels via webcam or video file

def detect_color_change(video_source=0):
    # Initialize the video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Video source not accessible.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read from video source.")
        return

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frames to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(gray_prev_frame, gray_frame)

            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Threshold for size of change
                    (x, y, w, h) = cv2.boundingRect(contour)
                    center = (x + w//2, y + h//2)  # Calculate center of the contour
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Mark with a red dot

            # Display
            cv2.imshow('Color Change Detection', frame)

            # Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_frame = frame.copy()

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Parse
parser = argparse.ArgumentParser(description="Detect color changes in a video stream.")
parser.add_argument("--video", help="Path to a video file. If not specified, webcam will be used.", default=0)
args = parser.parse_args()

detect_color_change(args.video)
