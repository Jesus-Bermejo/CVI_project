import cv2
import numpy as np

data = np.load("calibration_jesus.npz")
cameraMatrix = data["cameraMatrix"]
distCoeffs = data["distCoeffs"]


def main(camera_index=0, width=1280, height=720):
    # # Initialize video capture with the specified camera index
    # cap = cv2.VideoCapture(camera_index)
    # if not cap.isOpened():
    #     print(f"Could not open the camera (index {camera_index}).")
    #     return
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Force to use DIRECTSHOW
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return
    
    window_name = "Live Camera - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # resizable window

    ret, frame = cap.read()
    h, w = frame.shape[:2]

    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix,
        distCoeffs,
        (w, h),
        0,
        (w, h))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received (the camera may have been disconnected).")
                break

            # uditsorted frame (sin distorsión)
            uframe = cv2.undistort(
                frame,
                cameraMatrix,
                distCoeffs,
                None,
                newCameraMatrix)
    
            #x, y, w_roi, h_roi = roi
            #uframe = uframe[y:y+h_roi, x:x+w_roi]
            
            # Optional: flip horizontally (mirror), comment out if not desired
            uframe = cv2.flip(uframe, 1)

            # Display the frame
            frame = cv2.flip(frame, 1)
            combined = np.hstack((frame, uframe))
            cv2.imshow("Original | Undistorted", combined)
            #cv2.imshow(window_name, uframe)

            # Wait 1 ms for a key; exit with 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print(roi)
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # If the built-in webcam is not at index 0, change the first argument: main(1)
    main(camera_index=0, width=1280, height=720)
