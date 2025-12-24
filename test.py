import cv2
import numpy as np
from security_functions import *

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

            gray = cv2.cvtColor(uframe, cv2.COLOR_BGR2GRAY)
            #gray = np.where(gray<220, 0, gray).astype(np.uint8)
            blur = cv2.GaussianBlur(gray, (7,7), 1)
            edges = cv2.Canny(blur, 50, 200)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            box = detect_paper(uframe)

            if box:
                x, y, w, h = box
                cv2.rectangle(uframe, (x, y), (x + w, y + h), (0,255,0), 2)
                # Aquí empezamos a aislar la letra

                square_roi = uframe[y:y+h, x:x+w]
                gray = cv2.cvtColor(square_roi, cv2.COLOR_BGR2GRAY)
                margin = int(0.1 * min(w, h))  # 10% del tamaño
                inner = gray[margin:h-margin, margin:w-margin]
                binary = np.where(inner<100, 0, 255).astype(np.uint8)
                # esto podría ser opcional, pero mejora las predicciones
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(
                    255 - binary,  # invertimos SOLO para encontrar contornos
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if not contours:
                    continue

                char_cnt = max(contours, key=cv2.contourArea)
                x_c, y_c, w_c, h_c = cv2.boundingRect(char_cnt)
                abs_x = x + margin + x_c
                abs_y = y + margin + y_c

                cv2.rectangle(uframe, (abs_x, abs_y), (abs_x + w_c, abs_y + h_c), (0,0,255), 2)

                char_roi = binary[y_c:y_c+h_c, x_c:x_c+w_c]
                char_roi = 255 - char_roi
                char_28 = cv2.resize(char_roi, (28,28), interpolation=cv2.INTER_AREA)   # resize para la NN

                cv2.imshow("28", char_28)
                cv2.imshow("roi", char_roi)
                cv2.imshow("bin", binary)


            else:
                cv2.putText(uframe, "No paper detected", (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Frame", uframe)
            #cv2.imshow("gray", gray)
            #cv2.imshow("blur", blur)
            #cv2.imshow("edges", edges)
            #cv2.imshow("contours", contours)

            # Wait 1 ms for a key; exit with 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # If the built-in webcam is not at index 0, change the first argument: main(1)
    main(camera_index=0, width=1280, height=720)
