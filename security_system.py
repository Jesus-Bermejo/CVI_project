import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from security_utils import *
import joblib
import time
import string

data = np.load("calibration_jesus.npz")
cameraMatrix = data["cameraMatrix"]
distCoeffs = data["distCoeffs"]

#model = joblib.load("char_mlp.pkl")

model = EMNIST_CNN()
model.load_state_dict(torch.load("emnist_cnn_balanced.pth", map_location="cpu"))
model.eval()

# PARAMS SECURITY
CHANGE_THRESHOLD = 1000
FORGET_TIME = 3.0
CONFIRMATION_TIME = 2.0
CONF_THRESHOLD = 0.8

digits = {i: str(i) for i in range(10)}
letters = {i+10: c for i, c in enumerate(string.ascii_uppercase)}
class_map = {**digits, **letters}



def main(camera_index=0, width=1280, height=720):
    # # Initialize video capture with the specified camera index
    # cap = cv2.VideoCapture(camera_index)
    # if not cap.isOpened():
    #     print(f"Could not open the camera (index {camera_index}).")
    #     return
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Force to use DIRECTSHOW

    attempts = 5
    true_password = "JESUS25"
    password_input = []
    last_char = None
    last_roi_frame = None
    last_seen_time = None
    char_confirmation = False


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

            # uditsorted frame
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
            #gray = np.where(gray<220, 0, gray).astype(np.uint8)   # this was unnecesssarily aggresive
            blur = cv2.GaussianBlur(gray, (7,7), 1)
            edges = cv2.Canny(blur, 50, 200)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            box = detect_paper(uframe)

            current_time = time.time()
            if box:

                x, y, w, h = box
                cv2.rectangle(uframe, (x, y), (x + w, y + h), (0,255,0), 2)

                square_roi = uframe[y:y+h, x:x+w]
                gray = cv2.cvtColor(square_roi, cv2.COLOR_BGR2GRAY)

                margin = int(0.1 * min(w, h))
                inner = gray[margin:h-margin, margin:w-margin]
                binary = np.where(inner < 180, 0, 255).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                inner_x = x + margin
                inner_y = y + margin
                inner_w = w - 2 * margin
                inner_h = h - 2 * margin

                cv2.rectangle(uframe, (inner_x, inner_y), (inner_x + inner_w, inner_y + inner_h), (0, 0, 255), 2)

                char_roi = 255 - binary
                char_28 = cv2.resize(char_roi, (28,28), interpolation=cv2.INTER_AREA)

                #char_28 = np.rot90(char_28, k=1)
                char_28 = np.fliplr(char_28)

                if last_roi_frame is None:
                    diff = np.inf
                else:
                    diff = np.sum(np.abs(char_28 - last_roi_frame))

                last_roi_frame = char_28
                last_seen_time = current_time

                display_text = "Waiting..."
                if diff > CHANGE_THRESHOLD:
                    # CHATGPT Preprocesado para PyTorch: [1, 1, 28, 28]
                    char_tensor = torch.tensor(
                        char_28 / 255.0,
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)

                    with torch.no_grad():
                        logits = model(char_tensor)
                        probs = torch.softmax(logits, dim=1)[0].numpy()

                    pred_class = np.argmax(probs)
                    confidence = probs[pred_class]

                    if confidence < CONF_THRESHOLD:
                        display_text = "Rejected"
                    else:
                        pred_char = class_map.get(pred_class, "?")
                        display_text = f"Detected: {pred_char} ({confidence:.2f})"

                        if pred_char != last_char:
                            char_confirmation = False
                            first_seen_time = current_time
                            last_char = pred_char
                            print("Checking input...")
                        elif current_time - first_seen_time > CONFIRMATION_TIME and not char_confirmation:
                            char_confirmation = True
                            password_input.append(pred_char)
                            print(f"New input detected: {pred_char} -> Current password_input: {''.join(password_input)}")
                            if not true_password.startswith(''.join(password_input)):
                                print("Contraseña incorrecta, inténtalo de nuevo")
                                password_input = []
                                attempts -= 1
                                if attempts == 0:
                                    print("You ran out of attempts")
                                    return 0
                            elif true_password == ''.join(password_input):
                                print("Contrasña correcta")
                                return 1


                cv2.putText(uframe, display_text, (inner_x, inner_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)


                cv2.imshow("bin", binary)
                cv2.imshow("roi", char_roi)
                cv2.imshow("28", char_28)


            else:
                cv2.putText(uframe, "No paper detected", (30,100),   # (horizontal, vertical)
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                if last_seen_time is None:
                    last_seen_time = current_time
                elif current_time - last_seen_time > FORGET_TIME:
                    last_seen_time = None
                    last_char = None
                    last_roi_frame = None
                    char_confirmation = False


            cv2.putText(uframe, f"Password: {''.join(password_input)}", (30,50),   # (horizontal, vertical)
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(uframe, f"Attempts: {attempts}", (460,50),   # (horizontal, vertical)
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
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
