import cv2
import mediapipe as mp
import numpy as np
from time import time

# --- Configuración landmarks ojos ---
EYE_R = [33, 160, 158, 133, 153, 144]  # derecho
EYE_L = [362, 385, 387, 263, 373, 380]  # izquierdo
EAR_THRESH = 0.21  # umbral de parpadeo
BUFFER_LEN = 5
COOLDOWN = 0.4  # cooldown entre movimientos/rotaciones

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    coords = [np.array([w*landmarks[i].x, h*landmarks[i].y]) for i in eye_indices]
    vertical = np.linalg.norm(coords[1]-coords[5]) + np.linalg.norm(coords[2]-coords[4])
    horizontal = np.linalg.norm(coords[0]-coords[3])
    return vertical / horizontal

def inclinacion_cabeza(o_izq, o_der):
    cambio_x = o_der[0] - o_izq[0]
    cambio_y = o_der[1] - o_izq[1]
    return cambio_y / cambio_x

def movimiento(inclinacion):
    if inclinacion > 0.4:
        return "izquierda"
    elif inclinacion < -0.4:
        return "derecha"
    else:
        return "Nada"

def computer_vision(app, calibration=False, camera_index=0):

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open the camera (index {camera_index}).")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    h, w = frame.shape[:2]

    try:
        if calibration is not False:
            data = np.load(calibration)
            cameraMatrix = data["cameraMatrix"]
            distCoeffs = data["distCoeffs"]
            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
                cameraMatrix,
                distCoeffs,
                (w, h),
                0,
                (w, h))
    except Exception as e:
        print(e)
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    blink_buffer_r = []
    blink_buffer_l = []
    blink_cooldown = time()
    head_cooldown = time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if calibration is not False:
                frame_rgb = cv2.undistort(
                    frame_rgb,
                    cameraMatrix,
                    distCoeffs,
                    None,
                    newCameraMatrix)
        results = face_mesh.process(frame_rgb)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            EAR_r = eye_aspect_ratio(landmarks, EYE_R, w, h)
            EAR_l = eye_aspect_ratio(landmarks, EYE_L, w, h)

            blink_buffer_r.append(EAR_r < EAR_THRESH)
            blink_buffer_l.append(EAR_l < EAR_THRESH)
            if len(blink_buffer_r) > BUFFER_LEN:
                blink_buffer_r.pop(0)
            if len(blink_buffer_l) > BUFFER_LEN:
                blink_buffer_l.pop(0)

            blink_both = sum(blink_buffer_r[-3:]) == 3 and sum(blink_buffer_l[-3:]) == 3

            # Orejas
            o_der = np.array([w*landmarks[454].x, h*landmarks[454].y])
            o_izq = np.array([w*landmarks[234].x, h*landmarks[234].y])
            inclin = inclinacion_cabeza(o_izq, o_der)
            mvmnt = movimiento(inclin)

            t = time()
            # Movimientos cabeza
            if mvmnt != "Nada" and (t - head_cooldown) > COOLDOWN:
                head_cooldown = t
                app.cola_de_eventos.put(mvmnt)

            # Parpadeo(ambos ojos)
            elif blink_both and (t - blink_cooldown) > COOLDOWN and (t - head_cooldown) > COOLDOWN:
                blink_cooldown = t
                blink_buffer_r.clear()
                blink_buffer_l.clear()
                app.cola_de_eventos.put("PSTN")


            for idx in EYE_R + EYE_L + [454, 234]:
                x, y = int(w*landmarks[idx].x), int(h*landmarks[idx].y)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Tracker
            xs = [int(w*lm.x) for lm in landmarks]
            ys = [int(h*lm.y) for lm in landmarks]
            cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (255, 0, 0), 2)

        frame = cv2.flip(frame, 1)
        cv2.imshow("Tetris Tracker", frame)

        # Salida con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
