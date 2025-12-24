import cv2
import numpy as np

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

def detect_paper(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = np.where(gray<220, 0, gray).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (7,7), 1)
    edges = cv2.Canny(blur, 100, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Contorno más grande
    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 3000:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h

def detect_paper_with_bg_sub(frame, min_area=3000):
    """
    Detecta la hoja usando sustracción de fondo + bounding box.
    """
    fg_mask = bg_subtractor.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, fg_mask

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None, fg_mask

    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h), fg_mask