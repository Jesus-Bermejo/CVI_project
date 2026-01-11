import cv2
from typing import List
import numpy as np
import copy
from funciones_calibracion import *
import os

print(os.getcwd())
base = os.path.dirname(__file__)
imgs_parent_path = os.path.join(base, "calibration_images")

imgs_path = []
# Asumimos que las imágenes para la calibración se llaman: "1.jpg", "2.jpg", ...
for i in range(1,24):   # ajustar para calibrar
    imgs_path.append(os.path.join(imgs_parent_path, f"{i}.jpg"))

imgs = []
for path in imgs_path:
    imgs.append(cv2.imread(path))

good_imgs = []
all_corners = []
for img in imgs:
    corners = cv2.findChessboardCorners(img,(9,6))
    if corners[0]:
        all_corners.append(corners)
        good_imgs.append(img)
    #print(corners[0])    # debug

corners = all_corners
imgs = good_imgs

corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)
 
# To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.
imgs_gray = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in imgs]
 
 
corners_refined = [cv2.cornerSubPix(i, cor[1], (9, 6), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

corners_drawn_images = []
for i, cor in zip(imgs, corners_refined):
    corners_drawn_images.append(cv2.drawChessboardCorners(i, (9,6), cor, True))

results_parent_path = os.path.join(base, "calibration_results")
imgs_path = [os.path.join(results_parent_path, f"{i+1}.jpg") for i in range(len(imgs))]
for i in range(len(corners_drawn_images)):
    write_image(imgs_path[i], corners_drawn_images[i])

objp_template = get_chessboard_points((9,6), 25, 25)
print("Número de puntos 3D del tablero:", objp_template.shape[0])

points_3d: List[np.ndarray] = []
points_2d: List[np.ndarray] = []

for refined in corners_refined:
    if isinstance(refined, np.ndarray) and refined.size > 0:
        points_3d.append(objp_template)
        points_2d.append(refined.astype(np.float32))

print(f"Número de frames utilizados para la calibración: {len(points_2d)}")

image_size = imgs[0].shape[:2]
print("Tamaño de imagen:", image_size)

rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    points_3d,
    points_2d,
    image_size,
    None,
    None
)

print("Matriz intrínseca (cameraMatrix):\n", intrinsics)
print("Coeficientes de distorsión (distCoeffs):\n", dist_coeffs.ravel())
print("Error RMS de reproyección:", rms)

# no tiene sentido hacer la de extrínsecos para nuetsro caso de uso

calibration_name = "calibration.npz"   # el .npz es importante
np.savez(
    calibration_name,
    cameraMatrix=intrinsics,
    distCoeffs=dist_coeffs
)