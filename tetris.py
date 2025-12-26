"""
Autor: Alejandro Boix 
Fecha: 24/12/2025
Para qué disfrutar de la Navidad en familia si podemos estar haciendo un proyecto de vision por ordenador?

"""


"""
Se utilizará la clase FaceMesh de la librería solutions de mediapipe.
Esta clase utiliza diferentes parámetros como static_image_mode, max_num_faces, refine_landmarks, min_detecton_confidence
y min_tracking confidence. Se utilizarán los valores predeterminados por la clase.
En un tiempo futuro, y depende de los resultados, se valorará la posibilidad de cambiar el estado base de refine_landmarks
de False a True.
"""

import cv2
import mediapipe

imagen_cara = cv2.imread("camera_captures/frame_20251225_205223.jpg")
cara = mediapipe.solutions.face_mesh.FaceMesh()

multiface_landmarks = cara.process(imagen_cara)
print(multiface_landmarks.multi_face_landmarks.shape)