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
import numpy as np

def punto(landmarks,i,w,h):
    """
    Toma el conjunto de landmarks y el número de landmark deseado y devuelve las coordenadas de ese punto
    
    :param landmarks: Conjunto de landmarks con los que se trata
    :param i: Número del landmark deseado
    :param w: Anchura de la imagen en píxeles
    :param h: Altura de la imagen en píxeles
    """

    return np.array([w*landmarks[i].x,h*landmarks[i].y])


def dist_puntos(p1,p2):
    """
    Devueleve la distancia euclidiana de dos puntos en una imagen
    
    :param p1: Punto 1
    :param p2: Punto 2
    """
    return np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

def pestaneo(p_sup, p_inf):
    """
    Verifica si los ojos están cerrados computando la distancia entre el párpado superior derecho y el inferior derecho.
    Se ha elegido 20 como threshold para definir un ojo cerrado o abierto tras varias pruebas con imágenes propias.
    
    :param p_sup: Pixel representando el párpado superior derecho
    :param p_inf: Pixel representando el párpado superior derecho
    """
    dist_ojos = dist_puntos(p_sup,p_inf)
    print(dist_ojos)
    if dist_ojos < 10:
        return True
    else:
        return False
    
def inclinacion_cabeza(o_izq,o_der):
    """
    Devuelve la inclinación de la cabeza en acorde a la inclinación de la recta que une el landmark de la oreja derecha 
    con el de la oreja izquierda
    
    :param o_izq: Pixel representando la oreja izquierda
    :param o_der: Pixel representando la oreja derecha
    """
    cambio_x = o_der[0] - o_izq[0]
    cambio_y = o_der[1] - o_izq[1]
    return cambio_y/cambio_x

def movimiento(inclinacion):
    if inclinacion > 0.5:
        return "derecha"
    elif inclinacion < -0.5:
        return "izquierda"
    else:
        return "Nada"
    
cap = cv2.VideoCapture(0)
cara = mediapipe.solutions.face_mesh.FaceMesh()

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break



    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)   #mediapipe trabaja con formate rgb
    multiface_landmarks = cara.process(frame)
    landmarks = multiface_landmarks.multi_face_landmarks[0].landmark

    w,h,_ = frame.shape
    p_sup = punto(landmarks,386,w,h)   #386  =  landmark del parpado superior derecho
    p_inf = punto(landmarks,374,w,h)   #374  =  landmark del parpado inferior derecho
    o_der = punto(landmarks,454,w,h)     #454  =  landmark de la oreja derecha
    o_izq = punto(landmarks,234,w,h)     #234  =  landmark de la oreja izquierda

    pstn = pestaneo(p_sup,p_inf)
    inclinacion = inclinacion_cabeza(o_izq,o_der)
    mvmnt = movimiento(inclinacion)
    print(pstn)
    if pstn == True:
        print("Pestañeó")
    if mvmnt != "Nada":
        print(f"Movió hacia la {mvmnt}")

