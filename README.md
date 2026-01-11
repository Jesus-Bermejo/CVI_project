# CVI Project

## Descripción

**CVI Project** combina **visión por ordenador, aprendizaje profundo y control gestual** para crear una experiencia interactiva única:

- **Sistema de seguridad**: Reconocimiento de caracteres EMNIST (0-9, A-Z) para validar contraseñas.  
- **Tetris controlado por gestos**: Una vez autenticado, el usuario juega usando **pestañeos y movimientos de cabeza**.  

El proyecto utiliza **OpenCV, PyTorch, MediaPipe y Tkinter** para ofrecer interacción natural y segura.

---

## Tecnologías principales

- **Python 3.9**  
- **OpenCV** – Procesamiento de imágenes y detección de gestos  
- **PyTorch** – CNN para reconocimiento de caracteres EMNIST  
- **MediaPipe** – Seguimiento de ojos  
- **Tkinter** – Interfaz gráfica del juego  
- **NumPy** – Manipulación de matrices y datos   

---

## Control de Tetris por cámara

| Acción         | Gestos detectados |
|----------------|-----------------|
| Rotar pieza    | Pestañeo fuerte con **ambos ojos** |
| Mover izquierda| Cabeza hacia **izquierda** |
| Mover derecha  | Cabeza hacia **derecha** |
| Salir del juego| Presionar `q` |

> Se recomienda buena iluminación y cámara frontal para mayor precisión.

---

## Funcionalidades destacadas

- **Reconocimiento de caracteres EMNIST**: Detecta números y letras manuscritas para validar contraseñas de acceso.
- **Gestos naturales**: Control del juego Tetris mediante pestañeos y movimientos de cabeza, sin necesidad de teclado.
- **Sistema de seguridad por intentos limitados**: El sistema se bloquea tras 5 intentos fallidos consecutivos.
- **Interfaz gráfica fluida**: El juego se actualiza dinámicamente según el nivel y la puntuación del usuario.

---

## Referencias

- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [MediaPipe](https://mediapipe.dev/)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

## Autores

- **Jesús Bermejo Úbeda**
- **Alejandro Boix Bayarri**

Para la asignatura de **Visión por Ordenador I** del curso 2025-2026