import cv2
import numpy as np

def show_image(img):
    cv2.imshow("Imagen",img)
    cv2.waitKey()
    cv2.destroyAllWindows()
 
def write_image(name, img):
    cv2.imwrite(name, img)

def get_chessboard_points(chessboard_shape,
                          dx: float,
                          dy: float) -> np.ndarray:
    """
    Build the 3D object points for a planar chessboard lying on Z = 0.
    """
    cols, rows = chessboard_shape
    grid_xy = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = np.zeros((grid_xy.shape[0], 3), dtype=np.float32)
    objp[:, 0] = grid_xy[:, 0] * dx
    objp[:, 1] = grid_xy[:, 1] * dy
    return objp