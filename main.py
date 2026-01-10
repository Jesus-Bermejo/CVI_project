import security_system
import juego

### VARIABLES PARA MODIFICAR
calibration = "calibration_jesus.npz"
#calibration = False

user_password = "COMPUTERVISION2025"

debug = False
    

if __name__ == "__main__":
    succes = security_system.security_system(camera_index=0, calibration=calibration, user_password=user_password, debug = debug)
    if succes:
        juego.play_tetris_cv(camera_index=0)