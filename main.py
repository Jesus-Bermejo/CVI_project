import security_system
import juego

### VARIABLES PARA MODIFICAR
calibration = "calibration_jesus.npz"
#calibration = False

user_password = "COMPUTERVISION2025"

debug = False
    

if __name__ == "__main__":
    success = security_system.security_system(camera_index=0, calibration=calibration, user_password=user_password, debug = debug)
    if success == 1:
        juego.play_tetris_cv(calibration=calibration, camera_index=0)