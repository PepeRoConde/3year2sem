from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from time import sleep

IP = 'localhost'
SPEED = 15
CENTER_X = 250
X_OFFSET = 30
CENTER_Y = 300
Y_OFFSET = 65
PAN_UPPER_LIMIT = 160
PAN_LOWER_LIMIT = -160
TILT_UPPER_LIMIT = 106
TILT_LOWER_LIMIT = 4
NEAR_DISTANCE = 20
last_qr = None
last_qr_sighting = 0
pan, tilt = 0, 90
 

def center_qr(qr):
    """
    Ajusta los movimientos pan (horizontal) y tilt (vertical) para centrar el QR en la cámara.
    
    Args:
        qr (object): Objeto QR detectado con atributos x e y de posición.
    """
    global pan, tilt
    if qr.x < CENTER_X and pan < PAN_UPPER_LIMIT:
        pan += 1
    elif pan > PAN_LOWER_LIMIT:
        pan -= 1
    
    if qr.y < CENTER_Y and tilt > TILT_LOWER_LIMIT:
        tilt -= 3
    elif tilt < TILT_UPPER_LIMIT:
        tilt += 3
    
    robobo.movePanTo(pan, 100)
    robobo.moveTiltTo(tilt, 100)

def is_qr_centered(qr):
    """
    Verifica si el QR está dentro del área central definida por los offsets.
    
    Args:
        qr (object): Objeto QR detectado.
        
    Returns:
        bool: True si el QR está centrado, False en caso contrario.
    """
    return (CENTER_X - X_OFFSET < qr.x < CENTER_X + X_OFFSET) and \
           (CENTER_Y - Y_OFFSET < qr.y < CENTER_Y + Y_OFFSET)

def danger(is_left):
    """
    Ejecuta una maniobra de evasión ante un peligro en izquierda o derecha.
    
    Args:
        is_left (bool): True si el peligro está a la izquierda, False si está a la derecha.
    """
    if is_left:
        robobo.moveWheelsByTime(SPEED, 0, 5)
    else:
        robobo.moveWheelsByTime(0, SPEED, 5)
    robobo.moveWheels(SPEED, SPEED)

def set_speed(speed):
    """
    Ajusta la velocidad de las ruedas del robot.
    
    Args:
        speed (int): Velocidad objetivo (0-100%).
    """
    robobo.moveWheels(speed * 0.5, speed * 0.5)

def pedestrians():
    """Reproduce un mensaje de voz para interactuar con peatones."""
    robobo.sayText('¡Hola peatones!')
    robobo.wait(1)

def stop():
    """Detiene todos los motores del robot."""
    robobo.stopMotors()

def yield_sign():
    """Detiene el robot durante 3 segundos y luego reanuda el movimiento."""
    robobo.stopMotors()
    robobo.wait(3)
    robobo.moveWheels(SPEED, SPEED)

def roundabout():
    """Gira el robot en círculos simulando una rotonda."""
    robobo.moveWheels(-SPEED, SPEED)


def qr_detected_callback():
    """
    Callback ejecutado al detectar un QR. Controla las acciones del robot según el QR leído.
    
    Returns:
        bool: True si el QR está lo suficientemente cerca, False en caso contrario.
    """
    global pan, tilt, last_qr, last_qr_sighting
    
    last_qr_sighting = 0
    qr = robobo.readQR()
    
    if not last_qr:
        last_qr = qr.id
    if not center_qr(qr):
        center_qr(qr)

    if last_qr == qr.id:
        robobo.stopMotors()
        
        if is_qr_centered(qr):
            robobo.moveWheelsByTime(5, 5, 5)
        
        if qr.distance > NEAR_DISTANCE:
            print(f'<< Señal: {qr.id}>>')
            match qr.id:
                case 'peligro izquierda': 
                    danger(is_left=True)
                case 'peligro derecha': 
                    danger(is_left=False)
                case '10': 
                    set_speed(10)
                case '20': 
                    set_speed(20)
                case '40': 
                    set_speed(40)
                case '50': 
                    set_speed(50)
                case 'peatones': 
                    pedestrians()
                case 'parar': 
                    stop()
                case 'rotonda': 
                    roundabout()
                case 'ceda': 
                    yield_sign()
            return True
    return False
    

if __name__ == "__main__":
    # Configuración inicial
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()

    # Asignar callback de detección de QR
    robobo.whenAQRCodeIsDetected(qr_detected_callback)
    
    try:
        while True:
            robobo.moveWheels(SPEED, SPEED)
            last_qr_sighting += 1
            
            # Resetear posición de la cámara si no hay detecciones recientes
            if last_qr_sighting > 15:
                pan, tilt = 0, 90
                robobo.movePanTo(pan, 100)
                robobo.moveTiltTo(tilt, 100)
            
            sleep(1)
    
    except KeyboardInterrupt:
        robobo.stopMotors()
        robobo.disconnect()

