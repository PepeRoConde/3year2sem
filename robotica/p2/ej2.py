from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

# USAR UN PI -> PRECISION SIN OSCILACIONES GRADNES


SPEED = 5 
TIME = 2
VERY_SHORT = 25 
ROTATION_SPEED = 10
IP = 'localhost'

KP = 0.1
KD = 0.1
KI = 0.05
Iav = 0 # para el control del AVance
Ict = 0 # para el control del CenTro
K_centrar = 0.4
task_completed = False
error_avanzar_previo = 0
CENTER = 50
ERROR_MARGIN = 30

def applyCorrection(speed, correction):
    return max(speed - correction, 5)


def moveToAColor(goal):
    '''
    Mueve el robot hasta llegar a una distancia indicada
    '''
    global Iav, error_avanzar_previo
    speed = 15
    robobo.stopMotors()
    robobo.moveWheelsByTime(speed, speed, TIME)
    print("Distance: ", robobo.readIRSensor(IR.FrontC))
    error_avanzar = goal - robobo.readIRSensor(IR.FrontC)
    while error_avanzar > ERROR_MARGIN:
        P = error_avanzar 
        D = error_avanzar - error_avanzar_previo
        Iav += error_avanzar
        correction = round(P * KP + D * KD + Iav * KI)
        speed = applyCorrection(speed, correction)
        print(f'speed: {speed}, P: {P}, D: {D}, I: {Iav}, correction: {correction}')
        robobo.moveWheelsByTime(speed, speed, TIME)
        error_avanzar_previo = error_avanzar


def centerToAColor(color_blob):
    '''
    centra el blob usando control proporcional
    '''
    global Ict

    robobo.stopMotors()

    error_centrar = color_blob.posx - CENTER
    ROTATION_SPEED = K_centrar * error_centrar
    if color_blob.posx < CENTER - ERROR_MARGIN:
        robobo.moveWheelsByTime(-ROTATION_SPEED, ROTATION_SPEED, 0.3)  # Giro suave a la derecha

    else:
        robobo.moveWheelsByTime(ROTATION_SPEED, -ROTATION_SPEED, 0.3)  # Giro suave a la izquierda
    
    time.sleep(0.5)  

def blobDetectedCallback():
    '''
    Detecta un color, lo centra en su camara y se mueve hasta el sin llegar a chocar.
    '''
    global task_completed

    robobo.stopMotors()

    color = BlobColor.RED
    color_blob = robobo.readColorBlob(color)
    
    if color_blob.size <= 0:
        return

    if task_completed:
        return


    while abs(color_blob.posx - CENTER) > ERROR_MARGIN:
        color_blob = robobo.readColorBlob(color)
        centerToAColor(color_blob)
    # Si hay un color nos movemos a el
    if color_blob.size > 0:
        moveToAColor(color_blob.posx)
        task_completed = True

if __name__ == "__main__":
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(110, 5)
    robobo.setActiveBlobs(True, False, False, False)
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)  # Corrección clave

    try:
        robobo.moveWheels(5, -5)  # Búsqueda giratoria
        while True:
            time.sleep(0.1)  # Reducir carga de CPU
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
