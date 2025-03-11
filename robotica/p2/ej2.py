from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

# USAR UN PI -> PRECISION SIN OSCILACIONES GRADNES


TIME = 0.5
ROTATION_SPEED = 10
IP = 'localhost'

KP = 0.2
KD = 0.2
KI = 0.05
Iav = 0 # para el control del AVance
Ict = 0 # para el control del CenTro
K_centrar = 0.3
task_completed = False
error_avanzar_previo = 0
CENTER = 50
ERROR_MARGIN_center = 7
ERROR_MARGIN_avance = 150 
searchSpeed = 15

def applyCorrection(speed, correction):
    return max(speed - correction, 5)


def moveToAColor(color_blob):
    '''
    Mueve el robot hasta llegar a una distancia indicada
    '''
    global Iav, error_avanzar_previo

    #if color_blob.size <= 0:
    #    return
    speed = 25
    robobo.stopMotors()
    robobo.moveWheelsByTime(speed, speed, TIME)
    print("Distance: ", robobo.readIRSensor(IR.FrontC))
    error_avanzar = robobo.readIRSensor(IR.FrontC)
    # error avanzar es 0 si esta muy lejos y XXX si esta muy cerca,
    # si esta suficientemente cerca, para
    while error_avanzar < ERROR_MARGIN_avance: 
        if abs(color_blob.posx - CENTER) > ERROR_MARGIN_center:
            centerToAColor(color_blob)
        P = error_avanzar 
        D = error_avanzar - error_avanzar_previo
        Iav += error_avanzar
        correction = round(P * KP + D * KD + Iav * KI)
        speed = applyCorrection(speed, correction)
        #print(f'speed: {speed}, P: {P}, D: {D}, I: {Iav}, correction: {correction}, blb:{abs(color_blob.posx - CENTER)}')
        robobo.moveWheelsByTime(speed, speed, TIME)
        error_avanzar_previo = error_avanzar
        error_avanzar = robobo.readIRSensor(IR.FrontC)
    robobo.stopMotors()
    sim.disconnect()


def centerToAColor(color_blob):
    '''
    centra el blob usando control proporcional
    '''
    global Ict

    #if color_blob.size <= 0:
    #    return

    robobo.stopMotors()
    print(f'hola {color_blob.posx}')
    error_centrar = color_blob.posx - CENTER # [0,100] - 50 = [-50,50]
    ROTATION_SPEED = K_centrar * error_centrar # 0.1 * [-50,50] = [-5,5]
    # si esta hacia la derecha, ROTATION_SPEED es positivo
    if abs(error_centrar) > ERROR_MARGIN_center: 
        print('rs', ROTATION_SPEED)
        robobo.moveWheelsByTime(-ROTATION_SPEED, ROTATION_SPEED, TIME)
    else:
        return
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

    while abs(color_blob.posx - CENTER) > ERROR_MARGIN_center:
        color_blob = robobo.readColorBlob(color)
        centerToAColor(color_blob)
    # Si hay un color nos movemos a el
    if color_blob.size > 0:
        moveToAColor(color_blob)
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
        robobo.moveWheels(searchSpeed, -searchSpeed)  # Búsqueda giratoria
        while True:
            time.sleep(0.2)  # Reducir carga de CPU
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
