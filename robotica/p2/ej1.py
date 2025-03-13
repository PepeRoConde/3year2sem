
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

# USAR UN PI -> PRECISION SIN OSCILACIONES GRADNES


TIME = 0.5
IP = 'localhost'

KPav = 0.2 
KPct = 0.25
KD = 0 # ej 1: se deja a cero porque equivale a usar sólo control proporcional
KI = 0 # ej 1: se deja a cero porque equivale a usar sólo control proporcional
Iav = 0 # para el control del AVance
Ict = 0 # para el control del CenTro
task_completed = False
error_avanzar_previo = 0
CENTER = 50
ERROR_MARGIN_center = 7
ERROR_MARGIN_avance = 190 
searchSpeed = 15
speedAvance = 35

def applyCorrection(speed, correction):
    '''
    Gestiona el caso en que la correccin llevaria a la parada. en ese caso que valga 5.
    '''
    return max(speed - correction, 5)


def moveToAColor(color_blob):
    '''
    Mueve el robot hasta llegar a una distancia indicada
    '''
    global Iav, error_avanzar_previo, speedAvance

    robobo.moveWheelsByTime(speedAvance, speedAvance, TIME)
    error_avanzar = robobo.readIRSensor(IR.FrontC)
    # error avanzar es 0 si esta muy lejos y ~200 si esta muy cerca,
    # si esta suficientemente cerca, para
    while error_avanzar < ERROR_MARGIN_avance: 

        if abs(color_blob.posx - CENTER) > ERROR_MARGIN_center:
            centerToAColor(color_blob)

        P = error_avanzar 
        D = error_avanzar - error_avanzar_previo
        Iav += error_avanzar

        correction = round(P * KPav + D * KD + Iav * KI)
        speed = applyCorrection(speedAvance, correction)
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

    robobo.stopMotors()
    error_centrar = color_blob.posx - CENTER # [0,100] - 50 = [-50,50]
    speedCenter = KPct * error_centrar # 0.1 * [-50,50] = [-5,5]
    # si esta hacia la derecha, speedCenter es positivo
    if abs(error_centrar) > ERROR_MARGIN_center: 
        print('rs', speedCenter)
        robobo.moveWheelsByTime(-speedCenter, ROTATION_SPEED, TIME)
    else:
        return

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
    robobo.moveTiltTo(110, 5) # posición inicial
    robobo.setActiveBlobs(True, True, False, False) 
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)  # Corrección clave

    try:
        robobo.moveWheels(searchSpeed, -searchSpeed)  # Búsqueda giratoria
        while True:
            time.sleep(0.1)  # Reducir carga de CPU
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
