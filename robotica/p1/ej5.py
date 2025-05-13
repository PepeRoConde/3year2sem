from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

SPEED = 5 
TIME = 2
VERY_SHORT = 45 
# IP = '10.20.28.140'
IP = 'localhost'
task_completed = False

def moveToAColor(speed, distance):
    '''
    Mueve el robot hasta llegar a una distancia indicada
    '''
    robobo.moveWheels(speed, speed)
    while robobo.readIRSensor(IR.FrontC) < distance and \
            robobo.readIRSensor(IR.FrontRR) < distance and \
            robobo.readIRSensor(IR.FrontLL) < distance:
        print("Distance Front: ", robobo.readIRSensor(IR.FrontC))
        time.sleep(1)
    robobo.stopMotors()
    print('para')
    #sim.disconnect()

def blobDetectedCallback():
    '''
    Detecta un color, lo centra en su camara y se mueve hasta el sin llegar a chocar.
    '''
    global task_completed

    color = BlobColor.GREEN
    color_blob = robobo.readColorBlob(color)
    
    if color_blob.size <= 0:
        return

    if task_completed:
        return

    CENTER = 50
    ERROR_MARGIN = 5

    while abs(color_blob.posx - CENTER) > ERROR_MARGIN:
        if color_blob.posx < CENTER - ERROR_MARGIN:
            robobo.moveWheelsByTime(-15, 15, 0.3)  # Giro suave a la derecha
        else:
            robobo.moveWheelsByTime(15, -15, 0.3)  # Giro suave a la izquierda
        
        time.sleep(0.5)  
        color_blob = robobo.readColorBlob(color)
    # Si hay un color nos movemos a el
    if color_blob.size > 0:
        moveToAColor(SPEED, VERY_SHORT)
        task_completed = True

if __name__ == "__main__":
    #sim = RoboboSim(IP)
    #sim.connect()
    #sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(90, 5)
    robobo.setActiveBlobs(True, True, False, False)
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)  # Corrección clave

    try:
        robobo.moveWheels(10, -10)  # Búsqueda giratoria
        while True:
            time.sleep(0.1)  # Reducir carga de CPU
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
