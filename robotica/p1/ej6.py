from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

SPEED = 5
IP = 'localhost'
TIME = 0.1

def move_forward(warning):
    '''
    Ajusta la velocidad de las ruedas en funcion del peligro
    '''
    if warning:
        speed = 5
    else:
        speed = 10 
    # print(speed)
    robobo.moveWheels(speed, speed)


def seesBlob():
    '''
    Comprueba si esta viendo el blob
    '''
    blob = robobo.readColorBlob(BlobColor.GREEN)
    if blob is None:
        return False
    else:
        return True


def blob_is_close(speed, distance=1000):
    '''
    Si esta tocando al blob, gira hacia la derecha para mover el objeto
    '''
    if distance < max(
        robobo.readIRSensor(IR.FrontC),
        robobo.readIRSensor(IR.FrontRR),
        robobo.readIRSensor(IR.FrontLL),
    ):
        
        robobo.moveWheels(-speed, speed)

    else:
        return


if __name__ == "__main__":
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()

    robobo.moveTiltTo(110, 5)
    # Activamos el blob verde
    robobo.setActiveBlobs(False, True, False, False)
    robobo.moveWheels(SPEED, SPEED)

    warning = False
    while True:
        # Nos movemos ajustando la speed
        move_forward(warning)
        # Si vemos el blob nos ponemos en modo alarma
        if seesBlob():
            cuidado = True
        # Si estamos tocando el blob giramos para moverlo
        blob_is_close(SPEED)
        time.sleep(TIME)
