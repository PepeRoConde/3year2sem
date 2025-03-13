from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

SPEED = 1
IP = '10.20.28.140'
#IP = 'localhost'
TIME = 0.1

def move_forward(warning):
    '''
    Ajusta la velocidad de las ruedas en funcion del peligro
    '''
    if warning:
        speed = 1
    else:
        speed = 1
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
    global var
    if distance < max(
        robobo.readIRSensor(IR.FrontC),
        robobo.readIRSensor(IR.FrontRR),
        robobo.readIRSensor(IR.FrontLL),
    ):
        
        robobo.moveWheelsByTime(-speed, speed, 5)
        var =True

    else:
        var = False


if __name__ == "__main__":
    #sim = RoboboSim(IP)
    #sim.connect()
    #sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()

    robobo.moveTiltTo(90, 5)
    # Activamos el blob verde
    robobo.setActiveBlobs(False, True, False, False)
    #robobo.moveWheels(SPEED, SPEED)

    warning = False
    var = False
    while not var:
        # Nos movemos ajustando la speed
        move_forward(warning)
        # Si vemos el blob nos ponemos en modo alarma
        if seesBlob():
            cuidado = True
        # Si estamos tocando el blob giramos para moverlo
        blob_is_close(SPEED)

        time.sleep(TIME)
