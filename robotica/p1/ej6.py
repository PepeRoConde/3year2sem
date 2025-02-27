# inspirar en el ej5 y funcion mover en el callbac


from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from time import sleep

"""
Hacer que (con el gancho o _pusher_) coja un cilindro y 
después pare. Usando cámara (para el blob) y el infrarojos
(para la proximitud).
"""

cuidado = False
SPEED = 5

def avanzar(cuidado):
    if cuidado:
        velocidad = 5
    else:
        velocidad = 10 
    print(velocidad)
    robobo.moveWheels(velocidad, velocidad)


def veeBlob():
    blob = robobo.readColorBlob(BlobColor.GREEN)
    if blob is None:
        return False
    else:
        return True


def cilindroEstaCerca():
    print(
        "dis",
        robobo.readIRSensor(IR.FrontLL),
        robobo.readIRSensor(IR.FrontRR),
        robobo.readIRSensor(IR.FrontC),
        min(robobo.readIRSensor(IR.FrontRR), robobo.readIRSensor(IR.FrontLL)),
    )
    if 1000 < max(
        robobo.readIRSensor(IR.FrontC),
        robobo.readIRSensor(IR.FrontRR),
        robobo.readIRSensor(IR.FrontLL),
    ):
        
        robobo.moveWheels(-SPEED, SPEED)

    else:
        return


if __name__ == "__main__":
    robobo = Robobo("localhost")
    robobo.connect()
    robobo.setActiveBlobs(False, True, False, False)
    robobo.moveWheels(SPEED, SPEED)
    while True:
        sleep(1)
        avanzar(cuidado)

        if veeBlob():
            cuidado = True
        cilindroEstaCerca()
