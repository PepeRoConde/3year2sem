from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
import time
import math

'''
- Añadir logica de giros
- Comprobar que las distancia es correcta
- Comprobar funcionamiento general 
'''

SPEED = 5
TIME = 0.1
DISTANCE = 15
IP = "localhost"


def moveToAColor():
    robobo.moveWheels(SPEED, SPEED)

    # Movement logic
    while True:
        front = robobo.readIRSensor(IR.FrontC)
        frontR = robobo.readIRSensor(IR.FrontRR)
        frontL = robobo.readIRSensor(IR.FrontLL)

        min_distance = min(front, frontR, frontL)
        if min_distance < DISTANCE:
            robobo.stopMotors()
            break
        time.sleep(TIME)


def blobDetectedCallback(color):
    print("A color has been detected")
    robobo.stopMotors()
    positionX = robobo.readColorBlob(color).posx
    positionY = robobo.readColorBlob(color).posy
    area = robobo.readColorBlob(color).size
    distance = robobo.readTiltPosition() * math.sqrt((positionX * positionY) / area)
    robobo.sayText(
        f"The object is at {distance} distance, and it's in the f{positionX},f{positionY} position."
    )

    # MIRAR QUE DEVUELVE positionX
    if positionX != "front":
        # Girar
        pass
    robobo.sayText("I'm going to move to the object.")


if __name__ == "__main__":
    # Conection
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()

    # Color Logic
    color = "red"
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback(color))
    moveToAColor()
