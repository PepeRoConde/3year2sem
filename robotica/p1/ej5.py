from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

SPEED = 5 
TIME = 2
VERY_SHORT = 25 
IP = 'localhost'

def moveToAColor():

    robobo.moveWheels(SPEED,SPEED)
    while robobo.readIRSensor(IR.FrontC) < VERY_SHORT and \
            robobo.readIRSensor(IR.FrontRR) < VERY_SHORT and \
            robobo.readIRSensor(IR.FrontLL) < VERY_SHORT:
        print("Distance Front: ", robobo.readIRSensor(IR.FrontC))
        print("Distance Right: ", robobo.readIRSensor(IR.FrontRR))
        print("Distance Left: ", robobo.readIRSensor(IR.FrontLL))
        time.sleep(1)
    robobo.stopMotors()
    robobo.disconnect()
    sim.disconnect()

def blobDetectedCallback():


    color = BlobColor.RED
    print("A color has been detected")
    robobo.stopMotors()

    color_blob = robobo.readColorBlob(color)
    positionX = color_blob.posx 
    area = color_blob.size

    # robobo.sayText(f"The object is at {area} distance, and it's in the f{positionX},f{positionY} position.")
    if positionX < 50:
        orientation = "left"
    elif positionX > 75:
        orientation = "right"
    else:
        orientation = "front"
    robobo.sayText(f"Area: {area}, Distance: {orientation}")
    # Definir centro de la imagen para alinear el objeto
    while positionX not in range(50,100):
        print(positionX)
        if positionX < 75:
            robobo.sayText("Moving Left")
            robobo.moveWheelsByTime(-10, 10, 0.5)  # Girar a la izquierda
        elif positionX > 75:
            robobo.sayText("Moving Right")
            robobo.moveWheelsByTime(10, -10, 0.5)  # Girar a la derecha
 
        color_blob = robobo.readColorBlob(color)
        positionX = color_blob.posx 
        area = color_blob.size
           
    print("Moving")
    moveToAColor()
    

if __name__ == "__main__":
    # Conection
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(110, 5) 
    robobo.setActiveBlobs(True,False,False,False)
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)
    # Color Logic
    try:
        robobo.moveWheels(10, -10)  # Wheels, Degree, Speed
        while True:
            time.sleep(TIME) 

    except KeyboardInterrupt:
            robobo.stopMotors()
            sim.disconnect()
