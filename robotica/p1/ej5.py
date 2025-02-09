from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.Wheels import Wheels
import time 
import math

SPEED = 5 
TIME = 2 
DISTANCE = 15
IP = 'localhost'

def moveToAColor():
    robobo.moveWheels(SPEED,SPEED)

    # Movement logic
    while True:
        front = robobo.readIRSensor(IR.FrontC)
        frontR = robobo.readIRSensor(IR.FrontRR)
        frontL = robobo.readIRSensor(IR.FrontLL)

        min_distance = min(front,frontR, frontL)
        if min_distance < DISTANCE: 
            robobo.stopMotors()
            break
        time.sleep(TIME)

def blobDetectedCallback():

    color = BlobColor.RED
    robobo.setActiveBlobs(1,0,0,0)
    print("A color has been detected")
    robobo.stopMotors()
    
    color_blob = robobo.readColorBlob(color)
    positionX = color_blob.posx 
    positionY = color_blob.posy
    area = color_blob.size

    # Valores de referencia para obtener la distancia real
    distance_ref = 30
    area_ref = 5000

    if area > 0:
        distance = distance_ref * math.sqrt(area_ref / area) 
    else:
        distance = float('inf')

    robobo.sayText(f"The object is at {distance} distance, and it's in the f{positionX},f{positionY} position.")

    # MIRAR QUE DEVUELVE positionX
    
    # Definir centro de la imagen para alinear el objeto
    image_center = 320  # Asumiendo una imagen de 640px de ancho
    tolerance = 30  # Tolerancia en píxeles

    if positionX < image_center - tolerance:
        robobo.sayText("Turning left to align with the object.")
        robobo.moveWheels(-10, 10)  # Girar a la izquierda
    elif positionX > image_center + tolerance:
        robobo.sayText("Turning right to align with the object.")
        robobo.moveWheels(10, -10)  # Girar a la derecha
        
    moveToAColor()

if __name__ == "__main__":
    # Conection
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)
    # Color Logic
    try:
        while True:
            robobo.moveWheels(-10, 10)  # Wheels, Degree, Speed
            time.sleep(TIME) 
    except KeyboardInterrupt:
            robobo.stopMotors()
            sim.disconnect()
