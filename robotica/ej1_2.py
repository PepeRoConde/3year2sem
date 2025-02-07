from robobopy.Robobo import Robobo
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.IR import IR

'''
CAMBIAR LOS SENSORES A: Front-L, Front-R, Back-R y Back-L
'''

LONG = 8
MEDIUM = 20 
SHORT = 50
VERY_SHORT = 100
SPEED = 5
TIME = 1
# Conection


robobo = Robobo('localhost')
robobo.connect()
    
# MOVEMENT
robobo.moveWheels(SPEED,SPEED)

# LONG DISTANCE
while (robobo.readIRSensor(IR.FrontC) < LONG) and \
        (robobo.readIRSensor(IR.FrontRR) < LONG) and \
        (robobo.readIRSensor(IR.FrontLL) < LONG):
    
    robobo.setLedColorTo(LED.All, Color.GREEN)
    print("Distancia Larga:", robobo.readIRSensor(IR.FrontC))
    robobo.wait(TIME)
# MEDIUM DISTANCE
while (robobo.readIRSensor(IR.FrontC) < MEDIUM) and \
        (robobo.readIRSensor(IR.FrontRR) < MEDIUM) and \
        (robobo.readIRSensor(IR.FrontLL) < MEDIUM):
    robobo.setLedColorTo(LED.All, Color.YELLOW)
    print("Distancia Media: ", robobo.readIRSensor(IR.FrontC))
    robobo.wait(TIME)

# SHORT DISTANCE
while (robobo.readIRSensor(IR.FrontC) < SHORT) and \
        (robobo.readIRSensor(IR.FrontRR) < SHORT) and \
        (robobo.readIRSensor(IR.FrontLL) < SHORT):
    robobo.setLedColorTo(LED.All, Color.RED)
    print("Distancia Corta: ", robobo.readIRSensor(IR.FrontC))
    robobo.wait(TIME)

# VERY SHORT DISTANCE
while (robobo.readIRSensor(IR.FrontC) < VERY_SHORT) and \
        (robobo.readIRSensor(IR.FrontRR) < VERY_SHORT) and \
        (robobo.readIRSensor(IR.FrontLL) < VERY_SHORT):
    robobo.stopMotors()

