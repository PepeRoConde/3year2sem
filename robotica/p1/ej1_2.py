from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
import time

VERY_SHORT_FRONT = 15  # Umbral para sensores frontales
# SLOW_SPEED = 10        # Velocidad reducida antes de frenar
SPEED = 30
TIME = 0.01
IP = 'localhost'

# Conection
sim = RoboboSim(IP)
sim.connect()
sim.resetSimulation()

robobo = Robobo(IP)
robobo.connect()

# MOVEMENT
robobo.moveWheels(SPEED,SPEED)
# robobo.moveWheelsByDegrees(Wheels.BOTH, TURN_DEGREES, TURN_SPEED)   # Gira en el lugar
while True:
    front_l = robobo.readIRSensor(IR.FrontL)
    front_r = robobo.readIRSensor(IR.FrontR)

    min_front = min(front_l, front_r)

    print(f"Front: {front_l}, {front_r}")
    # Reducción de velocidad antes de parar
    # if min_front < (VERY_SHORT_FRONT * 2):
    #     robobo.moveWheels(SLOW_SPEED, SLOW_SPEED)

    # Parada antes de que las ruedas traseras queden en el aire
    if min_front < VERY_SHORT_FRONT:
        robobo.stopMotors()
        break

    time.sleep(0.1)  
