from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
import time

VERY_SHORT = 15  # Umbral para sensores frontales
SPEED = 5 
TIME = 0.01
IP = 'localhost'

def avoid_falling(speed, distance):
    '''
    Evita la caida al llegar a una distancia indicada utilizando sensores
    '''
    robobo.moveWheels(speed, speed)

    while True:
        front_l = robobo.readIRSensor(IR.FrontL)
        front_r = robobo.readIRSensor(IR.FrontR)

        min_front = min(front_l, front_r)

        print(f"Front: {front_l}, {front_r}")
        # Reducción de velocidad antes de parar
        # if min_front < (VERY_SHORT_FRONT * 2):
        #     robobo.moveWheels(SLOW_SPEED, SLOW_SPEED)

        # Parada antes de que las ruedas traseras queden en el aire
        if min_front < distance:
            robobo.stopMotors()
            break

        time.sleep(0.1)  


if __name__ == "__main__":
    # Conection
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()

    try:
        avoid_falling(SPEED, VERY_SHORT)
    except KeyboardInterrupt:
            robobo.stopMotors()
            sim.disconnect()
