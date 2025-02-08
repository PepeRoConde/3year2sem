from robobopy.Robobo import Robobo, Wheels
from robobosim.RoboboSim import RoboboSim
import time

VERY_SHORT_FRONT = 15  # Umbral para sensores frontales
SPEED = 5 
TIME = 0.1
IP = 'localhost'

def tapDetectedCallback():
    print("A tap has been detected, stoping the robot...")
    robobo.stopMotors()
    sim.disconnect()
    exit()

if __name__ == "__main__":
    # Conection
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    
    # MOVEMENT
    try:
        while True:
            robobo.moveWheels(SPEED, SPEED)  
            robobo.wait(2)  
            robobo.sayText("Si me tocas la cara, dejaré de moverme.")
            robobo.whenATapIsDetected(tapDetectedCallback())  

    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
