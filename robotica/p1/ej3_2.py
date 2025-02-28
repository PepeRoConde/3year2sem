from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import time

VERY_SHORT_FRONT = 15  # Umbral para sensores frontales
SPEED = 5 
TIME = 0.1
IP = 'localhost'

def tapDetectedCallback():
    '''
    Para el robot al tocarle
    ''' 
    # print("A tap has been detected, stoping the robot...")
    robobo.stopMotors()

if __name__ == "__main__":
    # Conection
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    
    # MOVEMENT
    try:
        robobo.moveWheels(SPEED, SPEED)  
        while True:
            # robobo.sayText("Si me tocas la cara, dejaré de moverme.")
            tap = robobo.readTapSensor()
            # Si es el ojo paramos el robot
            if tap.zone == 'eye':
                robobo.whenATapIsDetected(tapDetectedCallback())  
            # En el resto de zonas reactivamos el movimiento
            elif tap.zone != 'eye':
                robobo.moveWheels(SPEED,SPEED)
            time.sleep(TIME)

    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
