from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

VERY_SHORT_FRONT = 15  # Umbral para sensores frontales
SPEED = 5 
TIME = 0.1
IP = 'localhost'

def tapDetectedCallback():
    '''
    Para el robot al tocarle la cara.
    '''
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
        robobo.moveWheels(SPEED, SPEED)  
        while True:
            robobo.sayText("Si me tocas la cara, dejaré de moverme.")
            robobo.whenATapIsDetected(tapDetectedCallback())  
            robobo.wait(TIME)  

    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
