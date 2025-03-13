from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

VERY_SHORT_FRONT = 15  # Umbral para sensores frontales
SPEED = 5 
TIME = 0.1
IP = '10.20.28.140'
#IP = 'localhost'

def tapDetectedCallback():
    '''
    Para el robot al tocarle la cara.
    '''
    print("A tap has been detected, stoping the robot...")
    robobo.stopMotors()
    #sim.disconnect()
    exit()

if __name__ == "__main__":
    # Conection
    #sim = RoboboSim(IP)
    #sim.connect()
    #sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    
    Robobo.resetTapSensor()
   # MOVEMENT
    robobo.whenATapIsDetected(tapDetectedCallback)  
    try:
        robobo.moveWheels(SPEED, SPEED)  
        while True:
            robobo.sayText("Si me tocas la cara, dejaré de moverme.")
            #robobo.wait(TIME)  
            time.sleep(2)

    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
