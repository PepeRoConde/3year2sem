from robobopy.Robobo import Robobo
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim

#IP = "localhost"
IP = '10.20.28.140'
LONG = 3
MEDIUM = 5 
SHORT = 10
VERY_SHORT = 20
SPEED = 2
TIME = .1

def avoid_obstacles(speed, very_short, short, medium, long):
    '''
    Indica las diferentes distancias con los objetos, y al llegar a la minima
    se para.
    '''
    # MOVEMENT
    robobo.moveWheels(speed, speed)

    # LONG DISTANCE
    robobo.setLedColorTo(LED.All, Color.GREEN)
    while (robobo.readIRSensor(IR.FrontC) < long) and \
            (robobo.readIRSensor(IR.FrontRR) < long) and \
            (robobo.readIRSensor(IR.FrontLL) < long):
        
        print("Distancia Larga:", robobo.readIRSensor(IR.FrontC))
        robobo.wait(TIME)

    # MEDIUM DISTANCE
    robobo.setLedColorTo(LED.All, Color.YELLOW)
    while (robobo.readIRSensor(IR.FrontC) < medium) and \
            (robobo.readIRSensor(IR.FrontRR) < medium) and \
            (robobo.readIRSensor(IR.FrontLL) < medium):
        print("Distancia Media: ", robobo.readIRSensor(IR.FrontC))
        robobo.wait(TIME)

    # SHORT DISTANCE
    robobo.setLedColorTo(LED.All, Color.RED)
    while (robobo.readIRSensor(IR.FrontC) < short) and \
            (robobo.readIRSensor(IR.FrontRR) < short) and \
            (robobo.readIRSensor(IR.FrontLL) < short):
        print("Distancia Corta: ", robobo.readIRSensor(IR.FrontC))
        robobo.wait(TIME)

    # VERY SHORT DISTANCE
    while (robobo.readIRSensor(IR.FrontC) < very_short) and \
            (robobo.readIRSensor(IR.FrontRR) < very_short) and \
            (robobo.readIRSensor(IR.FrontLL) < very_short):
        robobo.wait(TIME)
        #sim.disconnect()
        return
    robobo.stopMotors()

if __name__ == "__main__":
    # Conection
    '''
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()
    '''
    robobo = Robobo(IP)
    robobo.connect()
    try:
        avoid_obstacles(SPEED, VERY_SHORT, SHORT, MEDIUM, LONG)
    except KeyboardInterrupt:
            robobo.stopMotors()
            sim.disconnect()
