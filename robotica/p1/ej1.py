from robobopy.Robobo import Robobo
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim

IP = "localhost"
LONG = 8
MEDIUM = 20 
SHORT = 50
VERY_SHORT = 100
SPEED = 5
TIME = 1

def avoid_obstacles(speed, very_short, short, medium, long):
    '''
    Indica las diferentes distancias con los objetos, y al llegar a la minima
    se para.
    '''
    # MOVEMENT
    robobo.moveWheels(speed, speed)

    # LONG DISTANCE
    while (robobo.readIRSensor(IR.FrontC) < long) and \
            (robobo.readIRSensor(IR.FrontRR) < long) and \
            (robobo.readIRSensor(IR.FrontLL) < long):
        
        robobo.setLedColorTo(LED.All, Color.GREEN)
        print("Distancia Larga:", robobo.readIRSensor(IR.FrontC))
        robobo.wait(TIME)

    # MEDIUM DISTANCE
    while (robobo.readIRSensor(IR.FrontC) < medium) and \
            (robobo.readIRSensor(IR.FrontRR) < medium) and \
            (robobo.readIRSensor(IR.FrontLL) < medium):
        robobo.setLedColorTo(LED.All, Color.YELLOW)
        print("Distancia Media: ", robobo.readIRSensor(IR.FrontC))
        robobo.wait(TIME)

    # SHORT DISTANCE
    while (robobo.readIRSensor(IR.FrontC) < short) and \
            (robobo.readIRSensor(IR.FrontRR) < short) and \
            (robobo.readIRSensor(IR.FrontLL) < short):
        robobo.setLedColorTo(LED.All, Color.RED)
        print("Distancia Corta: ", robobo.readIRSensor(IR.FrontC))
        robobo.wait(TIME)

    # VERY SHORT DISTANCE
    while (robobo.readIRSensor(IR.FrontC) < very_short) and \
            (robobo.readIRSensor(IR.FrontRR) < very_short) and \
            (robobo.readIRSensor(IR.FrontLL) < very_short):
        robobo.stopMotors()
        sim.disconnect()
        return

if __name__ == "__main__":
    # Conection
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()

    try:
        avoid_obstacles(SPEED, VERY_SHORT, SHORT, MEDIUM, LONG)
    except KeyboardInterrupt:
            robobo.stopMotors()
            sim.disconnect()
