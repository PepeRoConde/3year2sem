from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.Emotions import Emotions
from robobopy.utils.Sounds import Sounds
import time


SHORT = 30
SPEED = 5
TIME = .1
IP = "10.56.43.179"
#IP = 'localhost'

def normal_state():
    '''
    Expresion normal, pone los leds a verde y dice que no hay peligro
    '''
    robobo.setLedColorTo(LED.All, Color.GREEN)
    print("We are fine, distance: ", robobo.readIRSensor(IR.FrontC))
    time.sleep(TIME)

def run_away(speed):
    '''
    Expresion de susto, pone los leds en rojo y se mueve hacia atras con un sonido
    '''
    robobo.setLedColorTo(LED.All, Color.RED)
    robobo.stopMotors()
    robobo.moveWheelsByTime(-speed, -speed,4)
    robobo.setEmotionTo(Emotions.SURPRISED)
    robobo.playSound(Sounds.DISCOMFORT)
    robobo.moveTiltTo(50, 15)

def return_to_normal_state():
    '''
    Paso a estado normal despues de haber sido asustado
    '''
    robobo.moveTiltTo(75, 4)
    robobo.setEmotionTo(Emotions.NORMAL)
    robobo.sayText("Oops, almost crashed!")
    robobo.playSound(Sounds.LAUGH)
    robobo.stopMotors()
    robobo.disconnect()

if __name__ == "__main__":
    # Conection
    #sim = RoboboSim(IP)
    #sim.connect()
    #sim.resetSimulation()

    # Connection
    robobo = Robobo(IP)
    robobo.connect()

    # Movement
    robobo.moveWheels(SPEED, SPEED)
    try: 
        while (
            (robobo.readIRSensor(IR.FrontC) < SHORT)
            and (robobo.readIRSensor(IR.FrontRR) < SHORT)
            and (robobo.readIRSensor(IR.FrontLL) < SHORT)
        ):
            normal_state()

        run_away(SPEED)
        return_to_normal_state()

    except KeyboardInterrupt:
            robobo.stopMotors()
            sim.disconnect()
