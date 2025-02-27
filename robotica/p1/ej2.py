from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.Emotions import Emotions
from robobopy.utils.Sounds import Sounds


"""
Cuando detecte un objeto se debe asustar (cara y sonido):
    - cabeza hacia atrás, retroceder con las ruedas
    - (luego) cara de realajado y sonido.
"""

SHORT = 40
SPEED = 5
TIME = 1

def estamos_bien():
    robobo.setLedColorTo(LED.All, Color.GREEN)
    print("Estamos bien, distancia: ", robobo.readIRSensor(IR.FrontC))
    robobo.wait(TIME)

def estamos_nerviosos():
    robobo.setLedColorTo(LED.All, Color.RED)
    robobo.stopMotors()
    robobo.moveWheels(-SPEED, -SPEED)
    robobo.setEmotionTo(Emotions.SURPRISED)
    robobo.playSound(Sounds.DISCOMFORT)
    robobo.moveTiltTo(50, 15)

def volvemos_a_estar_bien():
    robobo.wait(5)
    robobo.moveTiltTo(75, 4)
    robobo.setEmotionTo(Emotions.NORMAL)
    robobo.sayText("Ui, casi choco!")
    robobo.playSound(Sounds.LAUGH)
    robobo.stopMotors()
    robobo.disconnect()

########

# Conection
robobo = Robobo("localhost")
robobo.connect()


# MOVEMENT
robobo.moveWheels(SPEED, SPEED)

# estamos bien
while (
    (robobo.readIRSensor(IR.FrontC) < SHORT)
    and (robobo.readIRSensor(IR.FrontRR) < SHORT)
    and (robobo.readIRSensor(IR.FrontLL) < SHORT)
):
    estamos_bien()

estamos_nerviosos()

volvemos_a_estar_bien()

