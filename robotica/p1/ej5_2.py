from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import time

SPEED = 5 
TIME = 2
# IP = '10.20.28.140'
IP = 'localhost'
NEW_DETECTION = 5 

# Evitamos estar constantemente viendo el mismo color con un counter
last_red_detection_time = 0

def runAwayFromRed(color_name):
    '''
    Dice el nombre del color y se aleja de el durante un tiempo. 
    Depues gira para evitar verlo otra vez
    '''
    robobo.sayText(color_name)
    robobo.moveWheelsByTime(-SPEED, -SPEED, 3)
    robobo.moveWheels(SPEED, -SPEED)


def blobDetectedCallback():
    '''
    Si detecta el color rojo huye de el, con el resto dice el color.
    '''
    global last_red_detection_time
    
    CENTER_X = 50
    MARGIN = 15  # Rango 35-65 (30% del centro)
    blobs = robobo.readAllColorBlobs()
    
    if not blobs:
        return

    current_time = time.time()
    for key in blobs:
        blob = blobs[key]
        if blob.size <= 0 or abs(blob.posx - CENTER_X) > MARGIN:
            continue  # Ignorar blobs no centrados

        if key == 'green':
            if current_time - last_red_detection_time < NEW_DETECTION:
                continue
            last_red_detection_time = current_time
            robobo.stopMotors()
            runAwayFromRed(key)
            break
        else:
            robobo.sayText(f"{key}")
    
    robobo.resetColorBlobs()

if __name__ == "__main__":
    #sim = RoboboSim(IP)
    #sim.connect()
    #sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(90, 5)
    # Activamos toods los colores
    robobo.setActiveBlobs(True, True, True, False)
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)

    try:
        robobo.moveWheels(5, -5)
        while True:
            time.sleep(TIME)
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()

