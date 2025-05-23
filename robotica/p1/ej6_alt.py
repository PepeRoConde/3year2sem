from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
import time

SPEED = 5 
TIME = 2
IP = 'localhost'
NEW_DETECTION = 5 

# Evitamos estar constantemente viendo el mismo color con un counter
last_red_detection_time = 0

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
            if 1000 < max(
                robobo.readIRSensor(IR.FrontC),
                robobo.readIRSensor(IR.FrontRR),
                robobo.readIRSensor(IR.FrontLL),
            ):

                robobo.stopMotors()
                robobo.moveWheels(-SPEED, SPEED)
            break
        else:
            robobo.sayText(f"{key}")
    
    robobo.resetColorBlobs()

if __name__ == "__main__":
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(110, 5)
    # Activamos toods los colores
    robobo.setActiveBlobs(True, True, True, False)
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)

    try:
        robobo.moveWheels(3, 3)
        while True:
            time.sleep(TIME)
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()

