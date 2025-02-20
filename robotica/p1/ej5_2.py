from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import time

SPEED = 5 
TIME = 2
IP = 'localhost'
NEW_DETECTION = 5 

last_red_detection_time = 0

def runAwayFromRed():
    robobo.sayText("Rojo")
    robobo.moveWheelsByTime(-SPEED, -SPEED, 1)
    robobo.moveWheels(SPEED, -SPEED)


def blobDetectedCallback():
    global last_red_detection_time
    blobs = robobo.readAllColorBlobs()
    if not blobs:
        return

    current_time = time.time()
    for key in blobs:
        blob = blobs[key]
        print(key)
        # Solo se considera el blob si tiene tamaño mayor a 0
        if blob.size <= 0:
            continue

        # Si es rojo, se ejecuta la acción de huida
        if key == 'red':
            if current_time - last_red_detection_time < NEW_DETECTION:
                continue
            last_red_detection_time = current_time
            robobo.stopMotors()
            runAwayFromRed()
            break
        else:
            robobo.sayText("Color: " + key)
    robobo.resetColorBlobs()

if __name__ == "__main__":
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(110, 5)
    robobo.setActiveBlobs(True, True, True, False)
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)

    try:
        robobo.moveWheels(5, -5)
        while True:
            time.sleep(TIME)
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()

