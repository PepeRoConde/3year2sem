from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

TIME = 0.5
IP = 'localhost'

KPav = 0.2
KPct = 0.25
KD = 0.3  # Variacion respecto al ejercicio 1
KI = 0.05  # Variacion respecto al ejercicio 1
Iav = 0 # para el control del AVance
Ict = 0 # para el control del CenTro
task_completed = False
error_avanzar_previo = 0
CENTER = 50
ROTATION_SPEED = 10
ERROR_MARGIN_center = 13 
ERROR_MARGIN_avance = 60
searchSpeed = 10
speedAvance = 15

def applyCorrection(speed, correction):
    '''
    Devuelve el maximo de la velocidad corregida, o un valor base (en este caso 5).
    '''
    return max(speed - correction, 5)


def moveToAColor(color_blob):
    '''
    Mueve el robot hasta llegar a una distancia indicada
    '''
    global Iav, error_avanzar_previo, speedAvance, task_completed

    error_avanzar = robobo.readIRSensor(IR.FrontC)

    if error_avanzar >= ERROR_MARGIN_avance:
        robobo.moveTiltTo(190, 5)
        blob_is_close(10, distance=ERROR_MARGIN_avance)
        # task_completed = True
        return

    # Si perdimos el blob, regresamos para buscar de nuevo
    if color_blob.size <= 0:
        # robobo.stopMotors()
        return
        
    # Nos aseguramos que el color esté centrado
    if abs(color_blob.posx - CENTER) > ERROR_MARGIN_center:
        centerToAColor(color_blob)
        return  # Volvemos para garantizar que esté centrado antes de avanzar
    
    # Calculamos la corrección para el avance
    P = error_avanzar
    D = error_avanzar - error_avanzar_previo
    Iav += error_avanzar

    correction = round(P * KPav + D * KD + Iav * KI)
    speed = applyCorrection(speedAvance, correction)
    
    # Movemos el robot hacia adelante
    robobo.moveWheels(speed, speed)
    
    error_avanzar_previo = error_avanzar


def centerToAColor(color_blob):
    '''
    Centra el blob usando control proporcional
    '''
    global Ict
    
    # Si el blob no es visible, retornamos
    if color_blob.size <= 0:
        return
    
    print(f"Centrando blob en posición {color_blob.posx}")
    
    error_centrar = color_blob.posx - CENTER  # [0,100] - 50 = [-50,50]
    
    # Si está hacia la derecha, error_centrar es positivo
    if abs(error_centrar) > ERROR_MARGIN_center:
        if error_centrar > 0:  # Blob está a la derecha
            print(f"Blob a la derecha, error: {error_centrar}")
            robobo.moveWheelsByTime(5, -5, 0.3)  # Giro a la izquierda
        else:  # Blob está a la izquierda
            print(f"Blob a la izquierda, error: {error_centrar}")
            robobo.moveWheelsByTime(-5, 5, 0.3)  # Giro a la derecha
        
        # Pequeña pausa para evitar movimientos bruscos
        time.sleep(0.2)
    else:
        print("Blob centrado")
        robobo.stopMotors()

def blobDetectedCallback():
    '''
    Detecta un color, lo centra en su cámara y se mueve hasta él sin llegar a chocar.
    '''
    global task_completed
    
    # Si ya completamos la tarea, no hacemos nada
    if task_completed:
        return
        
    # Paramos los motores para iniciar el proceso de detección y acercamiento
    robobo.stopMotors()
    
    color = BlobColor.RED
    color_blob = robobo.readColorBlob(color)
    
    # Si no detectamos el blob, volvemos a la búsqueda
    if color_blob.size <= 0:
        return
    
    # Procesamos el blob detectado
    print(f"Blob detectado: Tamaño {color_blob.size}, Posición X: {color_blob.posx}")
    
    # Primero centramos el blob
    if abs(color_blob.posx - CENTER) > ERROR_MARGIN_center:
        centerToAColor(color_blob)
    
    # Luego nos movemos hacia él
    moveToAColor(color_blob)

def blob_is_close(speed, distance=1000):
    '''
    Si esta tocando al blob, gira hacia la derecha para mover el objeto
    '''
    global task_completed
    ir_value = max(
        robobo.readIRSensor(IR.FrontC),
        robobo.readIRSensor(IR.FrontRR),
        robobo.readIRSensor(IR.FrontLL),
    )
    
    if ir_value > distance:
        print(f"Blob detected at distance {ir_value}, pushing now")
        # First push forward a bit to make contact
        robobo.moveWheelsByTime(15, 15, 1)
        # Then rotate to move the object
        robobo.moveWheelsByTime(-speed, speed, 5)
        # DESCOMENTAR PARA QUE SOLO SE MUEVA UNA VEZ
        task_completed = True
    else:
        print(f"Blob not close enough: {ir_value} < {distance}")
        task_completed = False


if __name__ == "__main__":
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(120, 5) # posición inicial
    robobo.setActiveBlobs(True, True, False, False)  # Mantén los blobs activados como en tu versión original
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)


    try:
        robobo.moveWheels(searchSpeed, -searchSpeed)  # Búsqueda giratoria
        while not task_completed:
            time.sleep(0.1)  # Reducir carga de CPU
            # print(f"Task status: {task_completed}")  # Debug output
        print("Task completed! Robot has pushed the blob.")
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
