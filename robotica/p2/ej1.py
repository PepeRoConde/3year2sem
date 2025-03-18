from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

# USAR UN PI -> PRECISION SIN OSCILACIONES GRADNES

TIME = 0.5
IP = 'localhost'

KPav = 0.2 
KPct = 0.25
task_completed = False
error_avanzar_previo = 0
CENTER = 50
ROTATION_SPEED = 10
ERROR_MARGIN_center = 13 
ERROR_MARGIN_avance = 80 
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
        robobo.stopMotors()
        task_completed = True
        return

    # Si perdimos el blob, regresamos para buscar de nuevo
    if color_blob.size <= 0:
        robobo.stopMotors()
        return
        
    
    # Calculamos la corrección para el avance
    P = error_avanzar

    correction = round(P * KPav)
    speed = applyCorrection(speedAvance, correction)
    
    # Movemos el robot hacia adelante
    robobo.moveWheels(speed, speed)

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

if __name__ == "__main__":
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(120, 5) # posición inicial
    robobo.setActiveBlobs(True, False, False, False) 
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)  # Corrección clave

    try:
        robobo.moveWheels(searchSpeed, -searchSpeed)  # Búsqueda giratoria
        while not task_completed:
            time.sleep(0.1)  # Reducir carga de CPU
        print("Task completed! Robot has pushed the blob.")
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
