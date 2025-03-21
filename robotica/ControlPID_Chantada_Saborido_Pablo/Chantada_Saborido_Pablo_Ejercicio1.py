from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time


TIME = 0.5
IP = 'localhost'

KPav = 0.2 
KPct = 0.25
task_completed = False
error_avanzar_previo = 0
CENTER = 50
ROTATION_SPEED = 10
ERROR_MARGIN_center = 10 
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
    Centra el blob usando control proporcional para la velocidad de giro
    '''
    
    # Si el blob no es visible, retornamos
    if color_blob.size <= 0:
        return False
    
    print(f"Centrando blob en posición {color_blob.posx}")
    
    error_centrar = color_blob.posx - CENTER  # [0,100] - 50 = [-50,50]
    
    # Si está fuera del margen de error, aplicamos control proporcional al giro
    if abs(error_centrar) > ERROR_MARGIN_center:
        # Calculamos la velocidad de giro proporcionalmente al error
        # Cuanto mayor sea el error, mayor será la velocidad de giro
        turn_speed = min(abs(error_centrar) * KPct, ROTATION_SPEED)
        turn_speed = max(turn_speed, 3)  # Velocidad mínima para garantizar movimiento
        
        if error_centrar > 0:  # Blob está a la derecha
            print(f"Blob a la derecha, error: {error_centrar}, velocidad: {turn_speed}")
            robobo.moveWheels(turn_speed, -turn_speed)  # Giro a la izquierda
        else:  # Blob está a la izquierda
            print(f"Blob a la izquierda, error: {error_centrar}, velocidad: {turn_speed}")
            robobo.moveWheels(-turn_speed, turn_speed)  # Giro a la derecha
        
        return False  # No está centrado todavía
    else:
        print("Blob centrado")
        robobo.stopMotors()
        return True  # Está centrado


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
    is_centered = centerToAColor(color_blob)
    
    # Solo nos movemos hacia él si está centrado
    if is_centered:
        moveToAColor(color_blob)


if __name__ == "__main__":
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    robobo.moveTiltTo(120, 5) # Inclinamos la cámara hacia abajo
    robobo.setActiveBlobs(True, False, False, False) 
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)  

    try:
        robobo.moveWheels(searchSpeed, -searchSpeed)  
        while not task_completed:
            time.sleep(0.1)  
        print("Task completed! Robot has pushed the blob.")
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
