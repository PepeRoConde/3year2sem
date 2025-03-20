from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

TIME = 0.5
IP = 'localhost'

# Constantes de control PID ajustadas para movimiento más suave
KPav = 0.05  # Reducido para movimiento más suave
KPct = 0.08  # Reducido para giros más suaves
KD = 0.5     # Aumentado para mayor amortiguación
KI = 0.02    # Reducido para evitar oscilaciones
Iav = 0      # para el control del AVance
Ict = 0      # para el control del CenTro
task_completed = False
error_avanzar_previo = 0
error_centrar_previo = 0
CENTER = 50
ROTATION_SPEED = 20
ERROR_MARGIN_center = 10
ERROR_MARGIN_avance = 100
searchSpeed = 10
speedAvance = 20

# Variables para implementar aceleración gradual
current_left_speed = 0
current_right_speed = 0
acceleration_rate = 0.8  # Tasa de aceleración (0-1)

# Constante para manejar la inversión de la cámara
CAMERA_INVERTED = True  # Establecemos que la cámara está en espejo

def smooth_move(target_left, target_right):
    """
    Ajusta gradualmente la velocidad actual hacia la velocidad objetivo
    para un movimiento más suave
    """
    global current_left_speed, current_right_speed
    
    # Calcular la diferencia entre la velocidad actual y la objetivo
    diff_left = target_left - current_left_speed
    diff_right = target_right - current_right_speed
    
    # Aplicar aceleración gradual
    current_left_speed += diff_left * acceleration_rate
    current_right_speed += diff_right * acceleration_rate
    
    # Aplicar las nuevas velocidades
    robobo.moveWheels(round(current_left_speed), round(current_right_speed))

def applyCorrection(speed, correction):
    '''
    Devuelve el maximo de la velocidad corregida, o un valor base (en este caso 3).
    '''
    return max(speed - correction, 3)  # Valor base reducido para movimientos más suaves


def moveToAColor(color_blob):
    '''
    Mueve el robot hasta llegar a una distancia indicada
    '''
    global Iav, error_avanzar_previo, speedAvance, task_completed

    error_avanzar = robobo.readIRSensor(IR.FrontC)

    if error_avanzar >= ERROR_MARGIN_avance:
        robobo.moveTiltTo(190, 4)  # Velocidad de inclinación reducida
        blob_is_close(distance=ERROR_MARGIN_avance)
        return

    # Si perdimos el blob, regresamos para buscar de nuevo
    if color_blob.size <= 0:
        return
        
    # Calculamos la corrección para el avance con límite en el término integral
    P = error_avanzar
    D = error_avanzar - error_avanzar_previo
    Iav += error_avanzar
    
    # Limitar el componente integral para evitar acumulación excesiva
    Iav = max(min(Iav, 100), -100)

    correction = round(P * KPav + D * KD + Iav * KI)
    speed = applyCorrection(speedAvance, correction)
    
    # Movemos el robot hacia adelante con aceleración gradual
    smooth_move(speed, speed)
    
    error_avanzar_previo = error_avanzar


def centerToAColor(color_blob):
    '''
    Centra el blob usando control proporcional
    Ajustado para cámara en espejo
    '''
    global Ict, error_centrar_previo
    
    # Si el blob no es visible, retornamos
    if color_blob.size <= 0:
        return
    
    print(f"Centrando blob en posición {color_blob.posx}")
    
    error_centrar = color_blob.posx - CENTER  # [0,100] - 50 = [-50,50]
    
    # Invertimos el error si la cámara está en espejo
    if CAMERA_INVERTED:
        error_centrar = -error_centrar
    
    # Si está fuera del margen de error, corregimos
    if abs(error_centrar) > ERROR_MARGIN_center:
        P = error_centrar
        D = error_centrar - error_centrar_previo
        Ict += error_centrar
        
        # Limitar el componente integral para evitar acumulación excesiva
        Ict = max(min(Ict, 100), -100)

        # Usar función no lineal para suavizar respuesta en errores pequeños
        P_factor = abs(error_centrar) / 50.0  # Normalizado entre 0 y 1 aproximadamente
        effective_KP = KPct * (0.5 + P_factor)  # Ajuste dinámico del KP

        correction = round(P * effective_KP + D * KD + Ict * KI)
        speed = applyCorrection(ROTATION_SPEED, correction)

        if error_centrar > 0:  # Blob está a la derecha (ajustado para espejo)
            print(f"Blob a la derecha, error: {error_centrar}")
            smooth_move(speed, -speed)
        else:  # Blob está a la izquierda (ajustado para espejo)
            print(f"Blob a la izquierda, error: {error_centrar}")
            smooth_move(-speed, speed)
        
        # Tiempo de espera reducido para actualizar más frecuentemente
        robobo.wait(0.05)
    else:
        print("Blob centrado")
        smooth_move(0, 0)  # Desacelerar suavemente hasta detenerse
    
    # Guardar el error actual como previo para la próxima iteración
    # Nota: guardamos el error invertido si la cámara está en espejo
    error_centrar_previo = error_centrar

def blobDetectedCallback():
    '''
    Detecta un color, lo centra en su cámara y se mueve hasta él sin llegar a chocar.
    '''
    global task_completed
    
    # Si ya completamos la tarea, no hacemos nada
    if task_completed:
        return
        
    # Reducir velocidad gradualmente en lugar de parar bruscamente
    smooth_move(0, 0)
    
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

def blob_is_close(speed=6, distance=1000):
    '''
    Si esta tocando al blob, gira hacia la derecha para mover el objeto
    '''
    global task_completed
    ir_value = max(
        robobo.readIRSensor(IR.FrontC),
        robobo.readIRSensor(IR.FrontRR),
        robobo.readIRSensor(IR.FrontLL),
    )
    
    # Al mover el blob deberia tambien moverse el robot para que estea mas pegado
    if ir_value > distance:
        print(f"Blob detected at distance {ir_value}, pushing now")
        # Nos movemos para hacer contacto con el blob - aceleración gradual
        for i in range(10):
            speed_factor = (i + 1) / 5.0  # Incremento gradual
            smooth_move(6 * speed_factor, 6 * speed_factor)
            time.sleep(0.15)
        
        # Rotamos para mover el objeto - con movimiento suave
        for i in range(10):
            speed_factor = (i + 1) / 10.0  # Incremento gradual
            smooth_move(-speed * speed_factor, speed * speed_factor)
            time.sleep(0.5)
        
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
    robobo.moveTiltTo(120, 4)  # Velocidad reducida para movimiento más suave
    robobo.setActiveBlobs(True, False, False, False)  
    robobo.whenANewColorBlobIsDetected(blobDetectedCallback)

    try:
        # Iniciar la búsqueda con aceleración gradual
        for i in range(10):
            speed_factor = (i + 1) / 10.0  # Incremento gradual
            # Si la cámara está en espejo, podemos también invertir la dirección inicial de búsqueda
            if CAMERA_INVERTED:
                smooth_move(-searchSpeed * speed_factor, searchSpeed * speed_factor)
            else:
                smooth_move(searchSpeed * speed_factor, -searchSpeed * speed_factor)
            time.sleep(0.1)
            
        while not task_completed:
            time.sleep(0.1)  
        print("Task completed! Robot has pushed the blob.")
        
        # Desacelerar suavemente al finalizar
        smooth_move(0, 0)
        
    except KeyboardInterrupt:
        smooth_move(0, 0)  # Desacelerar suavemente al interrumpir
        sim.disconnect()
