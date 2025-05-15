from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR
from behaviours import AvoidFall, AvoidObstacle, FindColor, PushColor
import time
import random
import sys
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color

def main():
    """
    Función principal que inicializa y ejecuta el robot con arquitectura subsumida.
    """

    #----------------------------------------
    # INICIALIZACIÓN
    #----------------------------------------
    
    # Inicializar cámara y sensores
    robobo.setActiveBlobs(True, False, False, False)  # Activar detección de blob verde
    # robobo.moveTiltTo(105, 10)  
    # Desactivar led's para evitar reflejos
    robobo.setLedColorTo(LED.All, Color.OFF)
    # Establecer el color objetivo 
    target_color = BlobColor.RED 
    
    # Diccionario para controlar supresión y compartir información entre comportamientos
    params = {
        "stop": False,
        "was_pushing": False,
        "pushing_to_light": False,
        "push_count": 0,
        "last_push_time": 0,
        "color": target_color,  
        "light_threshold": 800,
        "mission_complete": False
    }

    #----------------------------------------
    # CREACIÓN DE COMPORTAMIENTOS
    #----------------------------------------
    
    # Crear comportamientos (del menos al más prioritario)
    find_color = FindColor(robobo, [], params, target_color)
    push_color = PushColor(robobo, [find_color], params, target_color)
    avoid_obstacle = AvoidObstacle(robobo, [push_color, find_color], params)
    avoid_fall = AvoidFall(robobo, [avoid_obstacle, push_color, find_color], params)

    # Lista de comportamientos para gestión
    threads = [find_color, push_color, avoid_obstacle, avoid_fall]

    print("\n=== JERARQUÍA DE COMPORTAMIENTOS ===")
    print("1. AvoidFall (máxima prioridad)")
    print("2. AvoidObstacle")
    print("3. PushColor")
    print("4. FindColor (mínima prioridad)")
    print("===========================\n")

    # Obtener nombre del color objetivo 
    color_name = "Verde" if target_color == BlobColor.GREEN else "Rojo"

    #----------------------------------------
    # EJECUCIÓN PRINCIPAL
    #----------------------------------------
    
    print("Iniciando todos los comportamientos...")
    print(f"Color objetivo: {color_name}")
    for thread in threads:
        thread.start()
        time.sleep(0.2)  # Pequeña pausa entre inicios para evitar conflictos

    robobo.moveTiltTo(105 , 100)

    # Mantener el programa principal en ejecución
    try:
        print("\nRobot listo. Presiona Ctrl+C para finalizar.")
        robobo.wait(3)
        # Tiempo máximo que permitimos que el robot esté sin cambios
        max_inactive_time = 20.0  # Segundos
        last_ir_values = [0, 0, 0]
        last_movement_time = time.time()
        
        # Mientras no se indique una parada
        while not params["stop"]:
            # Mostrar información de diagnóstico cada segundo
            time.sleep(1)
            
            # Mostrar lecturas de sensores relevantes
            ir_front_c = robobo.readIRSensor(IR.FrontC)
            ir_front_l = robobo.readIRSensor(IR.FrontL)
            ir_front_r = robobo.readIRSensor(IR.FrontR)
            brightness = robobo.readBrightnessSensor()
            
            blob = robobo.readColorBlob(target_color)
            blob_info = f"Tamaño: {blob.size}, Posición X: {blob.posx}" if blob.size > 0 else "No detectado"
            
            print(f"Estado - IR(C/L/R): {ir_front_c}/{ir_front_l}/{ir_front_r}, " 
                f"Blob {color_name}: {blob_info}, Brillo: {brightness}")
            
            # Finalizar al llegar al umbral indicado
            if brightness >= params["light_threshold"]:
                print("\n!!! UMBRAL DE LUZ ALCANZADO !!!")
                print(f"Alcanzada fuente de luz con brillo: {brightness}")
                print("Misión completada: Deteniendo todos los comportamientos.")
                
                # Detener el robot
                robobo.stopMotors()
                
                # Establecer las flags para finalizar la ejecución
                params["mission_complete"] = True
                params["stop"] = True
                break
            
            # Comprobar si el robot está atascado (valores IR no cambian durante mucho tiempo)
            current_ir = [ir_front_c, ir_front_l, ir_front_r]
            
            # Determinar si hay cambio significativo en sensores
            ir_changed = any(abs(current_ir[i] - last_ir_values[i]) > 50 for i in range(3))
            
            if ir_changed:
                last_movement_time = time.time()
                last_ir_values = current_ir.copy()
                
            if time.time() - last_movement_time > max_inactive_time:
                print("\nRobot posiblemente atascado. Reiniciando comportamientos...")
                
                # Intento de recuperación: detener motores y liberar todas las supresiones
                robobo.stopMotors()
                
                # Liberar supresión para todos los comportamientos
                for thread in threads:
                    for bh in thread.supress_list:
                        bh.supress = False
                    thread.supress = False
                
                # Hacer un movimiento aleatorio para intentar desatascarse
                robobo.moveWheelsByTime(-25, -25, 1.0)  # Retroceder
                
                if random.random() < 0.5:
                    robobo.moveWheelsByTime(20, -20, 1.2)  # Girar a la derecha
                else:
                    robobo.moveWheelsByTime(-20, 20, 1.2)  # Girar a la izquierda
                
                # Actualizar último tiempo de movimiento para evitar repetición inmediata
                last_movement_time = time.time()
                
                # Reiniciar valores iniciales importantes
                params["was_pushing"] = False
                
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por teclado")
    finally:
        # Finalizar todos los comportamientos
        print("\nDeteniendo todos los comportamientos...")
        params["stop"] = True
        
        # Esperar a que todos los hilos terminen
        for thread in threads:
            thread.join(timeout=0.5)
        
        # Desconectar
        robobo.stopMotors()
        robobo.disconnect()
        
        if params.get("mission_complete", False):
            print("Programa finalizado con éxito: ¡Misión completa!")
        else:
            print("Programa finalizado correctamente. Misión incompleta")

if __name__ == "__main__":

    IP = "10.56.43.36"

    robobo = Robobo(IP)
    robobo.connect()
    print("Conectado a Robobo")
    main()