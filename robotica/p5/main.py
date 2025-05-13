from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR

from behaviours import AvoidFall, AvoidObstacle, FindColor, PushColor
# Ya no importamos FindLight

import time, random

def main():
    # Conectar con el robot
    robobo = Robobo("localhost")
    robobo.connect()
    
    # Inicializar cámara y sensores
    robobo.setActiveBlobs(True, False, False, False)  # Activar detección de color verde
    robobo.moveTiltTo(105, 10)  # Posicionar la cámara para mejor visión

    # Diccionario para controlar la finalización y compartir información
    params = {
        "stop": False,
        "blob_detected": False,
        "blob_centeRED": False,
        "was_pushing": False,
        "color": BlobColor.RED,
        "last_turn_direction": 1,  # Para alternar giros al evitar obstáculos
        "last_activity_time": time.time()  # Tiempo de la última actividad
    }

    # Crear comportamientos (del menos prioritario al más prioritario)
    # Ya no tenemos FindLight
    find_color = FindColor(robobo, [], params, BlobColor.RED)
    push_color = PushColor(robobo, [find_color], params, BlobColor.RED)
    avoid_obstacle = AvoidObstacle(robobo, [push_color, find_color], params)
    avoid_fall = AvoidFall(robobo, [avoid_obstacle, push_color, find_color], params)

    # Lista de comportamientos para gestión
    threads = [find_color, push_color, avoid_obstacle, avoid_fall]

    # Mostrar la jerarquía de comportamientos
    print("\n=== JERARQUÍA DE COMPORTAMIENTOS ===")
    print("1. AvoidFall (mayor prioridad)")
    print("2. AvoidObstacle")
    print("3. PushColor")
    print("4. FindColor (menor prioridad)")
    print("=====================================\n")

    # Iniciar todos los comportamientos
    print("Iniciando todos los comportamientos...")
    for thread in threads:
        thread.start()
        time.sleep(0.2)  # Pequeña pausa entre inicios para evitar conflictos

    # Mantener el programa principal en ejecución
    try:
        print("\nRobot listo y en funcionamiento. Presiona Ctrl+C para finalizar.")
        
        # Tiempo máximo de ejecución (opcional, establecer a None para ejecutar indefinidamente)
        max_runtime = 180  # Segundos (3 minutos)
        start_time = time.time()
        
        # Tiempo máximo que permitimos que el robot esté sin cambios
        max_inactive_time = 10.0  # Segundos
        last_ir_values = [0, 0, 0]
        last_movement_time = time.time()
        
        while not params["stop"]:
            # Mostrar información de diagnóstico cada segundo
            time.sleep(1)
            
            # Mostrar lecturas de sensores relevantes
            ir_front_c = robobo.readIRSensor(IR.FrontC)
            ir_front_l = robobo.readIRSensor(IR.FrontL)
            ir_front_r = robobo.readIRSensor(IR.FrontR)
            brightness = robobo.readBrightnessSensor()
            
            blob = robobo.readColorBlob(BlobColor.RED)
            blob_info = f"Tamaño: {blob.size}, Posición X: {blob.posx}" if blob.size > 0 else "No detectado"
            
            print(f"Estado - IR(C/L/R): {ir_front_c}/{ir_front_l}/{ir_front_r}, " 
                f"Blob verde: {blob_info}, Brillo: {brightness}, "
                f"Empujando: {'Sí' if params['was_pushing'] else 'No'}")
            
            # Comprobación de si el robot está atascado (valores IR no cambian durante mucho tiempo)
            current_ir = [ir_front_c, ir_front_l, ir_front_r]
            
            # Determinar si hay cambio significativo en los sensores
            ir_changed = any(abs(current_ir[i] - last_ir_values[i]) > 50 for i in range(3))
            
            if ir_changed:
                last_movement_time = time.time()
                last_ir_values = current_ir.copy()
            # Comprobar si el robot está atascado (no hay cambios en valores IR por un tiempo)
            if time.time() - last_movement_time > max_inactive_time:
                print("\n¡ALERTA! Robot posiblemente atascado. Reiniciando comportamientos...")
                
                # Intentar recuperación: parar motores y liberar todos los supress
                robobo.stopMotors()
                
                # Liberar el supress de todos los comportamientos
                for thread in threads:
                    for bh in thread.supress_list:
                        bh.supress = False
                    thread.supress = False
                
                # Hacer un movimiento aleatorio para intentar desatascarse
                robobo.moveWheelsByTime(-25, -25, 1.0)  # Retroceder
                
                if random.random() < 0.5:
                    robobo.moveWheelsByTime(-20, 20, 1.2)  # Girar a la derecha
                else:
                    robobo.moveWheelsByTime(20, -20, 1.2)  # Girar a la izquierda
                
                # Actualizar tiempo de último movimiento para evitar repetir inmediatamente
                last_movement_time = time.time()
                
                # Restaurar valores iniciales importantes
                params["was_pushing"] = False
                
            # Comprobar tiempo máximo de ejecución
            if max_runtime is not None and (time.time() - start_time) > max_runtime:
                print("Tiempo máximo de ejecución alcanzado.")
                params["stop"] = True
                
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por teclado")
    finally:
        # Finalizar todos los comportamientos
        print("\nDeteniendo todos los comportamientos...")
        params["stop"] = True
        
        # Esperar a que terminen todos los threads
        for thread in threads:
            thread.join(timeout=1.0)
        
        # Desconectar
        robobo.stopMotors()
        robobo.disconnect()
        print("Programa finalizado correctamente.")

if __name__ == "__main__":
    main()