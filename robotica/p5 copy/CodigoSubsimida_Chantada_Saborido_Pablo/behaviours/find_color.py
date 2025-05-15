from behaviours.behaviour import Behaviour
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR
import time
from robobopy.utils.Wheels import Wheels

class FindColor(Behaviour):
    """
    Comportamiento para encontrar y centrar objetos de colores.
    """
    def __init__(self, robot, supress_list, params, color=BlobColor.RED):
        super().__init__(robot, supress_list, params)
        self.color = color
        self.center_x = 50  # Centro de la imagen en x
        self.center_threshold = 10  # Umbral para considerar centrado
        self.search_speed = 5  # Velocidad de giro durante búsqueda
        self.last_error = 0  # Almacena el último error para determinar dirección de búsqueda
        self.search_start_time = None  # Inicializado como None para control
        self.search_direction = 1  # Dirección inicial de búsqueda
        
    #----------------------------------------
    # CONTROL DE ACTIVACIÓN
    #----------------------------------------
    
    def take_control(self):
        """
        Toma el control si no está suprimido por comportamientos de mayor prioridad.
        """
        if not self.supress:
            # No tomar el control si sabemos que el blob está justo delante (pero no visible)
            was_pushing = self.params.get("was_pushing", False)
            ir_value = self.robot.readIRSensor(IR.FrontC)
            if was_pushing and ir_value > 100:
                # El blob probablemente está delante pero fuera de la camara
                return False
            
            # En otros casos, siempre queremos buscar o seguir el blob
            return True  

    #----------------------------------------
    # ACCIÓN PRINCIPAL
    #----------------------------------------
    
    def action(self):
        """
        Ejecuta comportamiento de búsqueda y centrado de blobs.
        """
        print("----> control: FindColor")
        # Desactivar otros comportamientos
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True
        
        search_mode = False  # Si estamos en modo búsqueda
        max_search_time = 10.0  # Tiempo máximo de búsqueda (segundos)
        
        # Mantener el comportamiento activo hasta que se detenga o se suprima
        while not self.supress and not self.stopped():
            current_time = time.time()
            blob = self.robot.readColorBlob(self.color)
            
            # Actualizar variables compartidas
            self.params["blob_detected"] = blob.size > 0
            self.params["blob_centered"] = blob.size > 0 and abs(blob.posx - self.center_x) <= self.center_threshold
            
            if blob.size > 0:  # Blob detectado
                # Salir del modo búsqueda
                search_mode = False
                self.search_start_time = None  # Reiniciar temporizador de búsqueda
                
                # Calcular error de centrado
                error = blob.posx - self.center_x
                self.last_error = error  # Guardar error para búsqueda futura
                
                if abs(error) <= self.center_threshold:
                    # Blob está centrado, moverse hacia él con aceleración suave
                    for speed_factor in [0.6, 0.8, 1.0]:
                        current_speed = int(10 * speed_factor)
                        # Usar orden correcto: moveWheels(derecha, izquierda)
                        self.robot.moveWheels(current_speed, current_speed)
                        time.sleep(0.1)
                elif error < 0:
                    # Blob está a la izquierda, girar a la izquierda con velocidad proporcional al error
                    turn_speed = min(10, max(5, abs(error) / 4))
                    self.robot.moveWheels(-turn_speed, turn_speed)
                else:
                    # Blob está a la derecha, girar a la derecha con velocidad proporcional al error
                    turn_speed = min(10, max(5, abs(error) / 4))
                    self.robot.moveWheels(turn_speed, -turn_speed)
            else:  # Blob no detectado
                if not search_mode:
                    print("Blob perdido, iniciando búsqueda simplificada")
                    self.robot.stopMotors()
                    search_mode = True
                    self.search_start_time = current_time  
                    
                    # Determinar dirección de giro basada en el último error
                    if self.last_error < 0:
                        self.search_direction = -1  # Buscar hacia la izquierda
                    else:
                        self.search_direction = 1   # Buscar hacia la derecha
                
                # Tiempo transcurrido en búsqueda 
                if self.search_start_time is not None:
                    # Calcular tiempo de búsqueda
                    search_time = current_time - self.search_start_time
                    
                    if search_time < max_search_time:
                        # Simplemente girar en la dirección elegida a velocidad constante
                        search_speed = self.search_speed  # Un poco más rápido para cubrir más área
                        self.robot.moveWheels(-search_speed * self.search_direction, 
                                            search_speed * self.search_direction)
                        print(f"Buscando blob - Tiempo: {search_time:.1f}s / {max_search_time:.1f}s")
                    else:
                        # Cuando se acaba el tiempo, cambiar estrategia
                        print("Tiempo de búsqueda agotado")
                        self.robot.stopMotors()
                        
                        # Avanzar un poco y reiniciar búsqueda en dirección opuesta
                        self.robot.moveWheelsByTime(10, 10, 1.5)
                        self.search_start_time = current_time  # Actualizar tiempo de inicio
                        self.search_direction *= -1  # Cambiar dirección
            
            time.sleep(0.1) 
        
        # Desacelerar suavemente al perder el control
        current_speeds = [self.robot.readWheelSpeed(Wheels.R), self.robot.readWheelSpeed(Wheels.L)]
        if current_speeds[0] != 0 or current_speeds[1] != 0:
            for i in range(3):
                right_speed = int(current_speeds[0] * (1 - (i+1)/4))
                left_speed = int(current_speeds[1] * (1 - (i+1)/4))

                self.robot.moveWheels(right_speed, left_speed)
                time.sleep(0.1)
        
        self.robot.stopMotors()