from behaviours.behaviour import Behaviour
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR
import time
from robobopy.utils.Wheels import Wheels

class FindColor(Behaviour):
    def __init__(self, robot, supress_list, params, color=BlobColor.RED):
        super().__init__(robot, supress_list, params)
        self.color = color
        self.center_x = 50  # Centro de la imagen en x
        self.center_threshold = 10  # Umbral para considerar centrado
        self.search_speed = 5  # Velocidad de giro durante la búsqueda
        self.last_error = 0  # Almacena el último error para determinar dirección de búsqueda
        self.last_direction_change = 0  # Tiempo del último cambio de dirección
        self.search_start_time = None  # Inicializado en None para control
        self.search_direction = 1  # Dirección de búsqueda inicial
        
    def take_control(self):
        if not self.supress:
            # No tomar el control si sabemos que el blob está justo delante (pero no visible)
            was_pushing = self.params.get("was_pushing", False)
            ir_value = self.robot.readIRSensor(IR.FrontC)
            if was_pushing and ir_value > 100:
                # El blob probablemente está delante pero fuera del campo visual
                return False
                
            return True  # En otros casos, siempre queremos buscar o seguir el blob

    def action(self):
        print("----> control: FindColor")
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True
        
        search_mode = False  # Si estamos en modo búsqueda
        max_search_time = 7.0  # Tiempo máximo de búsqueda (segundos)
        
        while not self.supress and not self.stopped():
            current_time = time.time()
            blob = self.robot.readColorBlob(self.color)
            
            # Actualizar variables compartidas
            self.params["blob_detected"] = blob.size > 0
            self.params["blob_centered"] = blob.size > 0 and abs(blob.posx - self.center_x) <= self.center_threshold
            
            if blob.size > 0:  # Blob detectado
                # Salir del modo búsqueda
                search_mode = False
                self.search_start_time = None  # Resetear timer de búsqueda
                
                # Calcular el error en el centrado
                error = blob.posx - self.center_x
                self.last_error = error  # Guardar el error para búsqueda futura
                
                if abs(error) <= self.center_threshold:
                    # El blob está centrado, avanzar hacia él con aceleración suave
                    for speed_factor in [0.6, 0.8, 1.0]:
                        current_speed = int(25 * speed_factor)
                        # Usar orden correcto: moveWheels(derecha, izquierda)
                        self.robot.moveWheels(current_speed, current_speed)
                        time.sleep(0.1)
                elif error < 0:
                    # El blob está a la izquierda, girar a la izquierda con velocidad proporcional al error
                    turn_speed = min(15, max(5, abs(error) / 4))
                    # Usar orden correcto: moveWheels(derecha, izquierda)
                    self.robot.moveWheels(turn_speed, -turn_speed)
                else:
                    # El blob está a la derecha, girar a la derecha con velocidad proporcional al error
                    turn_speed = min(15, max(5, abs(error) / 4))
                    # Usar orden correcto: moveWheels(derecha, izquierda)
                    self.robot.moveWheels(-turn_speed, turn_speed)
            else:  # Blob no detectado
                if not search_mode:
                    print("Blob perdido, iniciando búsqueda simplificada")
                    self.robot.stopMotors()
                    search_mode = True
                    self.search_start_time = current_time  # Importante: establecer tiempo de inicio
                    
                    # Determinar dirección de giro basado en último error
                    if self.last_error < 0:
                        self.search_direction = -1  # Buscar hacia la izquierda
                    else:
                        self.search_direction = 1   # Buscar hacia la derecha
                
                # Tiempo transcurrido en búsqueda (comprobando que search_start_time exista)
                if self.search_start_time is not None:
                    search_time = current_time - self.search_start_time
                    
                    if search_time < max_search_time:
                        # Simplemente girar en la dirección elegida con velocidad constante
                        search_speed = self.search_speed * 2  # Un poco más rápido para cubrir más área
                        # Usar orden correcto: moveWheels(derecha, izquierda)
                        self.robot.moveWheels(-search_speed * self.search_direction, 
                                            search_speed * self.search_direction)
                        print(f"Buscando blob - Tiempo: {search_time:.1f}s / {max_search_time:.1f}s")
                    else:
                        # Al finalizar el tiempo, cambiar estrategia
                        print("Tiempo de búsqueda agotado")
                        self.robot.stopMotors()
                        
                        # Avanzar un poco y reiniciar búsqueda en dirección contraria
                        self.robot.moveWheelsByTime(10, 10, 0.8)
                        self.search_start_time = current_time  # Actualizar tiempo de inicio
                        self.search_direction *= -1  # Cambiar dirección
            
            time.sleep(0.1)  # Pequeño delay para no saturar la CPU
        
        # Desacelerar suavemente al perder el control
        current_speeds = [self.robot.readWheelSpeed(Wheels.R), self.robot.readWheelSpeed(Wheels.L)]
        if current_speeds[0] != 0 or current_speeds[1] != 0:
            for i in range(3):
                right_speed = int(current_speeds[0] * (1 - (i+1)/4))
                left_speed = int(current_speeds[1] * (1 - (i+1)/4))
                # Usar orden correcto: moveWheels(derecha, izquierda)
                self.robot.moveWheels(right_speed, left_speed)
                time.sleep(0.1)
        
        self.robot.stopMotors()