from behaviours.behaviour import Behaviour
from robobopy.utils.IR import IR
import time

class AvoidObstacle(Behaviour):
    """
    Comportamiento para evitar obstáculos.
    """
    def __init__(self, robot, supress_list, params, blob_threshold=15):
        super().__init__(robot, supress_list, params)
        # Parámetros de configuración
        self.obstacle_distance = 60  # Umbral para detectar obstáculos
        self.rear_obstacle_distance = 150  # Umbral para detectar obstáculos traseros
        self.blob_threshold = blob_threshold  # Umbral para considerar un blob centrado
        self.max_control_time = 5.0  # Tiempo máximo de control en segundos (aumentado)
        self.avoid_in_progress = False  # Bandera para indicar si una maniobra de evitación está en progreso
        self.avoid_start_time = 0  # Tiempo de inicio de la maniobra
        self.min_avoid_time = 2.0  # Tiempo mínimo que debe durar una maniobra de evitación
        
    #----------------------------------------
    # CONTROL DE ACTIVACIÓN
    #----------------------------------------
    
    def take_control(self):
        """
        Determina si este comportamiento debe tomar el control basado en lecturas IR
        y el estado del blob.
        """
        # Si una maniobra está en progreso, mantener el control hasta que esté completa
        if self.avoid_in_progress and time.time() - self.avoid_start_time < self.min_avoid_time:
            return True
            
        if not self.supress:
            # Leer sensores frontales
            front_c = self.robot.readIRSensor(IR.FrontC)
            front_ll = self.robot.readIRSensor(IR.FrontLL)
            front_rr = self.robot.readIRSensor(IR.FrontRR)
            
            # Leer sensores traseros
            back_c = self.robot.readIRSensor(IR.BackC)
            back_l = self.robot.readIRSensor(IR.BackL)
            back_r = self.robot.readIRSensor(IR.BackR)
            
            # Comprobar si un blob está centrado 
            blob_centered = self.params.get("blob_centered", False)
            was_pushing = self.params.get("was_pushing", False)
            
            # Detectar obstáculos delante y atrás
            front_obstacle_detected = (front_c >= self.obstacle_distance or 
                                     front_ll >= self.obstacle_distance or 
                                     front_rr >= self.obstacle_distance)
            
            rear_obstacle_detected = (back_c >= self.rear_obstacle_distance or 
                                    back_l >= self.rear_obstacle_distance or 
                                    back_r >= self.rear_obstacle_distance)
            
            # Si estamos en modo "push to light", solo considerar sensores laterales para detección frontal
            if self.params.get("pushing_to_light", False) and front_c >= self.obstacle_distance:
                # Si el centro detecta un objeto (nuestro blob), solo evitar si los sensores laterales detectan obstáculos
                front_obstacle_detected = front_ll >= self.obstacle_distance or front_rr >= self.obstacle_distance
            
            # No evitar obstáculos frontales si son el blob que queremos empujar
            if front_obstacle_detected:
                # Si el blob está centrado y solo el sensor central detecta algo, probablemente sea el blob
                if (blob_centered or was_pushing) and front_c >= self.obstacle_distance and front_ll < self.obstacle_distance and front_rr < self.obstacle_distance:
                    front_obstacle_detected = False
                # Si estamos en modo push, priorizar empujar sobre evitar obstáculos laterales
                # pero seguir evitando obstáculos que están muy cerca
                elif was_pushing and front_c >= 500 and (front_ll < 100 and front_rr < 100):
                    front_obstacle_detected = False
            
            # Tomar el control si se detecta algún obstáculo (frontal o trasero)
            if front_obstacle_detected or rear_obstacle_detected:
                # Nueva maniobra de evitación, marcar inicio
                self.avoid_in_progress = True
                self.avoid_start_time = time.time()
                if rear_obstacle_detected:
                    print(f"Obstáculo trasero detectado - Sensores: C={back_c}, L={back_l}, R={back_r}")
                return True
            
            # Si no hay obstáculo, terminar la maniobra en progreso
            if self.avoid_in_progress:
                if time.time() - self.avoid_start_time >= self.min_avoid_time:
                    self.avoid_in_progress = False
                    
            return False

    #----------------------------------------
    # ACCIÓN PRINCIPAL
    #----------------------------------------
    
    def action(self):
        """
        Ejecuta el comportamiento de evitación de obstáculos.
        """
        print("----> control: AvoidObstacle")
        # Marcar que la maniobra de evitación está en progreso
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True
        
        # Registrar tiempo de inicio para control de tiempo máximo
        start_time = time.time()
        
        try:
            # Determinar dirección del obstáculo - sensores frontales
            front_c = self.robot.readIRSensor(IR.FrontC)
            front_ll = self.robot.readIRSensor(IR.FrontLL)
            front_rr = self.robot.readIRSensor(IR.FrontRR)
            
            # Determinar dirección del obstáculo - sensores traseros
            back_c = self.robot.readIRSensor(IR.BackC)
            back_l = self.robot.readIRSensor(IR.BackL)
            back_r = self.robot.readIRSensor(IR.BackR)
            
            # Determinar si el obstáculo está delante o detrás
            front_obstacle = (front_c >= self.obstacle_distance or 
                             front_ll >= self.obstacle_distance or 
                             front_rr >= self.obstacle_distance)
            
            rear_obstacle = (back_c >= self.rear_obstacle_distance or 
                            back_l >= self.rear_obstacle_distance or 
                            back_r >= self.rear_obstacle_distance)
            
            # Si hay obstáculos tanto delante como detrás, priorizar obstáculo frontal
            if front_obstacle:
                # Determinar ubicación más específica del obstáculo frontal
                center_obstacle = front_c >= self.obstacle_distance
                left_obstacle = front_ll >= self.obstacle_distance
                right_obstacle = front_rr >= self.obstacle_distance
                
                print(f"Obstáculo frontal detectado - C={front_c}, LL={front_ll}, RR={front_rr}")
                
                # Movimientos más pronunciados para evitación
                if center_obstacle and not (left_obstacle or right_obstacle):
                    # Si el obstáculo solo está en el centro, verificar sensores traseros antes de retroceder
                    if rear_obstacle:
                        print("¡Obstáculo detectado detrás! No se puede retroceder con seguridad.")
                        # Intentar girar en el sitio en lugar de retroceder
                        last_turn_direction = self.params.get("last_turn_direction", 1)
                        if last_turn_direction > 0:
                            self.robot.moveWheelsByTime(15, -15, 1.5)  # Girar a la derecha en el sitio
                        else:
                            self.robot.moveWheelsByTime(-15, 15, 1.5)  # Girar a la izquierda en el sitio
                        self.params["last_turn_direction"] = -last_turn_direction
                    else:
                        # Seguro retroceder
                        self.robot.moveWheelsByTime(-20, -20, 1.5)  
                        
                        # Girar basado en dirección histórica o aleatoria
                        last_turn_direction = self.params.get("last_turn_direction", 1)
                        if last_turn_direction > 0:
                            self.robot.moveWheelsByTime(10, -25, 1.2)  # Giro más pronunciado a la derecha
                        else:
                            self.robot.moveWheelsByTime(-25, 10, 1.2)  # Giro más pronunciado a la izquierda
                        
                        # Alternar dirección para la próxima vez
                        self.params["last_turn_direction"] = -last_turn_direction
                else:
                    # Si el obstáculo está en los lados, verificar sensores traseros antes de retroceder
                    if rear_obstacle:
                        print("¡Obstáculo detectado detrás! No se puede retroceder con seguridad.")
                        # Intentar alejarse del obstáculo frontal sin retroceder
                        if left_obstacle and not right_obstacle:
                            self.robot.moveWheelsByTime(10, -20, 1.5)  # Girar a la derecha sin retroceder
                        elif right_obstacle and not left_obstacle:
                            self.robot.moveWheelsByTime(-20, 10, 1.5)  # Girar a la izquierda sin retroceder
                        else:
                            # Ambos lados tienen obstáculos, girar más bruscamente
                            last_turn_direction = self.params.get("last_turn_direction", 1)
                            if last_turn_direction > 0:
                                self.robot.moveWheelsByTime(15, -25, 1.8)  # Giro brusco a la derecha
                            else:
                                self.robot.moveWheelsByTime(-25, 15, 1.8)  # Giro brusco a la izquierda
                            self.params["last_turn_direction"] = -last_turn_direction
                    else:
                        # Seguro retroceder
                        self.robot.moveWheelsByTime(-18, -18, 1.8)  
                        
                        # Giros más pronunciados alejándose del obstáculo
                        if left_obstacle and not right_obstacle:
                            # Obstáculo a la izquierda - giro pronunciado a la derecha
                            self.robot.moveWheelsByTime(5, -30, 1.2)  # Giro más pronunciado
                            self.params["last_turn_direction"] = 1
                        elif right_obstacle and not left_obstacle:
                            # Obstáculo a la derecha - giro pronunciado a la izquierda
                            self.params["last_turn_direction"] = -1
                        else:
                            # Obstáculos en ambos lados, hacer un giro mucho más pronunciado
                            if front_ll > front_rr:
                                self.robot.moveWheelsByTime(12, -28, 1.5)  # Giro más pronunciado a la derecha
                                self.params["last_turn_direction"] = 1
                            else:
                                self.robot.moveWheelsByTime(-28, 12, 1.5)  # Giro más pronunciado a la izquierda
                                self.params["last_turn_direction"] = -1
            
            # Caso de solo obstáculo trasero
            elif rear_obstacle:
                print(f"Obstáculo trasero detectado - C={back_c}, L={back_l}, R={back_r}")
                
                # No retroceder, avanzar en su lugar
                self.robot.moveWheelsByTime(20, 20, 1.5)
                
                # Si el obstáculo está más en un lado, alejarse de ese lado
                if back_l > back_r:
                    # Obstáculo más en la parte trasera izquierda, girar ligeramente a la derecha mientras avanza
                    self.robot.moveWheelsByTime(15, 25, 1.0)
                    self.params["last_turn_direction"] = 1
                elif back_r > back_l:
                    # Obstáculo más en la parte trasera derecha, girar ligeramente a la izquierda mientras avanza
                    self.robot.moveWheelsByTime(25, 15, 1.0)
                    self.params["last_turn_direction"] = -1
            
            # Avanzar un poco después de la maniobra para salir de la zona de peligro
            # Solo si no hay obstáculo trasero
            if not rear_obstacle:
                self.robot.moveWheelsByTime(15, 15, 0.7)
            
            # Asegurar que la maniobra dure al menos el tiempo mínimo
            elapsed_time = time.time() - start_time
            if elapsed_time < self.min_avoid_time:
                time.sleep(self.min_avoid_time - elapsed_time)
            
            # Mantener el control hasta el tiempo máximo para verificar que evitó correctamente
            while time.time() - start_time < self.max_control_time:
                # Comprobar si el obstáculo ha sido completamente evitado
                front_c = self.robot.readIRSensor(IR.FrontC)
                front_ll = self.robot.readIRSensor(IR.FrontLL)
                front_rr = self.robot.readIRSensor(IR.FrontRR)
                back_c = self.robot.readIRSensor(IR.BackC)
                back_l = self.robot.readIRSensor(IR.BackL)
                back_r = self.robot.readIRSensor(IR.BackR)
                
                front_clear = (front_c < self.obstacle_distance - 20 and  # Margen de seguridad adicional
                              front_ll < self.obstacle_distance - 20 and 
                              front_rr < self.obstacle_distance - 20)
                              
                rear_clear = (back_c < self.rear_obstacle_distance - 20 and
                             back_l < self.rear_obstacle_distance - 20 and
                             back_r < self.rear_obstacle_distance - 20)
                
                if front_clear and rear_clear:
                    print("Obstáculos completamente evitados, liberando control")
                    break
                
                time.sleep(0.1)
            
            print("Maniobra de evitación completa, liberando control")
            # Establecer que ya no hay una maniobra en progreso
            self.avoid_in_progress = False
            
            # Liberar la supresión para otros comportamientos
            for bh in self.supress_list:
                bh.supress = False
                
        except Exception as e:
            print(f"Error en AvoidObstacle.action(): {e}")
            self.avoid_in_progress = False  # Asegurar que no se quede atascado
        finally:
            # Asegurar que se libere la supresión en caso de error
            for bh in self.supress_list:
                bh.supress = False
            self.robot.stopMotors()