from behaviours.behaviour import Behaviour
from robobopy.utils.IR import IR
from robobopy.utils.Sounds import Sounds
import time

class AvoidObstacle(Behaviour):
    def __init__(self, robot, supress_list, params, blob_threshold=15):
        super().__init__(robot, supress_list, params)
        self.obstacle_distance = 60  # Umbral para detectar obstáculos
        self.blob_threshold = blob_threshold  # Umbral para considerar que un blob está centrado
        self.max_control_time = 5.0  # Aumentado: Tiempo máximo de control en segundos
        self.avoid_in_progress = False  # Flag para indicar si una maniobra de esquiva está en progreso
        self.avoid_start_time = 0  # Tiempo de inicio de la maniobra
        self.min_avoid_time = 2.0  # Tiempo mínimo que debe durar una maniobra de esquiva
        
    def take_control(self):
        # Si hay una maniobra en progreso, mantener el control hasta que esté completa
        if self.avoid_in_progress and time.time() - self.avoid_start_time < self.min_avoid_time:
            return True
            
        if not self.supress:
            front_c = self.robot.readIRSensor(IR.FrontC)
            front_ll = self.robot.readIRSensor(IR.FrontLL)
            front_rr = self.robot.readIRSensor(IR.FrontRR)
            
            # Comprobar si hay un blob centrado (usar la variable compartida)
            blob_centered = self.params.get("blob_centered", False)
            blob_detected = self.params.get("blob_detected", False)
            was_pushing = self.params.get("was_pushing", False)
            
            # Si estamos en modo "llevar a la luz", solo considerar los sensores laterales
            if self.params.get("pushing_to_light", False) and front_c >= self.obstacle_distance:
                # Si el centro detecta un objeto (posiblemente nuestro blob), solo evitamos si los laterales detectan obstáculos
                if front_ll >= self.obstacle_distance or front_rr >= self.obstacle_distance:
                    return True
                return False
            
            # Caso normal: detectar obstáculos evitando confundirlos con el blob objetivo
            obstacle_detected = (front_c >= self.obstacle_distance or 
                               front_ll >= self.obstacle_distance or 
                               front_rr >= self.obstacle_distance)
            
            # No evitar si el obstáculo es el blob que queremos empujar
            if obstacle_detected:
                # Si el blob está centrado y solo el sensor central detecta algo, probablemente es el blob
                if (blob_centered or was_pushing) and front_c >= self.obstacle_distance and front_ll < self.obstacle_distance and front_rr < self.obstacle_distance:
                    return False
                # Si estamos en modo push, priorizar empujar sobre evitar obstáculos laterales 
                # pero seguir evitando obstáculos que estén muy cerca
                elif was_pushing and front_c >= 500 and (front_ll < 100 and front_rr < 100):
                    return False
                else:
                    # Nueva maniobra de esquiva, marcar inicio
                    self.avoid_in_progress = True
                    self.avoid_start_time = time.time()
                    return True
            
            # Si ya no hay obstáculo, terminar la maniobra en progreso
            if self.avoid_in_progress:
                if time.time() - self.avoid_start_time >= self.min_avoid_time:
                    self.avoid_in_progress = False
                    
            return False

    def action(self):
        print("----> control: AvoidObstacle")
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True
        
        # Registrar el tiempo de inicio para control de tiempo máximo
        start_time = time.time()
        
        try:
            # Determinar dirección del obstáculo
            front_c = self.robot.readIRSensor(IR.FrontC)
            front_ll = self.robot.readIRSensor(IR.FrontLL)
            front_rr = self.robot.readIRSensor(IR.FrontRR)
            
            # Determinar si el obstáculo está en el centro o en los lados
            center_obstacle = front_c >= self.obstacle_distance
            left_obstacle = front_ll >= self.obstacle_distance
            right_obstacle = front_rr >= self.obstacle_distance
            
            # Movimientos más pronunciados para esquivas
            if center_obstacle and not (left_obstacle or right_obstacle):
                # Si solo hay obstáculo al centro, retroceder más y girar más
                self.robot.moveWheelsByTime(-20, -20, 1.5)  # Más velocidad y tiempo
                
                # Girar en dirección basada en lecturas históricas o aleatoria
                last_turn_direction = self.params.get("last_turn_direction", 1)
                if last_turn_direction > 0:
                    self.robot.moveWheelsByTime(-10, 25, 1.2)  # Giro más pronunciado a la derecha
                else:
                    self.robot.moveWheelsByTime(25, -10, 1.2)  # Giro más pronunciado a la izquierda
                
                # Alternar dirección para próxima vez
                self.params["last_turn_direction"] = -last_turn_direction
            else:
                # Retroceder más y más rápido
                self.robot.moveWheelsByTime(-18, -18, 1.8)  # Más retroceso
                
                # Giros más pronunciados lejos del obstáculo
                if left_obstacle and not right_obstacle:
                    # Obstáculo a la izquierda - giro pronunciado a la derecha
                    self.robot.moveWheelsByTime(-5, 30, 1.2)  # Giro más pronunciado
                    self.params["last_turn_direction"] = 1
                elif right_obstacle and not left_obstacle:
                    # Obstáculo a la derecha - giro pronunciado a la izquierda
                    self.robot.moveWheelsByTime(30, -5, 1.2)  # Giro más pronunciado
                    self.params["last_turn_direction"] = -1
                else:
                    # Obstáculos en ambos lados o caso difícil, hacer un giro mucho más pronunciado
                    if front_ll > front_rr:
                        self.robot.moveWheelsByTime(-12, 28, 1.5)  # Giro más pronunciado a la derecha
                        self.params["last_turn_direction"] = 1
                    else:
                        self.robot.moveWheelsByTime(28, -12, 1.5)  # Giro más pronunciado a la izquierda
                        self.params["last_turn_direction"] = -1
            
            # Avanzar un poco después de la maniobra para salir de la zona de peligro
            self.robot.moveWheelsByTime(15, 15, 0.7)
            
            # IMPORTANTE: Asegurar que la maniobra dure por lo menos el tiempo mínimo
            elapsed_time = time.time() - start_time
            if elapsed_time < self.min_avoid_time:
                time.sleep(self.min_avoid_time - elapsed_time)
            
            # Mantener control hasta el tiempo máximo para verificar que esquivó correctamente
            while time.time() - start_time < self.max_control_time:
                # Comprobar si el obstáculo ha sido evitado completamente
                front_c = self.robot.readIRSensor(IR.FrontC)
                front_ll = self.robot.readIRSensor(IR.FrontLL)
                front_rr = self.robot.readIRSensor(IR.FrontRR)
                
                if (front_c < self.obstacle_distance - 20 and  # Margen de seguridad adicional
                    front_ll < self.obstacle_distance - 20 and 
                    front_rr < self.obstacle_distance - 20):
                    print("Obstáculo evitado completamente, liberando control")
                    break
                
                # Pequeña pausa para no saturar la CPU
                time.sleep(0.1)
            
            print("Maniobra de esquiva completa, liberando control")
            # Establecer que ya no hay una maniobra en progreso
            self.avoid_in_progress = False
            
            # IMPORTANTE: Liberar explícitamente supress para otros comportamientos
            for bh in self.supress_list:
                bh.supress = False
                
        except Exception as e:
            print(f"Error en AvoidObstacle.action(): {e}")
            self.avoid_in_progress = False  # Asegurar que no quede atascado
        finally:
            # Asegurarse de liberar el supress en caso de error
            for bh in self.supress_list:
                bh.supress = False
            self.robot.stopMotors()