from behaviours.behaviour import Behaviour
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

class PushColor(Behaviour):
    def __init__(self, robot, supress_list, params, color=BlobColor.RED):
        super().__init__(robot, supress_list, params)
        self.color = color
        self.push_distance = 80  # Distancia para empujar
        self.max_ir_value = 5000  # Valor máximo aceptable de IR
        self.center_x = 50  # Centro de la imagen en x
        self.center_threshold = 10  # Umbral para considerar centrado
        self.min_brightness = 20  # Umbral mínimo para considerar luz
        self.pan_positions = [-60, -30, 0, 30, 60]  # Posiciones para escaneo
        self.consecutive_low_brightness = 0  # Contador para detección de luz a la espalda
        self.low_brightness_threshold = 25  # Brillo por debajo del cual se considera bajo
        self.max_low_brightness_count = 3  # Máximo de lecturas de bajo brillo antes de girar
        self.front_light_bonus = 1.5  # Multiplicador de velocidad cuando la luz está al frente
        self.front_light_angle_threshold = 20  # Ángulo considerado como "frontal" (grados)
        
        # Parámetros para el nuevo enfoque escanear-mover
        self.movement_duration = 0.8  # Duración de cada movimiento (segundos)
        self.max_speed = 25  # Velocidad máxima de las ruedas
        self.min_speed = 10  # Velocidad mínima de las ruedas
        
    def take_control(self):
        if not self.supress:
            blob = self.robot.readColorBlob(self.color)
            ir_value = self.robot.readIRSensor(IR.FrontC)
            
            # CASO 1: Blob visible y centrado
            is_visible = blob.size > 0
            is_centered = is_visible and abs(blob.posx - self.center_x) <= self.center_threshold
            is_close = self.push_distance <= ir_value < self.max_ir_value
            
            # CASO 2: No vemos el blob pero el IR indica que hay algo cerca
            ir_indicates_object = ir_value > 100 and ir_value < self.max_ir_value
            was_pushing = self.params.get("was_pushing", False)
            
            if (is_visible and is_close) or (ir_indicates_object and was_pushing):
                print(f"Objeto detectado - Visible: {is_visible}, IR: {ir_value}")
                self.params["was_pushing"] = True
                return True
                
            return False
            
        return False

    def action(self):
        print("----> control: PushColor (empujando hacia la luz)")
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True
        
        # Velocidad base
        base_push_speed = 20
        
        # Bucle principal con enfoque escanear-mover
        while not self.supress and not self.stopped():
            # Detener el robot antes de escanear para lecturas más precisas
            self.robot.stopMotors()
            time.sleep(0.1)  # Pausa breve para estabilización
            
            # 1. FASE DE ESCANEO
            print("Fase de escaneo: buscando dirección de luz...")
            light_angle, light_brightness = self.scan_for_light()
            
            # Verificar objeto
            blob = self.robot.readColorBlob(self.color)
            ir_value = self.robot.readIRSensor(IR.FrontC)
            has_object = ir_value > 100 and ir_value < self.max_ir_value
            
            if not has_object and blob.size == 0:
                print("Objeto perdido, liberando control")
                break
                
            # Verificar luz a la espalda
            if light_brightness < self.low_brightness_threshold:
                self.consecutive_low_brightness += 1
                print(f"Brillo bajo: {light_brightness}, contador: {self.consecutive_low_brightness}/{self.max_low_brightness_count}")
                
                if self.consecutive_low_brightness >= self.max_low_brightness_count:
                    print("Posible luz a la espalda, girando 180 grados")
                    self.robot.moveWheelsByTime(20, -20, 2.2)
                    self.consecutive_low_brightness = 0
                    continue  # Volver a escanear después del giro
            else:
                self.consecutive_low_brightness = 0
            
            # 2. CÁLCULO DE VELOCIDADES
            # Ajustar velocidad base según distancia al objeto
            if ir_value > 1000:  # Muy cerca
                push_speed = base_push_speed - 2
            elif ir_value > 500:  # Distancia media
                push_speed = base_push_speed
            else:  # Más lejos
                push_speed = base_push_speed + 2
                
            # Calcular velocidades izquierda y derecha según dirección de luz
            right_speed, left_speed = self.calculate_movement_speeds(light_angle, light_brightness, push_speed)
            
            print(f"Fase de movimiento: Aplicando velocidades - Derecha: {right_speed}, Izquierda: {left_speed}")
            
            # 3. FASE DE MOVIMIENTO
            # Aplicar velocidades calculadas (con orden invertido)
            self.robot.moveWheels(right_speed, left_speed)
            
            # Mover durante un tiempo fijo
            movement_start = time.time()
            
            # Durante el movimiento, comprobar obstáculos y caídas
            while time.time() - movement_start < self.movement_duration and not self.supress:
                # Comprobar riesgo de caída durante el movimiento
                if self.check_fall_risk():
                    print("Riesgo de caída detectado, interrumpiendo movimiento")
                    self.robot.stopMotors()
                    self.avoid_fall()
                    break
                    
                # Pequeña pausa durante el movimiento para permitir comprobaciones
                time.sleep(0.05)
            
            # Pequeña pausa entre ciclos
            time.sleep(0.05)
        
        # Detener motores al salir
        self.robot.stopMotors()
        
        # Mantener la bandera was_pushing activa
        self.params["was_pushing"] = True
    
    def scan_for_light(self):
        """
        Escanea para encontrar la dirección de la luz
        
        Returns:
            tuple: (mejor_ángulo, brillo_máximo)
        """
        brightness_readings = []
        
        # Guardar posición original de pan
        original_pan = self.robot.readPanPosition()
        
        # Escanear en diferentes ángulos
        for angle in self.pan_positions:
            self.robot.movePanTo(angle, 100)
            time.sleep(0.1)  # Tiempo para estabilizar
            # Tomar múltiples lecturas para mayor precisión
            reading_sum = 0
            readings = 3
            for _ in range(readings):
                reading_sum += self.robot.readBrightnessSensor()
                time.sleep(0.02)
            avg_reading = reading_sum / readings
            brightness_readings.append(avg_reading)
        
        # Restaurar posición original
        self.robot.movePanTo(original_pan, 100)
        
        # Encontrar posición más brillante
        max_brightness = max(brightness_readings)
        best_index = brightness_readings.index(max_brightness)
        best_angle = self.pan_positions[best_index]
        
        print(f"Escaneo de luz - Ángulos: {self.pan_positions}")
        print(f"Lecturas: {[round(b, 1) for b in brightness_readings]}")
        print(f"Mejor ángulo: {best_angle}, Brillo: {max_brightness}")
        
        return best_angle, max_brightness
    
    def calculate_movement_speeds(self, angle, brightness, base_speed):
        """
        Calcula las velocidades de las ruedas basadas en el ángulo de la luz
        
        Args:
            angle: Ángulo en que se detectó la luz
            brightness: Brillo detectado
            base_speed: Velocidad base para empujar
            
        Returns:
            tuple: (velocidad_derecha, velocidad_izquierda)
        """
        # Si el brillo es muy bajo, mantener velocidades iguales
        if brightness < self.min_brightness:
            return base_speed, base_speed
        
        right_speed = base_speed
        left_speed = base_speed
        
        # Ajustar velocidades según el ángulo
        is_front_light = abs(angle) <= self.front_light_angle_threshold
        
        if is_front_light:
            # Luz al frente - bonus de velocidad
            front_bonus = self.front_light_bonus
            # Bonus adaptativo según centralidad
            angle_factor = 1.0 - (abs(angle) / self.front_light_angle_threshold)
            brightness_factor = min(1.0, brightness / 100)
            
            dynamic_bonus = 1.0 + (front_bonus - 1.0) * angle_factor * brightness_factor
            
            print(f"Luz al frente - Bonus de velocidad: x{dynamic_bonus:.2f}")
            right_speed = int(right_speed * dynamic_bonus)
            left_speed = int(left_speed * dynamic_bonus)
            
            # Pequeña corrección para luz no perfectamente centrada
            if angle < 0:  # Luz ligeramente a la izquierda
                right_speed = int(right_speed * 0.9)
            elif angle > 0:  # Luz ligeramente a la derecha
                left_speed = int(left_speed * 0.9)
        else:
            # Luz a los lados - velocidades diferenciales para girar
            if angle < 0:  # Luz a la izquierda
                # Más negativo = más a la izquierda = más diferencia de velocidad
                diff_factor = abs(angle) / 90.0  # Normalizado de 0 a 1
                right_speed = int(base_speed)  # Mantener velocidad derecha
                left_speed = int(base_speed * (1.0 - diff_factor * 0.8))  # Reducir velocidad izquierda
                print(f"Girando hacia izquierda - Factor: {diff_factor:.2f}")
            else:  # Luz a la derecha
                diff_factor = abs(angle) / 90.0  # Normalizado de 0 a 1
                left_speed = int(base_speed)  # Mantener velocidad izquierda
                right_speed = int(base_speed * (1.0 - diff_factor * 0.8))  # Reducir velocidad derecha
                print(f"Girando hacia derecha - Factor: {diff_factor:.2f}")
        
        # Asegurar que las velocidades están dentro de límites razonables
        right_speed = max(self.min_speed, min(self.max_speed, right_speed))
        left_speed = max(self.min_speed, min(self.max_speed, left_speed))
        
        return right_speed, left_speed
    
    def avoid_fall(self):
        """
        Maniobra para evitar caída
        """
        # Retroceder
        self.robot.moveWheelsByTime(-20, -20, 1.0)
        
        # Girar en dirección opuesta al borde
        front_l = self.robot.readIRSensor(IR.FrontL)
        front_r = self.robot.readIRSensor(IR.FrontR)
        
        if front_l < front_r:
            # Borde a la izquierda, girar a la derecha
            self.robot.moveWheelsByTime(15, -15, 0.8)
        else:
            # Borde a la derecha, girar a la izquierda
            self.robot.moveWheelsByTime(-15, 15, 0.8)
    
    def check_fall_risk(self):
        """
        Detecta si hay riesgo de caída
        
        Returns:
            bool: True si hay riesgo de caída, False en caso contrario
        """
        front_l = self.robot.readIRSensor(IR.FrontL)
        front_r = self.robot.readIRSensor(IR.FrontR)
        safe_distance = 10
        
        return min(front_l, front_r) < safe_distance