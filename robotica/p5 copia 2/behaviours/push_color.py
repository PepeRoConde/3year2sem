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
        self.light_scan_interval = 0.2  # Intervalo para escanear luz (reducido a 0.2 segundos)
        self.last_light_scan = 0  # Tiempo del último escaneo de luz
        self.pan_positions = [-60, -30, 0, 30, 60]  # Posiciones para escaneo rápido
        self.quick_scan_positions = [-30, 0, 30]  # Posiciones para escaneo ultra-rápido
        self.alternate_scan = False  # Alternar entre escaneo completo y rápido
        self.last_correction_time = 0  # Tiempo de la última corrección
        self.correction_interval = 0.1  # Intervalo mínimo entre correcciones (reducido a 0.1 segundos)
        self.consecutive_low_brightness = 0  # Contador para detección de luz a la espalda
        self.low_brightness_threshold = 25  # Brillo por debajo del cual se considera bajo
        self.max_low_brightness_count = 3  # Máximo de lecturas de bajo brillo antes de girar
        self.front_light_bonus = 1.5  # Multiplicador de velocidad cuando la luz está al frente
        self.front_light_angle_threshold = 20  # Ángulo considerado como "frontal" (grados)
        self.last_light_angle = 0  # Último ángulo de luz detectado
        self.light_angle_change_threshold = 15  # Umbral para considerar cambio significativo
        self.last_significant_change_time = 0  # Tiempo del último cambio significativo de ángulo
        
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
        
        # Estado inicial de empuje
        base_push_speed = 20
        current_time = time.time()
        self.last_light_scan = current_time
        
        # Hacer un escaneo inicial completo
        light_angle, light_brightness = self.scan_for_light(use_quick_scan=False)
        self.last_light_angle = light_angle
        self.last_significant_change_time = current_time
        
        # Bucle principal de empuje mientras mantenemos el control
        while not self.supress and not self.stopped():
            current_time = time.time()
            blob = self.robot.readColorBlob(self.color)
            ir_value = self.robot.readIRSensor(IR.FrontC)
            
            # Verificar si seguimos teniendo el objeto
            has_object = ir_value > 100 and ir_value < self.max_ir_value
            if not has_object:
                # Verificar también con la cámara
                blob_visible = blob.size > 0
                if not blob_visible:
                    print("Objeto perdido, liberando control")
                    break
            
            # Ajustar velocidad base según distancia al objeto
            if ir_value > 1000:  # Muy cerca
                push_speed = base_push_speed - 2
            elif ir_value > 500:  # Distancia media
                push_speed = base_push_speed
            else:  # Más lejos
                push_speed = base_push_speed + 2
            
            # Escanear periódicamente para encontrar la luz
            if current_time - self.last_light_scan >= self.light_scan_interval:
                # Determinar si usar escaneo rápido o completo
                time_since_change = current_time - self.last_significant_change_time
                
                # Usar escaneo rápido si ha habido un cambio reciente o si estamos alternando
                use_quick_scan = (time_since_change < 2.0) or self.alternate_scan
                
                # Alternar entre escaneo rápido y completo
                self.alternate_scan = not self.alternate_scan
                
                light_angle, light_brightness = self.scan_for_light(use_quick_scan)
                self.last_light_scan = current_time
                
                # Detectar cambios significativos en la dirección de la luz
                angle_change = abs(light_angle - self.last_light_angle)
                if angle_change > self.light_angle_change_threshold:
                    print(f"Cambio significativo en dirección de luz: {self.last_light_angle}° → {light_angle}°")
                    self.last_significant_change_time = current_time
                    
                self.last_light_angle = light_angle
                
                # Verificar si la luz está posiblemente a la espalda (brillo muy bajo)
                if light_brightness < self.low_brightness_threshold:
                    self.consecutive_low_brightness += 1
                    print(f"Brillo bajo: {light_brightness}, contador: {self.consecutive_low_brightness}/{self.max_low_brightness_count}")
                    
                    if self.consecutive_low_brightness >= self.max_low_brightness_count:
                        print("Posible luz a la espalda, girando 180 grados")
                        # Girar 180 grados (derecha, izquierda)
                        self.robot.moveWheelsByTime(20, -20, 2.2)
                        self.consecutive_low_brightness = 0
                else:
                    self.consecutive_low_brightness = 0
                
                # Calcular corrección basada en la dirección de la luz
                correction = self.calculate_light_correction(light_angle, light_brightness)
                
                # Aplicar corrección a las velocidades (derecha, izquierda)
                right_speed = push_speed
                left_speed = push_speed
                
                # Determinar si la luz está "al frente" (entre -threshold y +threshold grados)
                is_front_light = abs(light_angle) <= self.front_light_angle_threshold
                
                if is_front_light and light_brightness >= self.min_brightness:
                    # Si la luz está al frente, aumentar ambas velocidades para ir más directo
                    front_bonus = self.front_light_bonus
                    # Más bonus mientras más centrada está y más brillante es
                    angle_factor = 1.0 - (abs(light_angle) / self.front_light_angle_threshold)
                    brightness_factor = min(1.0, light_brightness / 100)  # Normalizar hasta 100
                    
                    # Calcular bonus: máximo cuando está perfectamente centrada y muy brillante
                    dynamic_bonus = 1.0 + (front_bonus - 1.0) * angle_factor * brightness_factor
                    
                    print(f"Luz al frente - Aplicando bonus de velocidad: x{dynamic_bonus:.2f}")
                    right_speed = int(right_speed * dynamic_bonus)
                    left_speed = int(left_speed * dynamic_bonus)
                
                # Aplicar corrección inmediatamente si es significativa
                apply_now = angle_change > self.light_angle_change_threshold
                
                if (correction != 0 and 
                    (current_time - self.last_correction_time >= self.correction_interval or apply_now)):
                    print(f"Aplicando corrección: {correction} hacia {'derecha' if correction > 0 else 'izquierda'}")
                    
                    if correction < 0:  # Luz a la izquierda
                        right_speed = int(right_speed * (1 - abs(correction)))
                    else:  # Luz a la derecha
                        left_speed = int(left_speed * (1 - abs(correction)))
                    
                    self.last_correction_time = current_time
                
                # Aplicar velocidades (NOTA: orden invertido para este robot)
                print(f"Velocidades - Derecha: {right_speed}, Izquierda: {left_speed}")
                self.robot.moveWheels(left_speed, right_speed)
            
            # Verificar si hay riesgo de caída
            if self.check_fall_risk():
                print("Riesgo de caída detectado, retrocediendo")
                self.robot.moveWheelsByTime(-20, -20, 1.0)
                # Girar en dirección opuesta al borde
                front_l = self.robot.readIRSensor(IR.FrontL)
                front_r = self.robot.readIRSensor(IR.FrontR)
                if front_l < front_r:
                    # Girar a la derecha
                    self.robot.moveWheelsByTime(15, -15, 0.8)
                else:
                    # Girar a la izquierda
                    self.robot.moveWheelsByTime(-15, 15, 0.8)
            
            # Pequeña pausa para no saturar CPU - más corta para respuestas más rápidas
            time.sleep(0.02)  # Reducido de 0.05 a 0.02 segundos
        
        # Detener motores al salir
        self.robot.stopMotors()
        
        # Mantener la bandera was_pushing activa
        self.params["was_pushing"] = True
    
    def scan_for_light(self, use_quick_scan=False):
        """
        Escanea rápidamente para encontrar la dirección de la luz
        
        Args:
            use_quick_scan: Si es True, usa menos posiciones para escaneo más rápido
            
        Returns:
            tuple: (mejor_ángulo, brillo_máximo)
        """
        brightness_readings = []
        
        # Determinar qué posiciones usar
        positions = self.quick_scan_positions if use_quick_scan else self.pan_positions
        
        # Guardar posición original de pan
        original_pan = self.robot.readPanPosition()
        
        # Escanear en diferentes ángulos
        for angle in positions:
            self.robot.movePanTo(angle, 100)
            time.sleep(0.05)  # Tiempo breve para estabilizar (reducido a 0.05)
            reading = self.robot.readBrightnessSensor()
            brightness_readings.append(reading)
        
        # Restaurar posición original
        self.robot.movePanTo(original_pan, 100)
        
        # Encontrar posición más brillante
        max_brightness = max(brightness_readings)
        best_index = brightness_readings.index(max_brightness)
        best_angle = positions[best_index]
        
        scan_type = "Rápido" if use_quick_scan else "Completo"
        print(f"Escaneo {scan_type} - Ángulos: {positions}, Brillo: {brightness_readings}")
        print(f"Mejor ángulo: {best_angle}, Brillo: {max_brightness}")
        
        return best_angle, max_brightness
    
    def calculate_light_correction(self, angle, brightness):
        """
        Calcula el factor de corrección para orientarse hacia la luz
        
        Args:
            angle: Ángulo en el que se detectó la luz máxima
            brightness: Intensidad de la luz detectada
            
        Returns:
            float: Factor de corrección entre -0.5 y 0.5
                  Negativo para girar a la izquierda, positivo para la derecha
        """
        if brightness < self.min_brightness:
            # Si el brillo es muy bajo, no corregir
            return 0
        
        # Corrección más fuerte para ángulos mayores
        angle_magnitude = abs(angle)
        response_factor = 1.0 + (angle_magnitude / 60) * 0.5  # Aumenta hasta 1.5x para ángulos extremos
        
        # Si la luz está al frente, reducir la corrección para mantener trayectoria más recta
        if abs(angle) <= self.front_light_angle_threshold:
            # Disminuir corrección en proporción a lo centrada que esté la luz
            center_factor = abs(angle) / self.front_light_angle_threshold  # 0 si perfectamente centrada, 1 en el límite
            
            # Normalizar el ángulo pero con efecto reducido en el centro
            max_angle = max(abs(min(self.pan_positions)), abs(max(self.pan_positions)))
            correction = angle / (max_angle * 2)
            
            # Aplicar reducción basada en lo centrada que está
            correction *= (0.5 + 0.5 * center_factor)  # Reducir hasta un 50% cuando está perfectamente centrada
        else:
            # Comportamiento normal para luz a los lados - más responsivo
            max_angle = max(abs(min(self.pan_positions)), abs(max(self.pan_positions)))
            correction = (angle / (max_angle * 2)) * response_factor
        
        # Limitar la corrección máxima
        correction = max(-0.5, min(0.5, correction))
        
        return correction
        
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