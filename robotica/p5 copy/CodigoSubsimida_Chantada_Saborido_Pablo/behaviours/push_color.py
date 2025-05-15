from behaviours.behaviour import Behaviour
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor
import time

class PushColor(Behaviour):
    """
    Comportamiento para empujar blobs detectados hacia fuentes de luz.
    """
    def __init__(self, robot, supress_list, params, color=BlobColor.RED):
        super().__init__(robot, supress_list, params)
        # Parámetros de configuración de blobs
        self.color = color
        self.push_distance = 80  # Distancia para empujar
        self.center_x = 50  # Centro de la imagen en x
        self.center_threshold = 10  # Umbral para considerar centrado
        self.pan_positions = [-60, -30, 0, 30, 60]  # Posiciones para escanear
        
        # Parámetros del controlador PID para movimiento más suave
        self.kp = 0.5  # Ganancia proporcional
        self.ki = 0.1  # Ganancia integral
        self.kd = 0.2  # Ganancia derivativa
        self.integral = 0
        self.prev_error = 0
        
        # Parámetros de movimiento
        self.base_speed = 10  # Velocidad base moderada para movimiento más seguro
        self.movement_duration = 2.0  
        
        # Seguimiento de conteo de empujones
        self.push_count_threshold = 3  # Después de esta cantidad de empujones, activar modo push-to-light
        
        # Detección muy cercana - usado para determinar si el blob está muy cerca
        self.very_close_threshold = 2000  # Valor IR que indica proximidad muy cercana
    
    #----------------------------------------
    # CONTROL DE ACTIVACIÓN
    #----------------------------------------
    
    def take_control(self):
        """
        Tomar control cuando un blob es visible, está centrado y a distancia empujable
        o cuando el sensor IR indica que un objeto (probablemente el blob) está cerca.
        """
        if not self.supress:
            blob = self.robot.readColorBlob(self.color)
            ir_value = self.robot.readIRSensor(IR.FrontC)
            
            # Detectar presencia de blob usando cámara o IR
            is_visible = blob.size > 0
            is_centered = is_visible and abs(blob.posx - self.center_x) <= self.center_threshold
            is_close = self.push_distance <= ir_value < 6000  
            
            # Considerar el objeto muy cerca si el valor IR es muy alto
            very_close = ir_value >= self.very_close_threshold
            
            # Comprobar si estábamos empujando previamente
            was_pushing = self.params.get("was_pushing", False)
            
            # Tomar control si vemos un blob que está centrado y lo suficientemente cerca,
            # o si estábamos empujando y todavía detectamos algo con el IR
            if (is_visible and is_centered and is_close) or (was_pushing and ir_value > 100):
                print(f"Objeto detectado - Visible: {is_visible}, IR: {ir_value}, Muy cerca: {very_close}")
                self.params["was_pushing"] = True
                
                # Al tomar el control, comprobar si debemos incrementar el conteo de empujones
                current_time = time.time()
                last_push_time = self.params.get("last_push_time", 0)
                
                # Solo incrementar si ha pasado un tiempo significativo desde el último empujón
                if current_time - last_push_time > 1.5:
                    # Actualizar conteo de empujones y tiempo
                    push_count = self.params.get("push_count", 0) + 1
                    self.params["push_count"] = push_count
                    self.params["last_push_time"] = current_time
                    print(f"Conteo de empujones incrementado a {push_count}")
                    
                    # Comprobar si hemos alcanzado el umbral para activar modo push-to-light
                    if push_count >= self.push_count_threshold and not self.params.get("pushing_to_light", False):
                        print("*** ¡Activando modo PUSH-TO-LIGHT! ***")
                        self.params["pushing_to_light"] = True
                
                return True
            
            return False
        
        return False

    #----------------------------------------
    # ACCIÓN PRINCIPAL
    #----------------------------------------
    
    def action(self):
        """
        Ejecutar comportamiento de empuje con movimiento más suave y consistente.
        """
        print("----> control: PushColor")
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True
        
        # Comprobar si estamos en modo push-to-light
        pushing_to_light = self.params.get("pushing_to_light", False)
        if pushing_to_light:
            print("Operando en modo PUSH-TO-LIGHT")
        
        # Reiniciar controlador PID
        self.integral = 0
        self.prev_error = 0
        
        try:
            # Primero, verificar y maximizar alineación del blob si es necesario
            self.ensure_blob_alignment()
            
            # Luego comenzar movimiento hacia la luz
            self.move_toward_light(pushing_to_light)
            
        finally:
            # Detener motores al salir
            self.robot.stopMotors()
            
            # Mantener flag was_pushing activa
            self.params["was_pushing"] = True
    
    #----------------------------------------
    # MÉTODOS DE ALINEACIÓN Y MOVIMIENTO
    #----------------------------------------
    
    def ensure_blob_alignment(self):
        """
        Asegurarse de que el blob está bien centrado antes de comenzar a empujar
        """
        print("Verificando alineación del blob...")
        
        # Obtener posición actual del blob
        blob = self.robot.readColorBlob(self.color)
        ir_value = self.robot.readIRSensor(IR.FrontC)
        
        # Si el blob es visible pero no está bien centrado, centrarlo
        if blob.size > 0 and abs(blob.posx - self.center_x) > 5:
            print(f"Alineando con blob en posición {blob.posx}")
            
            # Calcular cuánto girar
            error = blob.posx - self.center_x
            turn_speed = min(10, max(5, abs(error) / 5))
            
            # Girar hacia el blob
            if error < 0:  # Blob está a la izquierda
                self.robot.moveWheelsByTime(turn_speed, -turn_speed, 0.3)
            else:  # Blob está a la derecha
                self.robot.moveWheelsByTime(-turn_speed, turn_speed, 0.3)
            
            # Pequeña pausa para estabilizar
            self.robot.stopMotors()
            time.sleep(0.1)
        
        # Si el blob no es visible pero IR lo detecta, intentar centrarlo
        elif blob.size == 0 and ir_value > 100:
            print("Blob detectado por IR pero no visible, intentando centrar")
            
            # Pequeño movimiento aleatorio para encontrar blob
            # Primero verificar izquierda, luego derecha
            self.robot.moveWheelsByTime(5, -5, 0.3)
            self.robot.stopMotors()
            time.sleep(0.1)
            
            # Comprobar si lo encontramos
            if self.robot.readColorBlob(self.color).size == 0:
                # No encontrado, intentar derecha
                self.robot.moveWheelsByTime(-5, 5, 0.6)
                self.robot.stopMotors()
                time.sleep(0.1)
            
        print("Verificación de alineación completa")
    
    def move_toward_light(self, pushing_to_light):
        """
        Moverse consistentemente hacia la luz usando control más suave
        """
        print("Iniciando movimiento hacia la luz")
        
        # Obtener dirección de la luz primero
        light_angle, light_brightness = self.scan_for_light()
        print(f"Luz detectada en ángulo {light_angle}, brillo {light_brightness}")
        
        # Establecer duración del movimiento según el modo
        duration = self.movement_duration
        if pushing_to_light:
            duration = duration * 1.5  # Movimiento más largo en modo push-to-light
        
        # Calcular velocidades para movimiento más suave
        right_speed, left_speed = self.calculate_speeds_pid(light_angle, light_brightness)
        
        # Aplicar velocidades para la duración completa
        print(f"Moviendo con velocidades - Derecha: {right_speed}, Izquierda: {left_speed} durante {duration}s")
        self.robot.moveWheels(right_speed, left_speed)
        
        # Monitorear movimiento durante toda la duración
        start_time = time.time()
        while time.time() - start_time < duration and not self.supress:
            # Comprobar si perdimos el blob (solo en modo push-to-light)
            blob = self.robot.readColorBlob(self.color)
            ir_value = self.robot.readIRSensor(IR.FrontC)
            
            if pushing_to_light and blob.size == 0 and ir_value < 50:
                print("Blob perdido durante modo push-to-light")
                break
            
            # Pequeña pausa durante monitoreo
            time.sleep(0.05)
    
    #----------------------------------------
    # CONTROL DE MOVIMIENTO Y ESCANEO
    #----------------------------------------
    
    def calculate_speeds_pid(self, angle, brightness):
        """
        Calcular velocidades de ruedas usando controlador PID para movimiento más suave
        """
        # Si el brillo es muy bajo, moverse recto
        if brightness < 20:
            return self.base_speed, self.base_speed
        
        # Calcular error (ángulo desde el centro)
        error = angle / 60.0  # Normalizar al rango -1 a 1
        
        # Actualizar términos integral y derivativo
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        
        # Limitar término integral para prevenir acumulación
        self.integral = max(-2.0, min(2.0, self.integral))
        
        # Calcular corrección usando fórmula PID
        correction = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Aplicar corrección a velocidades de ruedas
        if correction < 0:  # Necesita girar a la izquierda
            right_speed = self.base_speed
            left_speed = self.base_speed * (1 + correction)  # Reducir velocidad izquierda
        else:  # Necesita girar a la derecha
            right_speed = self.base_speed * (1 - correction)  # Reducir velocidad derecha
            left_speed = self.base_speed
        
        # Asegurar que las velocidades están dentro de límites
        right_speed = max(5, min(25, right_speed))
        left_speed = max(5, min(25, left_speed))
        
        return int(right_speed), int(left_speed)