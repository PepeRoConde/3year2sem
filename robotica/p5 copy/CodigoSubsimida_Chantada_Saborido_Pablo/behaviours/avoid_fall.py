from behaviours.behaviour import Behaviour
from robobopy.utils.IR import IR
import time

class AvoidFall(Behaviour):
    """
    Comportamiento de máxima prioridad que evita que el robot caiga por los bordes.
    """
    def __init__(self, robot, supress_list, params):
        super().__init__(robot, supress_list, params)
        # Parámetros de configuración
        self.safe_distance = 45  # Valor mínimo de IR que indica un posible borde, mayores valores reconecen el suelo
        self.backup_time = 1.5   # Tiempo de retroceso
        self.turn_time = 1.2     # Tiempo de giro
        self.backup_speed = 20   # Velocidad de retroceso
        self.turn_speed = 15     # Velocidad de giro
        self.pan_positions = [-60, -30, 0, 30, 60]  # Posiciones para escanear
        
    #----------------------------------------
    # CONTROL DE ACTIVACIÓN
    #----------------------------------------
    
    def take_control(self):
        """
        Toma el control si alguno de los sensores frontales detecta una posible caída
        (valores muy bajos de IR significan que no hay superficie)
        """
        front_l = self.robot.readIRSensor(IR.FrontL)
        front_r = self.robot.readIRSensor(IR.FrontR)
        
        return (front_l < self.safe_distance or 
                front_r < self.safe_distance)

    #----------------------------------------
    # ACCIÓN PRINCIPAL
    #----------------------------------------
    
    def action(self):
        """
        Ejecuta el comportamiento de evitar caídas 
        """
        print("----> control: AvoidFall")
        # Desactivar otros comportamientos
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True
        
        try:
            # Determinar de qué lado viene el riesgo de caída
            front_l = self.robot.readIRSensor(IR.FrontL)
            front_r = self.robot.readIRSensor(IR.FrontR)
            front_c = self.robot.readIRSensor(IR.FrontC)
            print(f"Lecturas IR - Izquierda: {front_l}, Centro: {front_c}, Derecha: {front_r}")
            left_fall = front_l < self.safe_distance
            right_fall = front_r < self.safe_distance
            
            # Comprobar si estamos empujando un blob actualmente (el blob es detectado en el centro)
            blob_present = self.params.get("was_pushing", False) and front_c > 60
            
            if blob_present:
                print("Blob detectado cerca del borde - ejecutando giro simplificado hacia la luz")
                
                # 1. Detener motores
                self.robot.stopMotors()
                time.sleep(0.1)
                
                # 2. Escanear dirección de la luz
                light_angle, light_brightness = self.scan_for_light()
                print(f"Luz detectada en ángulo: {light_angle}, brillo: {light_brightness}")

                while front_c > 500:
                    # 4. Girar 90 grados hacia la dirección de la luz
                    if light_angle > 0:  # La luz está a la derecha
                        print("Girando 90 grados a la derecha hacia la luz")
                        self.robot.moveWheelsByTime(9, -9, 2.5)  # Girar a la derecha
                        break
                    else:  # La luz está a la izquierda o centro
                        print("Girando 90 grados a la izquierda hacia la luz")
                        self.robot.moveWheelsByTime(-9, 9, 2.5)  # Girar a la izquierda
                        break

                time.sleep(0.5)

                # 5. Moverse hacia adelante en línea recta
                print("Avanzando en línea recta")
                self.robot.moveWheelsByTime(20, 20, 2.5)
                
            else:
                # Evitación de caída estándar sin blob
                print("Evitación de caída estándar - no se detectó blob")
                self.robot.stopMotors()
                
                # 1. Retroceder
                self.robot.moveWheelsByTime(-self.backup_speed, -self.backup_speed, self.backup_time)
                
                # 2. Girar alejándose del borde detectado
                if left_fall:
                    # Si el borde está a la izquierda, girar a la derecha
                    print("Girando a la derecha")
                    self.robot.moveWheelsByTime(self.turn_speed, -self.turn_speed, self.turn_time)
                else:  # right_fall
                    # Si el borde está a la derecha, girar a la izquierda
                    print("Girando a la izquierda")
                    self.robot.moveWheelsByTime(-self.turn_speed, self.turn_speed, self.turn_time)
                
                # 3. Avanzar un poco para salir completamente de la zona de peligro
                self.robot.moveWheelsByTime(10, 10, 0.5)
            
        finally:
            # Asegurar que se libera la supresión al terminar o en caso de error
            for bh in self.supress_list:
                bh.supress = False
            
            # Detener motores al finalizar la maniobra
            self.robot.stopMotors()