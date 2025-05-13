from behaviours.behaviour import Behaviour
from robobopy.utils.IR import IR
from robobopy.utils.Sounds import Sounds

class AvoidFall(Behaviour):
    def __init__(self, robot, supress_list, params):
        super().__init__(robot, supress_list, params)
        self.safe_distance = 10  # Valor mínimo de IR que indica un posible borde
        self.backup_time = 1.5   # Tiempo de retroceso
        self.turn_time = 1.2     # Tiempo de giro
        self.backup_speed = 20   # Velocidad de retroceso
        self.turn_speed = 15     # Velocidad de giro
        
    def take_control(self):
        # Si cualquiera de los sensores frontales detecta una posible caída
        # (valores muy bajos de IR significan que no hay superficie)
        front_l = self.robot.readIRSensor(IR.FrontL)
        front_r = self.robot.readIRSensor(IR.FrontR)
        
        return (front_l < self.safe_distance or 
                front_r < self.safe_distance)

    def action(self):
        print("----> control: AvoidFall")
        self.supress = False
        for bh in self.supress_list:
            bh.supress = True
        
        try:
            # Determinar de qué lado viene el riesgo de caída
            front_l = self.robot.readIRSensor(IR.FrontL)
            front_r = self.robot.readIRSensor(IR.FrontR)
            
            left_fall = front_l < self.safe_distance
            right_fall = front_r < self.safe_distance
            
            # Indicar la dirección de la caída detectada
            if left_fall and right_fall:
                print("¡Caída detectada en ambos lados! Retrocediendo y girando")
            elif left_fall:
                print("¡Caída detectada a la izquierda! Retrocediendo y girando a la derecha")
            elif right_fall:
                print("¡Caída detectada a la derecha! Retrocediendo y girando a la izquierda")
            
            # Detener el robot antes de iniciar la maniobra
            self.robot.stopMotors()
            
            # 1. Retroceder para alejarse del borde
            print(f"Retrocediendo durante {self.backup_time} segundos")
            self.robot.moveWheelsByTime(-self.backup_speed, -self.backup_speed, self.backup_time)
            
            # 2. Girar en la dirección opuesta al borde detectado
            if left_fall and right_fall:
                # Si ambos lados tienen riesgo, girar en dirección aleatoria o predeterminada
                # Usamos la última dirección de giro guardada en params o por defecto a la derecha
                last_turn_dir = self.params.get("last_turn_direction", 1)
                if last_turn_dir > 0:
                    print("Girando a la derecha")
                    self.robot.moveWheelsByTime(self.turn_speed, -self.turn_speed, self.turn_time)
                else:
                    print("Girando a la izquierda")
                    self.robot.moveWheelsByTime(-self.turn_speed, self.turn_speed, self.turn_time)
                
                # Invertir la dirección para el próximo giro
                self.params["last_turn_direction"] = -last_turn_dir
            elif left_fall:
                # Si el borde está a la izquierda, girar a la derecha
                print("Girando a la derecha")
                self.robot.moveWheelsByTime(self.turn_speed, -self.turn_speed, self.turn_time)
                self.params["last_turn_direction"] = 1
            else:  # right_fall
                # Si el borde está a la derecha, girar a la izquierda
                print("Girando a la izquierda")
                self.robot.moveWheelsByTime(-self.turn_speed, self.turn_speed, self.turn_time)
                self.params["last_turn_direction"] = -1
            
            # 3. Avanzar ligeramente para salir completamente de la zona de peligro
            print("Avanzando ligeramente para alejarse de la zona de peligro")
            self.robot.moveWheelsByTime(10, 10, 0.5)
            
        finally:
            # Asegurarse de liberar el supress al terminar o en caso de error
            for bh in self.supress_list:
                bh.supress = False
            
            # Detener motores al finalizar la maniobra
            self.robot.stopMotors()