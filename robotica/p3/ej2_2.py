from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from tkinter import simpledialog
import re
import speech_recognition as sr

'''
AÑADIR COMPORATAMIENTOS EXTRA COMO EMOCIONES O LEDS
(me daba pereza xddd)
'''



class RoboboController:
    def __init__(self, robobo, sim, default_speed=10, default_time=1):
        self.robobo = robobo
        self.sim = sim
        self.default_speed = default_speed
        self.default_time = default_time
        
        self.movement_strategies = {
            # 'forward': (lambda speed: (speed, speed)),
            # 'backward': (lambda speed: (-speed, -speed)),
            # 'left': (lambda speed: (-speed, speed)),
            # 'right': (lambda speed: (speed, -speed))
            'delante': (lambda speed: (speed, speed)),
            'atrás': (lambda speed: (-speed, -speed)),
            'izquierda': (lambda speed: (-speed, speed)),
            'derecha': (lambda speed: (speed, -speed))
        }
    
    def parse_command(self, user_response):
        """
        Parsear el comando del usuario y mover el robot
        """
        if user_response is None:
            print("No command entered")
            return None

        # Estandarizamos el input
        user_response = user_response.lower()
        
        if user_response == 'quit':
            self.robobo.stopMotors()
            self.sim.disconnect()
            return False
        
        # Obtener direcciones y velocidades del texto
        # directions = re.findall(r'\b(forward|backward|left|right)\b', user_response)

        directions = re.findall(r'\b(delante|atrás|izquierda|derecha)\b', user_response)
        speeds = re.findall(r'\d+', user_response)
        
        current_speed = int(speeds[0]) if speeds else self.default_speed
        
        # Ejecutar los movimientos 
        for direction in directions:
            if direction in self.movement_strategies:
                # Obtener velocidades de las ruedas (las velocidades pueden ser diferentes)
                left, right = self.movement_strategies[direction](current_speed)
                
                # Mover el robot
                self.robobo.moveWheelsByTime(left, right, self.default_time)
                print(f"Moving {direction} at speed {current_speed}")
        
        return True
    
    def get_text_command(self):
        """
        Obtener el comando del usuario
        """
        return simpledialog.askstring("Input", "Enter command:")

    def voice_to_text(self, timeout=None, phrase_time_limit=5):
        """
        Convertir la voz a texto
        """
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            try:
                # Escuchar con parámetros adicionales
                audio = r.listen(
                    source, 
                    timeout=timeout,  # Tiempo máximo de espera para comenzar a escuchar
                    phrase_time_limit=phrase_time_limit  # Tiempo máximo para hablar
                )
                
                # Reconocer el audio con mayor precisión
                command = r.recognize_google(audio, language='es-ES')  # type: ignore 
                print(f"Command: {command}")
                return command        

            except sr.UnknownValueError:
                print("Could not understand audio")
                return None

            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                return None

            except sr.WaitTimeoutError:
                print("Tiempo de espera agotado")
                return None


def main(ip='localhost'):

    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()
    
    controller = RoboboController(robobo, sim)
    
    try:
        while True:
            user_response = controller.voice_to_text()
            result = controller.parse_command(user_response)

            if result is False:
                break
    
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()

if __name__ == '__main__':
    main()
