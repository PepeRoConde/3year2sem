from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from tkinter import simpledialog
import re

class RoboboController:
    def __init__(self, robobo, sim, default_speed=10, default_time=1):
        self.robobo = robobo
        self.sim = sim
        self.default_speed = default_speed
        self.default_time = default_time
        
        self.movement_strategies = {
            'forward': (lambda speed: (speed, speed)),
            'backward': (lambda speed: (-speed, -speed)),
            'left': (lambda speed: (-speed, speed)),
            'right': (lambda speed: (speed, -speed))
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
        directions = re.findall(r'\b(forward|backward|left|right)\b', user_response)
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

def main(ip='localhost'):

    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()
    
    controller = RoboboController(robobo, sim)
    
    try:
        while True:
            user_response = controller.get_text_command()
            result = controller.parse_command(user_response)
            
            if result is False:
                break
    
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()

if __name__ == '__main__':
    main()
