from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from tkinter import simpledialog
import re
import openai

'''
Entiendo que es un vector de strings/ints, no un vector con valores float

Las keys son para cada conversacion o para entrar a chatgpt normal ?

Vector de comandos: [vx, vy, emocion, sonido, texto]

- vx y vy representan los componentes de velocidad para cada rueda,
- emoción es el estado emocional de la cara de Robobo,
- sonido es un comando específico de sonido que debe reproducir Robobo,
- y texto es un mensaje que el robot debe vocalizar.
'''

class RoboboController:
    def __init__(self, robobo, sim, default_speed=10, default_time=1):
        self.robobo = robobo
        self.sim = sim
        self.default_speed = default_speed
        self.default_time = default_time

        self.movement_strategies = {
            "forward": (lambda speed: (speed, speed)),
            "backward": (lambda speed: (-speed, -speed)),
            "left": (lambda speed: (-speed, speed)),
            "right": (lambda speed: (speed, -speed)),
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

        if user_response == "quit":
            self.robobo.stopMotors()
            self.sim.disconnect()
            return False

        # Obtener direcciones y velocidades del texto
        directions = re.findall(r"\b(forward|backward|left|right)\b", user_response)
        speeds = re.findall(r"\d+", user_response)

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

    def get_chatgpt_text(self, command):
        response = openai.chat.completions.create(
            model="gpt-3.0-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Te voy a decir comandos para mover un robot. Quiero que me categorices estos comandos. "
                    "Unicamente tienes que moverte hacia X. Dame solo las categorias: forward, backward, right, left. "
                    "Considera las acciones a realizar segun los escenarios planteados.",
                },
                {"role": "user", "content": command},
            ],
        )
        print(f"Generated text: {response}")
        return response

    def parse_vector_string(self, vector_str):
        # Convertir el vector de strings a una lista
        vector = eval(vector_str)
        # Extraer los elementos del vector
        vx, vy, emocion, sonido, texto = vector
        return vx, vy, emocion, sonido, texto

# vector_str = "[10, 10, 'hello', 'alegre', 'forward']"
# result = RoboboController(None, None).parse_vector_string(vector_str)
# print(result)

def main(ip="localhost"):
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = RoboboController(robobo, sim)
    vector_str = "[10, 10, 'hello', 'alegre', 'forward']"
    result = controller.parse_vector_string(vector_str)
    print(result)

    try:
        while True:
            user_response = controller.get_text_command()
            # result = controller.parse_command(user_response)
            result = controller.parse_command(
                controller.get_chatgpt_text(user_response)
            )
            if result is False:
                break

    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()


if __name__ == "__main__":
    main()
