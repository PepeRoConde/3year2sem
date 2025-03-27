from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from tkinter import simpledialog
import re
import openai

client = openai.OpenAI(api_key=api_key) # type: ignore  


class RoboboController:
    def __init__(self, robobo, sim, default_speed=20, default_time=2):
        self.robobo = robobo
        self.sim = sim
        self.default_speed = default_speed
        self.default_time = default_time

        self.movement_strategies = {
            "forward": (lambda speed: (speed, speed)),
            "back": (lambda speed: (-speed, -speed)),
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

        if user_response == "stop":
            self.robobo.stopMotors()
            # self.sim.disconnect()
            # return False
            return

        # Obtener direcciones y velocidades del texto
        directions = re.findall(r"\b(forward|back|left|right)\b", user_response)
        
        # Improved speed parsing
        speeds = re.findall(r"speed (\d+)", user_response)
        current_speed = int(speeds[0]) if speeds else self.default_speed

        # Improved time parsing
        time_matches = re.findall(r"\btime (\d+(?:\.\d+)?)\b", user_response)
        move_time = float(time_matches[0]) if time_matches else None

        # Ejecutar los movimientos
        for direction in directions:
            if direction in self.movement_strategies:
                
                # Obtener velocidades de las ruedas (las velocidades pueden ser diferentes)
                left, right = self.movement_strategies[direction](current_speed)

                # Mover el robot
                if move_time is not None:
                    print(f"Moving {direction} at speed {current_speed} for {move_time} seconds")
                    self.robobo.moveWheelsByTime(left, right, move_time)
                    if direction == 'left'or direction == 'right':
                        self.robobo.moveWheelsByTime(current_speed, current_speed, self.default_time)
                else:
                    print(f"Moving {direction} at speed {current_speed}")
                    self.robobo.moveWheels(left, right)
                    if direction == 'left'or direction == 'right':
                        self.robobo.moveWheelsByTime(current_speed, current_speed, self.default_time)
        return True

    def get_text_command(self):
        """
        Obtener el comando del usuario
        """
        return simpledialog.askstring("Input", "Enter command:")

    def get_chatgpt_text(self, command):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Necesito que me des comandos para controlar un robot segun la frase que te ponga. Los comandos son: forward, back, left, right y stop. Ademas, tambien se pueden incluir velocidades (enteros) o tiempo (floats). Los comandos pueden ser complejos como: forward-left. Dame unicamente el comando, nada mas.",
                        
                },
                {"role": "user", "content": command},
            ],
            temperature=0
        )

        # Depuración para ver toda la respuesta JSON
        print(f"Generated response: {response.choices[0].message.content}")

        return response.choices[0].message.content


def main(ip="localhost"):
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = RoboboController(robobo, sim)

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
