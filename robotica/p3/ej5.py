from robobopy.Robobo import Robobo
from robobopy.utils.Emotions import Emotions
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from robobopy.utils.Sounds import Sounds

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

        self.strategies = {
            "happy": ...,
            "sad": ...,
            "sleepy": ...,
            "scream": ...,
            "laught": ...,
            "beep": ...
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

        # Vector de comandos: [vx, vy, emocion, sonido, texto]
        vx, vy, emotion, sound, text = eval(user_response)
        # Ejecutar los movimientos
        self.robobo.moveWheels(vx,vy)
        self.robobo.setEmotion(Emotions.emotion)
        self.robobo.playSound(Sounds.sound)
        self.robobo.sayText(text)


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
                    "content": "CAMBIAR A ALGO CORRECTO PARA ESTE CASO",},
                {"role": "user", "content": command},
            ],
        )
        print(f"Generated text: {response}")
        return response


def main(ip="localhost"):
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = RoboboController(robobo, sim)
    vector_str = "[10, 10, 'hello', 'alegre', 'forward']"



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
