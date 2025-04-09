from robobo_controller import RoboboController
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from tkinter import simpledialog
import openai

client = openai.OpenAI(api_key=api_key)


class ChatGPTRoboboController(RoboboController):
    """
    Controlador de Robobo que utiliza ChatGPT para interpretar comandos en lenguaje natural.
    """
    
    def __init__(self, robobo, sim, default_speed=20, default_time=1):
        """
        Args:
            robobo: Instancia del robot Robobo
            sim: Instancia del simulador
            default_speed: Velocidad base
            default_time: Tiempo predeterminado
        """
        super().__init__(robobo, sim, default_speed, default_time)
        
    def get_command(self):
        """
        Obtiene un comando de texto del usuario usando un cuadro de diálogo.
        
        Returns:
            Texto del comando 
        """
        return simpledialog.askstring("Input", "Enter command:")

    def process_natural_language(self, user_input):
        """
        Procesa lenguaje natural mediante ChatGPT para obtener comandos de robot.
        
        Args:
            user_input: Texto en lenguaje natural
            
        Returns:
            Comando de robot procesado por ChatGPT
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Necesito que extraigas comandos de movimiento para un robot a partir de frases en lenguaje natural. 
                        
                        Los comandos permitidos son:
                        - forward: para mover hacia adelante
                        - back: para mover hacia atrás
                        - left: para girar a la izquierda
                        - right: para girar a la derecha
                        - stop: para detener el movimiento
                        
                        También puedes combinar los comandos de dirección para movimientos compuestos como:
                        - forward-left: para avanzar girando a la izquierda
                        - forward-right: para avanzar girando a la derecha
                        - back-left: para retroceder girando a la izquierda
                        - back-right: para retroceder girando a la derecha
                        
                        Además, puedes incluir parámetros:
                        - speed X: donde X es un número entero que indica la velocidad
                        - time Y: donde Y es un número que indica los segundos de movimiento
                        
                        Ejemplos:
                        - "Avanza un poco": forward
                        - "Retrocede a velocidad 30": back speed 30
                        - "Gira a la derecha durante 3 segundos": right time 3
                        - "Avanza girando a la izquierda a velocidad 25": forward-left speed 25
                        
                        Responde ÚNICAMENTE con el comando extraído, sin explicaciones ni formato adicional.
                        """,
                    },
                    {"role": "user", "content": user_input},
                ],
                temperature=0,
            )

            print(
                f"Generated response from ChatGPT: {response.choices[0].message.content}"
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling ChatGPT API: {e}")
            return "stop"  


def main(ip="localhost"):
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = ChatGPTRoboboController(robobo, sim)

    print("Introduce frases para controlar el robot.")
    print("Ejemplos:")
    print("- \"Avanza un poco\": forward")
    print("- \"Retrocede a velocidad 30\": back speed 30")
    print("- \"Gira a la derecha durante 3 segundos\": right time 3")
    print("- \"Avanza girando a la izquierda a velocidad 25\": forward-left speed 25")
    print("Escribe 'quit' para salir.")

    try:
        while True:
            # Obtener comando del usuario
            user_response = controller.get_command()

            if user_response is None or user_response.lower() == "quit":
                print("Exiting program")
                break

            # Procesar a través de ChatGPT
            chatgpt_command = controller.process_natural_language(user_response)
            print(f"Interpreted command: {chatgpt_command}")

            # Ejecutar el comando
            result = controller.parse_command(chatgpt_command, language="en")

            if result is False:
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Detener motores y desconectar al finalizar
        robobo.stopMotors()
        sim.disconnect()


if __name__ == "__main__":
    main()
