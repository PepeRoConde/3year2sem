from robobo_controller import RoboboController
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from tkinter import simpledialog
import openai
import time

client = openai.OpenAI(api_key=api_key)

class VectorRoboboController(RoboboController):
    """
    Controlador paramétrico de Robobo que utiliza ChatGPT para generar vectores de control.
    """
    
    def __init__(self, robobo, sim, default_speed=10, default_time=1):
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
        return simpledialog.askstring("Input", "Enter command or natural language input:")

    def process_natural_language(self, user_input):
        """
        Procesa lenguaje natural mediante ChatGPT para obtener vectores de control.
        
        Args:
            user_input: Texto en lenguaje natural
            
        Returns:
            Vector de control procesado por ChatGPT
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Necesito que generes comandos de control para un robot Robobo a partir de frases en lenguaje natural.
                        
                        Tu respuesta DEBE ser un vector de 5 elementos con el siguiente formato exacto: [vx, vy, emocion, sonido, texto]
                        
                        Donde:
                        - vx y vy son valores numéricos entre -100 y 100 que representan las velocidades de las ruedas derecha e izquierda, usa 20 como valor por defecto
                        - emocion es un string que debe ser uno de los siguientes: "happy", "sad", "angry"
                        - sonido es un string que debe ser uno de los siguientes: "happy", "sad", "angry"
                        - texto es un string con el mensaje que el robot debe decir
                        
                        Ejemplos de comandos y sus respuestas:
                        - "Avanza rápido con cara feliz": [50, 50, "happy", "happy", "¡Avanzando a toda velocidad!"]
                        - "Gira a la derecha lentamente con cara triste": [-20, 20, "sad", "sad", "Girando a la derecha..."]
                        
                        Para los movimientos, considera:
                        - Avanzar: valores positivos iguales en vx y vy
                        - Retroceder: valores negativos iguales en vx y vy
                        - Girar a la derecha: vx positivo, vy negativo
                        - Girar a la izquierda: vx negativo, vy positivo                        
                        - Velocidad rápida: valores cercanos a 100
                        - Velocidad lenta: valores cercanos a 20
                        
                        Responde ÚNICAMENTE con el vector, sin explicaciones ni texto adicional.
                        """,
                    },
                    {"role": "user", "content": user_input},
                ],
                temperature=0.5,
            )

            result = response.choices[0].message.content
            print(f"Respuesta de ChatGPT: {result}")
            return result

        except Exception as e:
            print(f"Error llamando a la API de ChatGPT: {e}")
            return "[0, 0, 'happy', 'happy', 'Error en la comunicación']"  

def main(ip="localhost"):
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = VectorRoboboController(robobo, sim)

    print("Introduce frases para controlar el robot.")
    print("Ejemplos:")
    print("- Avanza rápido con cara feliz")
    print("- Gira a la derecha lentamente con cara triste")
    print("- Quédate quieto y cuenta hasta diez")
    print("Escribe 'quit' para salir.")

    try:
        while True:
            # Obtener comando del usuario
            user_response = controller.get_command()
            
            if user_response is None or user_response.lower() == "quit":
                print("Saliendo del programa")
                break
                
            # Procesar a través de ChatGPT
            chatgpt_command = controller.process_natural_language(user_response)
            
            try:
                # Evaluar comando como vector Python
                vector_command = eval(chatgpt_command) # type: ignore
                
                # Ejecutar el comando vectorial
                result = controller.parse_vector_command(vector_command)
                
                if result is False:
                    break
                    
                if result is True:
                    time.sleep(3)
                    robobo.stopMotors()
                    
            except Exception as e:
                print(f"Error al procesar el comando: {e}")
                print("Por favor, intenta con otro comando.")

    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
    finally:
        # Detener motores y desconectar al finalizar
        robobo.stopMotors()
        sim.disconnect()


if __name__ == "__main__":
    main()
