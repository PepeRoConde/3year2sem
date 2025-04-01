from robobopy.Robobo import Robobo
from robobopy.utils.Emotions import Emotions
from robobopy.utils.Sounds import Sounds
from robobosim.RoboboSim import RoboboSim

from tkinter import simpledialog
import openai

client = openai.OpenAI(api_key=api_key)

class RoboboController:
    def __init__(self, robobo, sim, default_speed=10, default_time=1):
        self.robobo = robobo
        self.sim = sim
        self.default_speed = default_speed
        self.default_time = default_time

        self.emotion_map = {
            "happy": Emotions.HAPPY,
            "sad": Emotions.SAD,
            "angry": Emotions.ANGRY,
        }
        
        self.sound_map = {
            "happy": Sounds.LIKES,
            "sad": Sounds.DISCOMFORT,
            "angry": Sounds.ANGRY,
        }

    def parse_command(self, user_response):
        """
        Parsear el comando del usuario y mover el robot
        """
        if user_response is None:
            print("No command entered")
            return None

        # Estandarizamos el input
        user_response = user_response.strip()

        if user_response.lower() == "quit":
            self.robobo.stopMotors()
            self.sim.disconnect()
            return False
            
        try:
            command_vector = eval(user_response)
            
            if not isinstance(command_vector, list) or len(command_vector) != 5:
                print("Invalid command format. Expected [vx, vy, emotion, sound, text]")
                return None
                
            vx, vy, emotion, sound, text = command_vector
            
            # Validar tipos de datos
            if not (isinstance(vx, (int, float)) and isinstance(vy, (int, float))):
                print("Invalid wheel speeds. vx and vy must be numbers")
                return None
                
            if not isinstance(emotion, str) or not isinstance(sound, str) or not isinstance(text, str):
                print("Invalid command format. emotion, sound, and text must be strings")
                return None
                
            # Ejecutar los movimientos
            print(f"Moving wheels: Left={vx}, Right={vy}")
            self.robobo.moveWheels(vx, vy)
            
            # Establecer emoción
            emotion_lower = emotion.lower()
            if emotion_lower in self.emotion_map:
                print(f"Setting emotion: {emotion}")
                self.robobo.setEmotionTo(self.emotion_map[emotion_lower])
            else:
                print(f"Unknown emotion: {emotion}")
                
            # Reproducir sonido
            sound_lower = sound.lower()
            if sound_lower in self.sound_map:
                print(f"Playing sound: {sound}")
                self.robobo.playSound(self.sound_map[sound_lower])
            else:
                print(f"Unknown sound: {sound}")
                
            # Decir texto
            if text:
                print(f"Saying: {text}")
                self.robobo.sayText(str(text))
                
            return True
                
        except Exception as e:
            print(f"Error parsing command: {e}")
            return None

    def get_text_command(self):
        """
        Obtener el comando del usuario
        """
        return simpledialog.askstring("Input", "Enter command or natural language input:")

    def get_chatgpt_text(self, command):
        """
        Procesar el comando de lenguaje natural a través de ChatGPT 
        para obtener el vector de control
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
                        - vx y vy son valores numéricos entre -100 y 100 que representan las velocidades de las ruedas izquierda y derecha, usa 20 como valor por defecto
                        - emocion es un string que debe ser uno de los siguientes: "happy", "sad", "angry"
                        - sonido es un string que debe ser uno de los siguientes: "happy", "sad", "angry"
                        - texto es un string con el mensaje que el robot debe decir
                        
                        Ejemplos de comandos y sus respuestas:
                        - "Avanza rápido con cara feliz": [50, 50, "happy", "happy", "¡Avanzando a toda velocidad!"]
                        - "Gira a la derecha lentamente con cara triste": [20, -20, "sad", "sad", "Girando a la derecha..."]
                        
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
                    {"role": "user", "content": command},
                ],
                temperature=0.7,
            )

            # Log de la respuesta para debug
            result = response.choices[0].message.content
            print(f"Respuesta de ChatGPT: {result}")
            return result

        except Exception as e:
            print(f"Error llamando a la API de ChatGPT: {e}")
            return "[0, 0, 'normal', 'beep', 'Error en la comunicación']"  # Comando de fallback


def main(ip="localhost"):
    # Inicializar simulador y robot
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = RoboboController(robobo, sim)

    print("Robobo Control Paramétrico con ChatGPT")
    print("======================================")
    print("Introduce comandos en lenguaje natural para controlar a Robobo.")
    print("Escribe 'quit' para salir.")
    print("\nEjemplos de comandos:")
    print("- Move forward, display happiness, and laugh quietly")
    print("- Spin slowly while appearing very scared and say a random number")
    print("- Remain stationary, appear sleepy, and count sheep to fall asleep")
    print("- Move backward while turning slightly and count down from ten to one")
    print("- Move randomly, adopt a random mood, and share an interesting fact about Robobo")

    try:
        while True:
            # Obtener comando del usuario
            user_response = controller.get_text_command()
            
            if user_response is None or user_response.lower() == "quit":
                print("Saliendo del programa")
                break
                
            # Procesar a través de ChatGPT
            chatgpt_command = controller.get_chatgpt_text(user_response)
            
            # Ejecutar el comando
            result = controller.parse_command(chatgpt_command)
            
            if result is False:
                break
                
            # Pausa de seguridad después de cada comando
            if result is True:
                # Después de 3 segundos, detener motores para seguridad
                import time
                time.sleep(3)
                robobo.stopMotors()

    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
    finally:
        # Limpieza
        robobo.stopMotors()
        sim.disconnect()
        print("Desconectado de Robobo y del simulador")


if __name__ == "__main__":
    main()
