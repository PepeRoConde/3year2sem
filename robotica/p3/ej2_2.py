from robobo_controller import RoboboController
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import speech_recognition as sr
import re

class VoiceRoboboController(RoboboController):
    """
    Controlador de Robobo que recibe comandos a través de entrada de voz.
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
        Obtiene un comando a través de entrada de voz.
        
        Returns:
            Texto del comando reconocido o None si no se reconoció
        """
        return self.voice_to_text()
        
    def voice_to_text(self, timeout=None, phrase_time_limit=4.5):
        """
        Convierte la voz a texto usando reconocimiento de voz.
        
        Args:
            timeout: Tiempo máximo de espera para comenzar a escuchar
            phrase_time_limit: Tiempo máximo para hablar
            
        Returns:
            Texto reconocido o None si no se pudo reconocer
        """
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            try:
                # Escuchar con parámetros adicionales
                audio = r.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

                # Usar reconocimiento en español
                try:
                    command = r.recognize_google(audio, language="es")  # type: ignore
                    print(f"Recognized: {command}")
                    return command
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    return None

            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None

            except sr.WaitTimeoutError:
                print("Listening timeout expired")
                return None

    def parse_command(self, user_response, language="es"):
        """
        Procesa el comando del usuario y ejecuta el movimiento.
        Args:
            user_response: Texto del comando.
            language: Idioma del comando ("en" o "es")
            
        Returns:
            True para continuar, False para salir, None si no hay comando
        """
        if user_response is None or user_response.strip() == "":
            print("No command entered")
            return None

        # Estandarizar la entrada
        user_response = user_response.lower().strip()

        # Seleccionar diccionarios según el idioma
        movement_strategies = self.movement_strategies_en if language == "en" else self.movement_strategies_es
        compound_movements = self.compound_movements_en if language == "en" else self.compound_movements_es
        stop_command = "stop" if language == "en" else "parar"
        quit_command = "quit" if language == "en" else "salir"

        # Comandos para detener/salir
        if user_response == stop_command or user_response == quit_command:
            print("Stopping motors")
            self.robobo.stopMotors()
            if user_response == quit_command:
                self.sim.disconnect()
                return False
            return True

        # Verificar emociones
        for emotion, emotion_function in self.emotions.items():
            if emotion in user_response:
                print(f"Setting emotion: {emotion}")
                emotion_function()

        # Verificar colores de LEDs
        for color, color_function in self.led_colors.items():
            if color in user_response:
                print(f"Setting LEDs to: {color}")
                color_function()

        # Extraer velocidad - buscando patrones en español
        speeds = re.findall(r"speed (\d+)", user_response)
        if not speeds:
            speeds = re.findall(r"velocidad (\d+)", user_response)
        current_speed = int(speeds[0]) if speeds else self.default_speed

        # Extraer tiempo
        time_matches = re.findall(r"\btime (\d+(?:\.\d+)?)\b", user_response)
        if not time_matches:
            time_matches = re.findall(r"\btiempo (\d+(?:\.\d+)?)\b", user_response)
        move_time = float(time_matches[0]) if time_matches else self.default_time

        # Verificar primero movimientos compuestos
        for compound_move, move_function in compound_movements.items():
            if (
                compound_move in user_response
                or compound_move.replace("-", " ") in user_response
            ):
                print(
                    f"Executing compound movement: {compound_move} with speed: {current_speed}"
                    + (f" for {move_time} seconds" if move_time else "")
                )
                return move_function(current_speed, move_time)

        # Obtener direcciones del texto
        directions_pattern = "|".join(movement_strategies.keys())
        directions = re.findall(rf"\b({directions_pattern})\b", user_response)

        # Si no se encuentra dirección, error
        if not directions:
            print("No valid direction found in command")
            return True

        # Ejecutar movimientos básicos
        for direction in directions:
            if direction in movement_strategies:
                # Obtener velocidades de las ruedas según la dirección
                left, right = movement_strategies[direction](current_speed)

                print(
                    f"Moving {direction} at speed {current_speed} for {move_time} seconds"
                )
                self.robobo.moveWheelsByTime(left, right, move_time)

        return True


def main(ip="localhost"):
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = VoiceRoboboController(robobo, sim)

    print("Comandos disponibles: delante, atrás, izquierda, derecha, parar, salir")
    print("Comandos compuestos: delante izquierda, delante derecha, etc.")
    print("Comandos de emoción: feliz, triste, enfadado")
    print("Comandos de color: rojo, verde, azul")
    print("Comando con velocidad: 'delante velocidad 30'")
    print("Comando con tiempo: 'delante tiempo 3.5'")

    try:
        while True:
            user_response = controller.get_command()

            if user_response is None:
                print("No input detected, trying again...")
                continue

            result = controller.parse_command(user_response, language="es")

            # Si el usuario sale
            if result is False:
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Detener motores y desconectar si usamos KeyboardInterrupt
        robobo.stopMotors()
        sim.disconnect()


if __name__ == "__main__":
    main()
