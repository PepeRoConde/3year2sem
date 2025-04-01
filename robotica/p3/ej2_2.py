from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.Emotions import Emotions
import re
import speech_recognition as sr


class RoboboController:
    def __init__(self, robobo, sim, default_speed=20, default_time=4):
        self.robobo = robobo
        self.sim = sim
        self.default_speed = default_speed
        self.default_time = default_time

        # Dictionary of movement strategies - lambda functions that return wheel speeds
        self.movement_strategies = {
            "delante": (lambda speed: (speed, speed)),
            "atrás": (lambda speed: (-speed, -speed)),
            "izquierda": (lambda speed: (-speed, speed)),
            "derecha": (lambda speed: (speed, -speed)),
        }

        # Adding compound movements
        self.compound_movements = {
            "delante izquierda": self._move_forward_left,
            "delante derecha": self._move_forward_right,
            "atrás izquierda": self._move_back_left,
            "atrás derecha": self._move_back_right,
        }

        # Emotional expressions
        self.emotions = {
            "feliz": lambda: self.robobo.setEmotionTo(Emotions.HAPPY),
            "triste": lambda: self.robobo.setEmotionTo(Emotions.SAD),
            "enfadado": lambda: self.robobo.setEmotionTo(Emotions.ANGRY),
        }

        # LED colors
        self.led_colors = {
            "rojo": lambda: self.robobo.setLedColorTo(LED.All, Color.RED),
            "verde": lambda: self.robobo.setLedColorTo(LED.All, Color.GREEN),
            "azul": lambda: self.robobo.setLedColorTo(LED.All, Color.BLUE),
        }

    def _move_forward_left(self, speed, time=None):
        """Execute a forward-left compound movement"""
        self.robobo.moveWheelsByTime(
            speed // 2, speed, self.default_time if time is None else time
        )
        return True

    def _move_forward_right(self, speed, time=None):
        """Execute a forward-right compound movement"""
        self.robobo.moveWheelsByTime(
            speed, speed // 2, self.default_time if time is None else time
        )
        return True

    def _move_back_left(self, speed, time=None):
        """Execute a back-left compound movement"""
        self.robobo.moveWheelsByTime(
            -speed, -speed // 2, self.default_time if time is None else time
        )
        return True

    def _move_back_right(self, speed, time=None):
        """Execute a back-right compound movement"""
        self.robobo.moveWheelsByTime(
            -speed // 2, -speed, self.default_time if time is None else time
        )
        return True

    def parse_command(self, user_response):
        """
        Parse the user command and move the robot
        """
        if user_response is None or user_response.strip() == "":
            print("No command entered")
            return None

        # Standardize input
        user_response = user_response.lower().strip()

        if user_response == "parar" or user_response == "salir":
            print("Stopping motors")
            self.robobo.stopMotors()
            if user_response == "salir":
                self.sim.disconnect()
                return False
            return True

        # Check for emotions
        for emotion, emotion_function in self.emotions.items():
            if emotion in user_response:
                print(f"Setting emotion: {emotion}")
                emotion_function()

        # Check for LED colors
        for color, color_function in self.led_colors.items():
            if color in user_response:
                print(f"Setting LEDs to: {color}")
                color_function()

        # Check for compound movements first
        for compound_move, move_function in self.compound_movements.items():
            if (
                compound_move in user_response
                or compound_move.replace("-", " ") in user_response
            ):
                # Extract speed and time parameters
                speeds = re.findall(r"speed (\d+)", user_response)
                current_speed = int(speeds[0]) if speeds else self.default_speed

                time_matches = re.findall(r"\btime (\d+(?:\.\d+)?)\b", user_response)
                move_time = float(time_matches[0]) if time_matches else None

                print(
                    f"Executing compound movement: {compound_move} with speed: {current_speed}"
                    + (f" for {move_time} seconds" if move_time else "")
                )

                return move_function(current_speed, move_time)

        # Get directions and speeds from text
        directions = re.findall(r"\b(delante|atrás|izquierda|derecha)\b", user_response)

        # If no direction found, report error
        if not directions:
            print("No valid direction found in command")
            return True

        # Improved speed parsing
        speeds = re.findall(r"speed (\d+)", user_response)
        current_speed = int(speeds[0]) if speeds else self.default_speed

        # Improved time parsing
        time_matches = re.findall(r"\btime (\d+(?:\.\d+)?)\b", user_response)
        move_time = float(time_matches[0]) if time_matches else self.default_time

        # Execute movements
        for direction in directions:
            if direction in self.movement_strategies:
                # Get wheel speeds (speeds can be different)
                left, right = self.movement_strategies[direction](current_speed)

                # Move the robot
                print(
                    f"Moving {direction} at speed {current_speed} for {move_time} seconds"
                )
                self.robobo.moveWheelsByTime(left, right, move_time)
        return True

    def voice_to_text(self, timeout=None, phrase_time_limit=4.5):
        """
        Convert voice to text
        """
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            try:
                # Listen with additional parameters
                audio = r.listen(
                    source,
                    timeout=timeout,  # Maximum wait time to start listening
                    phrase_time_limit=phrase_time_limit,  # Maximum time for speaking
                )

                # Only using English recognition
                try:
                    command = r.recognize_google(audio, language="es")  # type: ignore
                    print(f"Recognized: {command}")
                    return command
                except sr.UnknownValueError:
                    print("Could not understand audio in English")
                    return None

            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None

            except sr.WaitTimeoutError:
                print("Listening timeout expired")
                return None


def main(ip="localhost"):
    # Initialize simulator and robot
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = RoboboController(robobo, sim)

    print("Robobo Voice Controller")
    print("======================")
    print("Speak commands to control Robobo.")
    print("Available commands: delante, atrás, izquierda, derecha, parar, salir")
    print(
        "You can also use compound commands: delante-derecha, delante-izquierda, etc."
    )
    print("Emotion commands: feliz, triste, enfadado")
    print("Color commands: rojo, verde, azul")
    print("You can specify speed: 'delante volocidad 30'")
    print("You can specify time: 'delante time 3.5'")

    try:
        while True:
            user_response = controller.voice_to_text()

            if user_response is None:
                print("No input detected, trying again...")
                continue

            result = controller.parse_command(user_response)

            if result is False:
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Clean up
        robobo.stopMotors()
        sim.disconnect()
        print("Disconnected from Robobo and simulator")


if __name__ == "__main__":
    main()
