import re
from typing_extensions import override
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.Emotions import Emotions
from robobopy.utils.Sounds import Sounds


class RoboboController:
    """
    Clase base que implementa las funcionalidades comunes para todos los ejercicios.
    Proporciona métodos para movimientos básicos, compuestos y gestión de emociones/LEDs.
    """
    
    def __init__(self, robobo, sim, default_speed=20, default_time=1):
        """
        Args:
            robobo: Instancia del robot Robobo
            sim: Instancia del simulador
            default_speed: Velocidad base
            default_time: Tiempo predeterminado
        """
        self.robobo = robobo
        self.sim = sim
        self.default_speed = default_speed
        self.default_time = default_time

        # Movimientos básicos 
        self.movement_strategies_en = {
            "forward": (lambda speed: (speed, speed)),
            "back": (lambda speed: (-speed, -speed)),
            "left": (lambda speed: (speed, -speed)),
            "right": (lambda speed: (-speed, speed)),
        }
        
        # Movimientos básicos (nombres en español para el ejercicio 2_2)
        self.movement_strategies_es = {
            "delante": (lambda speed: (speed, speed)),
            "atrás": (lambda speed: (-speed, -speed)),
            "izquierda": (lambda speed: (speed, -speed)),
            "derecha": (lambda speed: (-speed, speed)),
        }

        # Movimientos compuestos 
        self.compound_movements_en = {
            "forward-left": self._move_forward_left,
            "forward-right": self._move_forward_right,
            "back-left": self._move_back_left,
            "back-right": self._move_back_right,
        }
        
        # Movimientos compuestos (nombres en español para el ejercicio 2_2)
        self.compound_movements_es = {
            "delante izquierda": self._move_forward_left,
            "delante derecha": self._move_forward_right,
            "atrás izquierda": self._move_back_left,
            "atrás derecha": self._move_back_right,
        }

        # Emociones
        self.emotions = {
            "feliz": lambda: self.robobo.setEmotionTo(Emotions.HAPPY),
            "happy": lambda: self.robobo.setEmotionTo(Emotions.HAPPY),
            "triste": lambda: self.robobo.setEmotionTo(Emotions.SAD),
            "sad": lambda: self.robobo.setEmotionTo(Emotions.SAD),
            "enfadado": lambda: self.robobo.setEmotionTo(Emotions.ANGRY),
            "angry": lambda: self.robobo.setEmotionTo(Emotions.ANGRY),
        }

        # LEDs
        self.led_colors = {
            "rojo": lambda: self.robobo.setLedColorTo(LED.All, Color.RED),
            "red": lambda: self.robobo.setLedColorTo(LED.All, Color.RED),
            "verde": lambda: self.robobo.setLedColorTo(LED.All, Color.GREEN),
            "green": lambda: self.robobo.setLedColorTo(LED.All, Color.GREEN),
            "azul": lambda: self.robobo.setLedColorTo(LED.All, Color.BLUE),
            "blue": lambda: self.robobo.setLedColorTo(LED.All, Color.BLUE),
        }
        
        # Sonidos
        self.sounds = {
            "feliz": lambda: self.robobo.playSound(Sounds.LIKES),
            "happy": lambda: self.robobo.playSound(Sounds.LIKES),
            "triste": lambda: self.robobo.playSound(Sounds.DISCOMFORT),
            "sad": lambda: self.robobo.playSound(Sounds.DISCOMFORT),
            "enfadado": lambda: self.robobo.playSound(Sounds.ANGRY),
            "angry": lambda: self.robobo.playSound(Sounds.ANGRY),
        }

    def _move_forward_left(self, speed, time=None):
        """
        Movimiento adelante-izquierda.

        Args:
            speed: Velocidad del movimiento
            time: Tiempo del movimiento en segundos 
            
        Returns:
            True si el movimiento se ejecutó correctamente
        """
        self.robobo.moveWheelsByTime(
            speed, speed // 2, self.default_time if time is None else time
        )
        return True

    def _move_forward_right(self, speed, time=None):
        """
        Movimiento adelante-derecha.

        Args:
            speed: Velocidad del movimiento
            time: Tiempo del movimiento en segundos
            
        Returns:
            True si el movimiento se ejecutó correctamente
        """
        self.robobo.moveWheelsByTime(
            speed // 2, speed, self.default_time if time is None else time
        )
        return True

    def _move_back_left(self, speed, time=None):
        """
        Movimiento atrás-izquierda.
        
        Args:
            speed: Velocidad del movimiento
            time: Tiempo del movimiento en segundos 
            
        Returns:
            True si el movimiento se ejecutó correctamente
        """
        self.robobo.moveWheelsByTime(
            -speed, -speed // 2, self.default_time if time is None else time
        )
        return True

    def _move_back_right(self, speed, time=None):
        """
        Movimiento atrás-derecha.
        
        Args:
            speed: Velocidad del movimiento
            time: Tiempo del movimiento en segundos
            
        Returns:
            True si el movimiento se ejecutó correctamente
        """
        self.robobo.moveWheelsByTime(
            -speed // 2, -speed, self.default_time if time is None else time
        )
        return True
        
    def parse_command(self, user_response, language="en"):
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

        # Verificar primero movimientos compuestos
        for compound_move, move_function in compound_movements.items():
            if (
                compound_move in user_response
                or compound_move.replace("-", " ") in user_response
            ):
                # Extraer parámetros de velocidad y tiempo
                speeds = re.findall(r"speed (\d+)", user_response)
                current_speed = int(speeds[0]) if speeds else self.default_speed

                time_matches = re.findall(r"\btime (\d+(?:\.\d+)?)\b", user_response)
                move_time = float(time_matches[0]) if time_matches else None

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

        # Velocidad básica
        speeds = re.findall(r"speed (\d+)", user_response)
        current_speed = int(speeds[0]) if speeds else self.default_speed

        # Tiempo básico
        time_matches = re.findall(r"\btime (\d+(?:\.\d+)?)\b", user_response)
        move_time = float(time_matches[0]) if time_matches else self.default_time

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
        
    def parse_vector_command(self, vector_command):
        """
        Procesa un comando vectorial para el control.
        
        Args:
            vector_command: [vx, vy, emotion, sound, text]
            
        Returns:
            True para continuar, False para salir, None si hay error
        """
        try:
            if not isinstance(vector_command, list) or len(vector_command) != 5:
                print("Invalid command format. Expected [vx, vy, emotion, sound, text]")
                return None
                
            vx, vy, emotion, sound, text = vector_command
            
            # Validar los tipos de datos
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
            if emotion_lower in self.emotions:
                print(f"Setting emotion: {emotion}")
                self.emotions[emotion_lower]()
            else:
                print(f"Unknown emotion: {emotion}")
                
            # Reproducir sonido
            sound_lower = sound.lower()
            if sound_lower in self.sounds:
                print(f"Playing sound: {sound}")
                self.sounds[sound_lower]()
            else:
                print(f"Unknown sound: {sound}")
                
            # Decir texto
            if text:
                print(f"Saying: {text}")
                self.robobo.sayText(str(text))
                
            return True
                
        except Exception as e:
            print(f"Error parsing vector command: {e}")
            return None
   
    def get_command(self) -> str |None:
        """
        Método base para obtener un comando.
        Debe ser sobrescrito por las clases derivadas.
        """
        return None 
