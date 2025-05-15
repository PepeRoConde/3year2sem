from threading import Thread
import time

class Behaviour(Thread):
    """
    Clase base que gestiona los hilos de comportamiento.
    """
    def __init__(self, robot, supress_list, params, **kwargs):
        """
        Inicializa un nuevo comportamiento.
        
        Args:
            robot: Instancia del robot 
            supress_list: Comportamientos de menor prioridad a suprimir
            params: Diccionario compartido para comunicación entre comportamientos
            **kwargs: Argumentos adicionales para Thread
        """
        super().__init__(**kwargs)
        self.robot = robot
        self.__supress = False
        self.supress_list = supress_list
        self.params = params

    #----------------------------------------
    # MÉTODOS DE SUBCLASES
    #----------------------------------------
    
    def take_control(self):
        """
        Determina si este comportamiento debe tomar el control.
        Debe ser implementado por las subclases.
        
        Returns:
            bool: True si el comportamiento debe tomar el control, False en caso contrario
        """
        pass

    def action(self):
        """
        Ejecuta acciones del comportamiento cuando tiene el control.
        Debe ser implementado por las subclases.
        """
        pass

    #----------------------------------------
    # MÉTODOS DE GESTIÓN DE HILOS
    #----------------------------------------
    
    def run(self):
        """
        Método de ejecución del hilo.
        Comprueba continuamente si debe tomar el control, y ejecuta la acción cuando debe.
        Se detiene si params["stop"] se establece en True.
        """
        while not self.params["stop"]:
            while not self.take_control() and not self.params["stop"]:
                time.sleep(0.01)
            if not self.params["stop"]:
                self.action()

    #----------------------------------------
    # PROPIEDADES Y MÉTODOS DE CONTROL
    #----------------------------------------
    
    @property
    def supress(self):
        """
        Obtiene el estado de supresión.
        
        Returns:
            bool: True si el comportamiento está suprimido, False en caso contrario
        """
        return self.__supress

    @supress.setter
    def supress(self, state):
        """
        Establece el estado de supresión.
        
        Args:
            state: Booleano que indica si el comportamiento debe ser suprimido
        """
        self.__supress = state

    def set_stop(self):
        """
        Establece la flag de parada para terminar todos los comportamientos.
        """
        self.params["stop"] = True

    def stopped(self):
        """
        Comprueba si la flag de parada está establecida.
        
        Returns:
            bool: True si la bandera de parada está establecida, False en caso contrario
        """
        return self.params["stop"]
    
    def scan_for_light(self):
        """
        Escanea para encontrar la dirección de la luz
        
        Returns:
            tuple: (mejor_ángulo, máximo_brillo)
        """
        brightness_readings = []
        
        # Guardar posición original del pan
        original_pan = self.robot.readPanPosition()
        
        # Escanear en diferentes ángulos
        for angle in self.pan_positions:
            self.robot.movePanTo(angle, 100)
            time.sleep(0.1)  # Tiempo para estabilizar
            # Tomar múltiples lecturas para mayor precisión
            reading_sum = 0
            readings = 3
            for _ in range(readings):
                reading_sum += self.robot.readBrightnessSensor()
                time.sleep(0.02)
            avg_reading = reading_sum / readings
            brightness_readings.append(avg_reading)
        
        # Restaurar posición original
        self.robot.movePanTo(original_pan, 100)
        
        # Encontrar posición más brillante
        max_brightness = max(brightness_readings)
        best_index = brightness_readings.index(max_brightness)
        best_angle = self.pan_positions[best_index]
        
        print(f"Escaneo de luz - Ángulos: {self.pan_positions}")
        print(f"Lecturas: {[round(b, 1) for b in brightness_readings]}")
        print(f"Mejor ángulo: {best_angle}, Brillo: {max_brightness}")
        
        return best_angle, max_brightness