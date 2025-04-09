from robobo_controller import RoboboController
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from tkinter import simpledialog


class TextRoboboController(RoboboController):
    """
    Controlador de Robobo que recibe comandos a través de un cuadro de texto.
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


def main(ip="localhost"):
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = TextRoboboController(robobo, sim)

    print("Comandos disponibles: forward, back, left, right, stop, quit")
    print("Comando compuesto: forward-left, forward-right, etc.")
    print("Comando con velocidad: 'forward speed 30'")
    print("Comando con tiempo: 'forward time 3.5'")

    try:
        while True:
            # Obtener comando del usuario
            user_response = controller.get_command()

            if user_response is None:
                print("Exiting program")
                break

            result = controller.parse_command(user_response, language="en")

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
