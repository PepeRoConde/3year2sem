from pynput import keyboard
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim


SPEED = 10         
IP = "localhost"   
TIME = 0.1         

def on_press(key):
    """
    Callback que se ejecuta cuando se presiona una tecla.
    
    Args:
        key: La tecla presionada
    
    Returns:
        False si se debe detener el listener (tecla Q), True en caso contrario
    """
    try:
        if key == keyboard.KeyCode.from_char("w"):
            robobo.moveWheels(SPEED, SPEED)
            print("Moving forward")
        elif key == keyboard.KeyCode.from_char("s"):
            robobo.moveWheels(-SPEED, -SPEED)
            print("Moving backward")
        elif key == keyboard.KeyCode.from_char("a"):
            robobo.moveWheels(SPEED, -SPEED)
            print("Turning left")
        elif key == keyboard.KeyCode.from_char("d"):
            robobo.moveWheels(-SPEED, SPEED)
            print("Turning right")
        elif key == keyboard.KeyCode.from_char("q"):
            robobo.stopMotors()
            sim.disconnect()
            print("Stopping and exiting...")
            return False

    except AttributeError:
        # Maneja teclas especiales que no tienen atributo 'char'
        print("Select one of the options: w, a , s, d, q")
        pass
    
    return True


def on_release(key):
    """
    Callback que se ejecuta cuando se suelta una tecla.
    Detiene los motores del robot al soltar cualquier tecla.
    
    Args:
        key: La tecla liberada
    """
    robobo.stopMotors()
    print("Stop")


def listen_keyboard():
    """
    Listener del teclado.
    """
    with keyboard.Listener(
            on_press=on_press, on_release=on_release, suppress=True  # type: ignore
    ) as listener:
        print("Control de teclado:")
        print("  W: Adelante")
        print("  S: Atrás")
        print("  A: Izquierda")
        print("  D: Derecha")
        print("  Q: Salir")
        
        listener.join()  # Mantiene el listener activo


if __name__ == "__main__":
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()
    
    robobo = Robobo(IP)
    robobo.connect()
    
    try:
        listen_keyboard()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
    finally:
        robobo.stopMotors()
        sim.disconnect()
