from pynput import keyboard
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim


SPEED = 10 
# IP = '10.20.28.140'
IP = 'localhost'
TIME = 0.1

def on_press(key):
    try:
        # Para teclas de caracteres normales
        if key == keyboard.KeyCode.from_char('w'):
            robobo.moveWheels(SPEED, SPEED)
            print('Moving forward')
        elif key == keyboard.KeyCode.from_char('s'):
            robobo.moveWheels(-SPEED, -SPEED)
            print('Moving backward')
        elif key == keyboard.KeyCode.from_char('a'):
            robobo.moveWheels(SPEED, -SPEED)
            print('Turning left')
        elif key == keyboard.KeyCode.from_char('d'):
            robobo.moveWheels(-SPEED, SPEED)
            print('Turning right')
        elif key == keyboard.KeyCode.from_char('q'):
            robobo.stopMotors()
            sim.disconnect()
            return False

    except AttributeError:
        # Maneja teclas especiales que no tienen atributo 'char'
        pass

def on_release(key):
    robobo.stopMotors()
    print('Stop')

def listen_keyboard():
    with keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True) as listener:
        listener.join()  # Esto mantiene el listener activo

if __name__ == '__main__':
    sim = RoboboSim(IP)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(IP)
    robobo.connect()
    
    try:
        listen_keyboard()
    except KeyboardInterrupt:
        robobo.stopMotors()
        sim.disconnect()
