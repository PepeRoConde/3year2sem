

# usar read line
# hacer acion particualr señal
# match case


from robobopy.Robobo import Robobo, Wheels
from robobosim.RoboboSim import RoboboSim
from time import sleep
'''
si no detecta, girar hasta que detecte,
si detecta, acercarse
si ya lo detectó ir a por otro
'''
pan = 0
tilt = 90
SPEED = 3
ultimoQR = None

def qrDetectedCallback():
    t = 0
    '''
    devuelve True cuando ya estamos cerca suficiente (momento de ir a por otro QR),
    devuelve False cuando aun no estamos cerca suficiente (debemos acercarnos con calma).
    '''
    global pan, tilt, ultimoQR
    qr = robobo.readQR()
    if ultimoQR == None: # nunca vio un QR, le asignamos el que está viendo ahora
        ultimoQR = qr.id
    if ultimoQR == qr.id: # si vemos el que estábamos viendo es porque estamos en el proceso de acercarnos, para ir controladamente paramos los motores.
        robobo.stopMotors()
        print(f'pan: {pan}, tilt: {tilt}, x: {qr.x}, y: {qr.y}, p1: {qr.p1}, p2: {qr.p2}, p3: {qr.p3}, distancia: {qr.distance}, id: {qr.id}')
        if qr.x < 250 and pan < 160: pan += 1 # en el caso de que queramos acercarnos, intentamos centrar el centro del QR al punto (250,300).
        elif pan > -160: pan -= 1 
        # iniciar 300 variable
        if qr.y < 300 and tilt > 4: tilt -= 3
        elif tilt < 106: tilt += 3
        robobo.movePanTo(pan, 100)
        robobo.moveTiltTo(tilt, 100)

        if (220 < qr.x < 280 ) and (235 < qr.y < 365): # si está suficientemente en el centro nos acercamos
            robobo.moveWheelsByTime(5,5,5)
        if qr.distance > 20: # si está lo cuficientemente cerca, cantamos victoria y empezamos a girar
            print(f'<< Señal: {qr.id}, distancia: {qr.distance} >>')
            robobo.moveWheelsByTime(30,-30,0.5)
            return True
    else: return False
    

if __name__ == "__main__":

    robobo = Robobo('localhost')
    robobo.connect()
    t = 0
    
    robobo.whenAQRCodeIsDetected(qrDetectedCallback)
    
    try:
        robobo.moveWheels(SPEED, SPEED)
        while True:
            t += 1
            sleep(1)
            print(f't: {t}')
            #ultimoQR = None
            #robobo.moveWheelsByTime(10,-10,0.5)
            sleep(1)
    except KeyboardInterrupt:
        robobo.stopMotors()
        robobo.disconnect()
