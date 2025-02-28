from robobopy.Robobo import Robobo, Wheels
from robobosim.RoboboSim import RoboboSim
from time import sleep
'''
si no detecta, girar hasta que detecte,
si detecta, acercarse
si ya lo detectó ir a por otro
'''

def centrarQr(qr):
    global pan, tilt
    if qr.x < xCENTRO and pan < limiteSuperiorPan: 
        pan += 1 # en el caso de que queramos acercarnos, intentamos centrar el centro del QR al punto (250,300).
    elif pan > limiteInferiorPan: 
        pan -= 1 
    if qr.y < yCENTRO and tilt > limiteInferiorTilt: 
        tilt -= 3
    elif tilt < limiteSuperiorTilt: 
        tilt += 3

    robobo.movePanTo(pan, 100)
    robobo.moveTiltTo(tilt, 100)

def esta_centradoQr(qr):
    return (xCENTRO -xDESFASE < qr.x < xCENTRO + xDESFASE ) and (yCENTRO - yDESFASE < qr.y < yCENTRO + yDESFASE)

def peligro(izquierda):
    if izquierda:
        robobo.moveWheelsByTime(SPEED, 0, 5)
        robobo.moveWheels(SPEED, SPEED)
    else:
        robobo.moveWheelsByTime(0, SPEED, 5)
        robobo.moveWheels(SPEED, SPEED)

def velocidad(velocidad):
    robobo.moveWheels(velocidad * 0.5, velocidad * 0.5)

def peatones():
    robobo.sayText('Hola peatones!')
    robobo.wait(1)

def parar():
    robobo.stopMotors()

def ceda():
    robobo.stopMotors()
    robobo.wait(3)
    robobo.moveWheels(SPEED, SPEED)

def rotonda():
    robobo.moveWheels(-SPEED, SPEED)

def qrDetectedCallback():
    '''
    devuelve True cuando ya estamos cerca suficiente (momento de ir a por otro QR),
    devuelve False cuando aun no estamos cerca suficiente (debemos acercarnos con calma).
    '''
    global pan, tilt, ultimoQR, ultimaVistaQR

    ultimaVistaQR = 0

    qr = robobo.readQR()

    if not ultimoQR: # nunca vio un QR, le asignamos el que está viendo ahora
        ultimoQR = qr.id
    if ultimoQR == qr.id: # si vemos el que estábamos viendo es porque estamos en el proceso de acercarnos, para ir controladamente paramos los motores.
        robobo.stopMotors()

        centrarQr(qr)

        if esta_centradoQr(qr): # si está suficientemente en el centro nos acercamos
            robobo.moveWheelsByTime(5,5,5)
        if qr.distance > distanciaCerca: # si está lo cuficientemente cerca, cantamos victoria y empezamos a girar
            print(f'<< Señal: {qr.id}>>')
            match qr.id:
                case 'peligro izquierda': peligro(izquierda=True)
                case 'peligro derecha': peligro(izquierda=False)
                case '10': velocidad(10)
                case '20': velocidad(20)
                case '40': velocidad(40)
                case '50': velocidad(50)
                case 'peatones': peatones()
                case 'parar': parar()
                case 'rotonda': rotonda()
                case 'ceda': ceda()

            return True
    else: return False
    

if __name__ == "__main__":

    pan = 0
    tilt = 90
    SPEED =5 
    xCENTRO = 250
    xDESFASE = 30
    yCENTRO = 300
    yDESFASE = 65
    limiteSuperiorPan = 160
    limiteInferiorPan = -160 
    limiteSuperiorTilt = 106
    limiteInferiorTilt = 4
    distanciaCerca = 20
    ultimoQR = None
    ultimaVistaQR = 0

    robobo = Robobo('localhost')
    robobo.connect()
    t = 0
    
    robobo.whenAQRCodeIsDetected(qrDetectedCallback)
    
    try:
        while True:
            robobo.moveWheels(SPEED, SPEED)
            ultimaVistaQR += 1
            if ultimaVistaQR > 15:
                pan, tilt = 0, 90
                robobo.movePanTo(pan, 100)
                robobo.moveTiltTo(tilt, 100)

            sleep(1)
    except KeyboardInterrupt:
        robobo.stopMotors()
        robobo.disconnect()
