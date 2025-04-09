# Prompt para control de Robobo con ChatGPT

Necesito que extraigas comandos de movimiento para un robot a partir de frases en lenguaje natural. 

Los comandos permitidos son:
- forward: para mover hacia adelante
- back: para mover hacia atrás
- left: para girar a la izquierda
- right: para girar a la derecha
- stop: para detener el movimiento

También puedes combinar los comandos de dirección para movimientos compuestos como:
- forward-left: para avanzar girando a la izquierda
- forward-right: para avanzar girando a la derecha
- back-left: para retroceder girando a la izquierda
- back-right: para retroceder girando a la derecha

Además, puedes incluir parámetros:
- speed X: donde X es un número entero que indica la velocidad
- time Y: donde Y es un número que indica los segundos de movimiento

Ejemplos:
- "Avanza un poco": forward
- "Retrocede a velocidad 30": back speed 30
- "Gira a la derecha durante 3 segundos": right time 3
- "Avanza girando a la izquierda a velocidad 25": forward-left speed 25

Responde ÚNICAMENTE con el comando extraído, sin explicaciones ni formato adicional.
