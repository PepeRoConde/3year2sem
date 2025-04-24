import random
import time
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim

# Constantes
ALPHA = 0.6  # Tasa de aprendizaje
GAMMA = 0.7  # Factor de descuento
EPSILON = 0.2  # Probabilidad de exploración (vs explotación)
MAX_LUX = 350  # Nivel de luz para considerar que se ha llegado al objetivo
NUM_EPISODES = 5  # Número de episodios de entrenamiento
SPEED = 30  # Velocidad de movimiento

# Definición de estados
# Estado 0: Luz muy a la izquierda
# Estado 1: Luz poco a la izquierda
# Estado 2: Luz al centro
# Estado 3: Luz poco a la derecha
# Estado 4: Luz muy a la derecha
# Estado 5: Estado no válido (sin luz visible)

# Definición de acciones
# Acción 0: Girar mucho a la derecha
# Acción 1: Girar poco a la derecha
# Acción 2: Avanzar recto
# Acción 3: Girar poco a la izquierda
# Acción 4: Girar mucho a la izquierda

def initialize_q_table():
    """
    Inicializa la tabla Q con valores aleatorios pequeños, tanto positivos como negativos
    """
    # 6 estados (5 direcciones + 1 no válido) y 5 acciones
    # Inicializamos con valores entre -0.1 y 0.1
    q_table = np.random.random((6, 5)) * 0.2 - 0.1
    return q_table

def update_q_table(q_table, state, action, reward, new_state):
    """
    Actualiza la tabla Q usando la ecuación de Bellman
    Permite valores negativos para penalizar malas acciones
    """
    # Calcular el valor Q máximo para el nuevo estado
    max_future_q = np.max(q_table[new_state])
    
    # Valor Q actual
    current_q = q_table[state, action]
    
    # Aplicar la ecuación de actualización de Q-learning
    new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
    
    # Actualizar la tabla Q (permitiendo valores negativos)
    q_table[state, action] = new_q
    
    return q_table

def take_action(rob, action):
    """
    Ejecuta la acción seleccionada con acciones graduales según los nuevos estados
    Con sistema de recompensas negativas para malas acciones
    """
    initial_brightness = rob.readBrightnessSensor()
    
    # Si la luz inicial es muy baja, intentamos reorientarnos antes de actuar
    if initial_brightness < 10:
        print("Brillo inicial muy bajo, buscando mejor orientación...")
        new_state = detailed_scan_for_light(rob)
        if new_state != 5:  # Si encontramos luz
            print(f"Encontrada mejor orientación, nuevo estado: {new_state}")
            # Adaptamos la acción según la dirección detectada
            if new_state == 0 or new_state == 1:  # Luz a la izquierda
                action = 4 if new_state == 0 else 3  # Girar hacia la izquierda
            elif new_state == 3 or new_state == 4:  # Luz a la derecha
                action = 0 if new_state == 4 else 1  # Girar hacia la derecha
    
    # Comprobar riesgo de caída antes de moverse
    if check_fall_risk(rob):
        print("¡Precaución! Detectado posible borde. Retrocediendo...")
        # Retroceder un poco para evitar la caída
        rob.moveWheelsByTime(-SPEED, -SPEED, 1.25)
        time.sleep(0.3)
        
        # Girar para cambiar de dirección
        rob.moveWheelsByTime(SPEED, -SPEED, 0.4)
        time.sleep(0.3)
        
        return -0.1, False  # Recompensa negativa por evitar caída
    
    # Ejecutar la acción con ajustes graduales basados en la acción
    if action == 0:  # Girar mucho a la derecha
        print("Girando mucho a la derecha")
        rob.moveWheelsByTime(-SPEED, SPEED, 0.7)
        
    elif action == 1:  # Girar poco a la derecha
        print("Girando poco a la derecha")
        rob.moveWheelsByTime(-SPEED // 2, SPEED // 2, 0.4)
        
    elif action == 2:  # Avanzar recto
        print("Avanzando recto")
        
        # Implementar movimiento adaptativo
        rob.moveWheels(SPEED, SPEED)
        
        # Monitorear sensores durante un tiempo máximo
        max_move_time = 3.0  # Tiempo máximo de movimiento en segundos
        start_time = time.time()
        
        while time.time() - start_time < max_move_time:
            # Comprobar si hay riesgo de caída durante el movimiento
            if check_fall_risk(rob):
                print("Detectado borde durante avance. Deteniendo...")
                rob.stopMotors()
                time.sleep(0.1)
                # Retroceder ligeramente por seguridad
                rob.moveWheelsByTime(-SPEED, -SPEED, 0.3)
                time.sleep(0.2)
                break
            
            collision, obstruction_direction = check_collision_risk(rob)
            # Comprobar si hay riesgo de colisión
            if collision:
                handle_collision(rob, obstruction_direction)
                break
                
            # Comprobar si perdemos la luz durante el movimiento
            current_brightness = rob.readBrightnessSensor()
            if current_brightness < 10 and initial_brightness > 20:
                print("Perdiendo visibilidad de la luz durante el avance, ajustando...")
                rob.stopMotors()
                time.sleep(0.1)
                break
   
            # Breve pausa para no saturar los sensores
            time.sleep(0.1)
        
        # Asegurarse de que los motores se detengan
        rob.stopMotors()
        
    elif action == 3:  # Girar poco a la izquierda
        print("Girando poco a la izquierda")
        rob.moveWheelsByTime(SPEED // 2, -SPEED // 2, 0.4)
        
    elif action == 4:  # Girar mucho a la izquierda
        print("Girando mucho a la izquierda")
        rob.moveWheelsByTime(SPEED, -SPEED, 0.7)
    
    time.sleep(0.3)  # Pequeña pausa para estabilizar
    
    # Leer el nuevo nivel de brillo
    new_brightness = rob.readBrightnessSensor()
    print(f"Brillo inicial: {initial_brightness}, Brillo nuevo: {new_brightness}")
    
    # Si perdimos significativamente la luz, intentar recuperarla
    if new_brightness < 10 and initial_brightness > 20:
        print("La luz se ha perdido, intentando recuperarla...")
        recover_state = detailed_scan_for_light(rob)
        if recover_state != 5:  # Si encontramos luz
            print("Luz recuperada")
            new_brightness = rob.readBrightnessSensor()
            print(f"Nuevo brillo tras recuperación: {new_brightness}")
    
    # Calcular la recompensa basada en el cambio de brillo
    brightness_change = new_brightness - initial_brightness
    
    # Recompensas ajustadas para permitir valores negativos como en la imagen
    if brightness_change > 50:  # Mejora muy significativa
        reward = 1.0
    elif brightness_change > 20:  # Mejora significativa
        reward = 0.5
    elif brightness_change > 0:  # Mejora leve
        reward = 0.2
    elif brightness_change >= -5:  # Cambio mínimo (casi igual)
        reward = -0.1
    else:  # Empeoramiento
        reward = -0.2
    
    # Comprobar si se ha alcanzado el objetivo
    if new_brightness >= MAX_LUX:
        reward = 1.0  # Recompensa máxima por alcanzar el objetivo
        return reward, True
    
    return reward, False
def get_state(rob):
    """
    Determina el estado actual basado en lecturas más detalladas de dirección
    """
    try:
        # Leer los sensores de luz inicialmente
        brightness = rob.readBrightnessSensor()
        
        # Si no hay suficiente luz, realizamos un barrido completo para buscarla
        if brightness < 5:
            print("Luz inicial muy baja, realizando barrido completo...")
            return detailed_scan_for_light(rob)
        
        # Barrido en 5 posiciones para determinar la dirección más precisa
        pan_positions = [-90, -45, 0, 45, 90]
        brightness_readings = []
        
        for angle in pan_positions:
            rob.movePanTo(angle, 50)
            time.sleep(0.5)
            reading = rob.readBrightnessSensor()
            brightness_readings.append(reading)
            print(f"Ángulo {angle}°: Brillo = {reading}")
        
        # Volver a la posición central
        rob.movePanTo(0, 50)
        time.sleep(0.5)
        
        print(f"Lecturas de brillo detalladas: {brightness_readings}")
        
        # Si todas las lecturas son muy bajas, realizamos un barrido completo
        if max(brightness_readings) < 42:
            print("Todas las lecturas son muy bajas, realizando barrido completo...")
            return detailed_scan_for_light(rob)
        
        # Determinar el estado basado en qué posición tiene mayor brillo
        max_index = brightness_readings.index(max(brightness_readings))
        
        # Mapear el índice al estado (0-4)
        state = max_index
        
        state_names = ["Muy izquierda", "Poco izquierda", "Centro", "Poco derecha", "Muy derecha"]
        print(f"Estado: {state} - Luz {state_names[state]}")
        return state
            
    except Exception as e:
        print(f"Error al leer sensores: {e}")
        return 5  # Estado no válido

def detailed_scan_for_light(rob):
    """
    Realiza un barrido completo para encontrar la luz con mayor precisión
    Devuelve un estado entre 0-5 (5 direcciones + no válido)
    """
    print("Iniciando barrido detallado para buscar la luz...")
    max_brightness = 0
    best_state = 5  # Estado no válido por defecto
    
    # Primero giramos el robot 360 grados, deteniéndonos cada 90 grados
    for i in range(4):  # 0, 90, 180, 270 grados
        # Girar 90 grados
        rob.moveWheelsByTime(SPEED // 2, -SPEED // 2, 1.0)
        time.sleep(0.5)  # Esperar a que se estabilice
        
        # En cada posición del robot, movemos el pan para buscar luz en 5 posiciones
        pan_positions = [-90, -45, 0, 45, 90]
        
        for j, angle in enumerate(pan_positions):
            rob.movePanTo(angle, 50)
            time.sleep(0.5)
            
            brightness = rob.readBrightnessSensor()
            print(f"Rotación {i*90}°, Pan {angle}°: Brillo = {brightness}")
            
            if brightness > max_brightness:
                max_brightness = brightness
                best_state = j  # El estado es directamente el índice de la posición del pan
    
    # Volver el pan a la posición central
    rob.movePanTo(0, 50)
    time.sleep(0.5)
    
    if max_brightness < 10:
        print("No se encontró suficiente luz en el barrido completo")
        return 5  # Estado no válido
    else:
        state_names = ["Muy izquierda", "Poco izquierda", "Centro", "Poco derecha", "Muy derecha"]
        print(f"Luz encontrada! Estado: {best_state} ({state_names[best_state]}), Brillo máximo: {max_brightness}")
        return best_state

def check_collision_risk(rob):
    """
    Comprueba si hay riesgo de colisión y determina la dirección del obstáculo
    
    Returns:
        tuple: (hay_riesgo, dirección)
            - hay_riesgo: True si hay riesgo de colisión, False si es seguro
            - dirección: 'izquierda', 'derecha', 'centro' o None si no hay obstáculo
    """
    try:
        # Comprobar sensores IR frontales y laterales
        front_c = rob.readIRSensor(IR.FrontC)
        front_l = rob.readIRSensor(IR.FrontL)
        front_r = rob.readIRSensor(IR.FrontR)
        
        # También comprobar los sensores laterales para tener más información
        front_ll = rob.readIRSensor(IR.FrontLL)
        front_rr = rob.readIRSensor(IR.FrontRR)
        
        # Umbral para determinar si hay un obstáculo cerca (valores altos indican proximidad)
        collision_threshold = 100
        
        # Valores para comparar la intensidad en diferentes direcciones
        left_intensity = max(front_l, front_ll)
        right_intensity = max(front_r, front_rr)
        center_intensity = front_c
        
        # Determinar si hay riesgo
        if front_c > collision_threshold or front_l > collision_threshold or front_r > collision_threshold:
            # Determinar la dirección del obstáculo
            obstruction_direction = None
            
            # Si el obstáculo está principalmente a la izquierda
            if left_intensity > right_intensity and left_intensity > center_intensity:
                obstruction_direction = 'izquierda'
                print(f"¡Riesgo de colisión detectado a la IZQUIERDA! Valores: C={front_c}, L={front_l}, LL={front_ll}, R={front_r}, RR={front_rr}")
            
            # Si el obstáculo está principalmente a la derecha
            elif right_intensity > left_intensity and right_intensity > center_intensity:
                obstruction_direction = 'derecha'
                print(f"¡Riesgo de colisión detectado a la DERECHA! Valores: C={front_c}, L={front_l}, LL={front_ll}, R={front_r}, RR={front_rr}")
            
            # Si el obstáculo está principalmente al centro
            else:
                obstruction_direction = 'centro'
                print(f"¡Riesgo de colisión detectado al CENTRO! Valores: C={front_c}, L={front_l}, LL={front_ll}, R={front_r}, RR={front_rr}")
            
            return True, obstruction_direction
                
        return False, None
        
    except Exception as e:
        print(f"Error al comprobar riesgo de colisión: {e}")
        return False, None

def handle_collision(rob, obstruction_direction):
    """
    Maneja la evasión de obstáculos según la dirección detectada
    
    Args:
        rob: Objeto Robobo
        obstruction_direction: Dirección del obstáculo ('izquierda', 'derecha', 'centro' o None)
    """
    # Si no hay dirección de obstrucción definida, retrocede y gira aleatoriamente
    if obstruction_direction is None:
        # Retroceder un poco
        rob.moveWheelsByTime(-SPEED, -SPEED, 0.5)
        time.sleep(0.2)
        
        # Girar aleatoriamente
        if random.random() < 0.5:
            rob.moveWheelsByTime(SPEED, -SPEED, 0.7)  # Girar a la izquierda
        else:
            rob.moveWheelsByTime(-SPEED, SPEED, 0.7)  # Girar a la derecha
            
        return
    
    # Retroceder primero en cualquier caso
    rob.moveWheelsByTime(-SPEED, -SPEED, 0.5)
    time.sleep(0.3)
    
    # Girar en la dirección opuesta al obstáculo
    if obstruction_direction == 'izquierda':
        # Obstáculo a la izquierda, girar a la derecha
        print("Obstáculo a la izquierda: girando a la DERECHA")
        rob.moveWheelsByTime(-SPEED, SPEED, 0.8)
    elif obstruction_direction == 'derecha':
        # Obstáculo a la derecha, girar a la izquierda
        print("Obstáculo a la derecha: girando a la IZQUIERDA")
        rob.moveWheelsByTime(SPEED, -SPEED, 0.8)
    else:  # centro
        # Obstáculo al frente, girar en dirección aleatoria pero más pronunciado
        print("Obstáculo al centro: evasión completa")
        # Retroceder un poco más
        rob.moveWheelsByTime(-SPEED, -SPEED, 0.3)
        time.sleep(0.2)
        
        # Giro más pronunciado en dirección aleatoria
        if random.random() < 0.5:
            rob.moveWheelsByTime(SPEED, -SPEED, 1.0)  # Girar más a la izquierda
        else:
            rob.moveWheelsByTime(-SPEED, SPEED, 1.0)  # Girar más a la derecha
    
    time.sleep(0.3)  # Pausa para estabilizar

def check_fall_risk(rob):
    """
    Comprueba si hay riesgo de caída usando los sensores IR frontales
    """
    try:
        # Leer los sensores IR frontales directamente
        front_l = rob.readIRSensor(IR.FrontL)
        front_r = rob.readIRSensor(IR.FrontR)
        
        # Imprimir los valores para diagnóstico
        print(f"Sensores frontales - Izquierdo: {front_l}, Derecho: {front_r}")
        
        # Valor mínimo de los dos sensores frontales
        min_front = min(front_l, front_r)
        
        # Umbral de distancia segura
        safe_distance = 10
        
        if min_front < safe_distance:
            print(f"¡Riesgo de caída detectado! Distancia mínima frontal: {min_front}")
            return True
                
        return False
        
    except Exception as e:
        print(f"Error al comprobar riesgo de caída: {e}")
        return False


def select_action(q_table, state, epsilon):
    """
    Selecciona una acción usando la política epsilon-greedy
    """
    if state == 5:  # Estado no válido
        return random.randint(0, 4)
    
    # Con probabilidad epsilon, exploramos (acción aleatoria)
    if random.random() < epsilon:
        return random.randint(0, 4)
    
    # Con probabilidad (1-epsilon), explotamos (mejor acción conocida)
    return np.argmax(q_table[state])


# Modificación de la función train_robot para recibir una tabla Q existente:
def train_robot(sim, initial_q_table=None):
    """
    Entrenamiento principal del robot usando Q-learning
    
    Args:
        sim: Objeto del simulador
        initial_q_table: Tabla Q inicial (opcional, si no se proporciona se inicializa una nueva)
    
    Returns:
        La mejor tabla Q aprendida
    """
    # Inicializar la conexión con Robobo
    print("Conectando con Robobo...")
    rob = Robobo('localhost')
    rob.connect()
    
    # Inicializar la tabla Q o usar la proporcionada
    q_table = initial_q_table if initial_q_table is not None else initialize_q_table()
    
    # Variables para seguimiento
    episode_rewards = []
    total_interactions = 0
    best_episode_reward = float('-inf')
    best_q_table = None
    
    print("Iniciando entrenamiento...")    
    # Bucle principal de entrenamiento
    for episode in range(NUM_EPISODES):
        episode_reward = 0
        goal_reached = False
        
        print(f"\n{'='*50}")
        print(f"Episodio {episode+1}/{NUM_EPISODES}")
        print(f"{'='*50}")
        
        # Restablecer la simulación al inicio de cada episodio
        sim.resetSimulation()
        time.sleep(1.5)  # Dar más tiempo a que se reinicie completamente
        
        # Determinar el estado inicial
        state = get_state(rob)
        print(f"Estado inicial: {state}")
        
        # Iterar hasta alcanzar el objetivo o un número máximo de pasos
        step = 0
        max_steps = 50
        
        while not goal_reached and step < max_steps:
            print(f"\n--- Paso {step+1} ---")
            
            # Seleccionar una acción
            action = select_action(q_table, state, EPSILON)
            print(f"Seleccionada acción {action}")
            
            # Ejecutar la acción y obtener la recompensa
            reward, goal_reached = take_action(rob, action)
            
            # Obtener el nuevo estado
            new_state = get_state(rob)
            
            # Actualizar la tabla Q
            q_table = update_q_table(q_table, state, action, reward, new_state)
            
            # Actualizar el estado
            state = new_state
            
            # Actualizar las estadísticas
            episode_reward += reward
            total_interactions += 1
            
            # Imprimir información de seguimiento
            print(f"Estado actual: {state}, Acción: {action}, Recompensa: {reward}")
            
            # Mostrar la tabla Q de manera más legible
            print("\nEstado de la tabla Q:")
            state_names = ["Muy izquierda", "Poco izquierda", "Centro", "Poco derecha", "Muy derecha", "No válido"]
            action_names = ["Girar mucho der", "Girar poco der", "Avanzar", "Girar poco izq", "Girar mucho izq"]
            
            for s in range(6):
                actions_values = q_table[s]
                print(f"  Estado {s} ({state_names[s]}): {actions_values}")
            
            step += 1
            
            # Si se alcanzó el objetivo, mostrar mensaje
            if goal_reached:
                print(f"\n¡OBJETIVO ALCANZADO en el paso {step}! 🎉")
                break
        
        # Guardar la recompensa del episodio
        episode_rewards.append(episode_reward)
        
        # Guardar la mejor tabla Q (la que obtuvo mayor recompensa)
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_q_table = q_table.copy()
        # Guardar la tabla cada 10 episodios
        if (episode + 1) % 10 == 0:
            save_q_table(q_table, f'q_table_episode_{episode+1}.npy')
            
        
        print(f"\nEpisodio {episode+1} completado.")
        print(f"Recompensa total del episodio: {episode_reward}")
        print(f"Mejor recompensa hasta ahora: {best_episode_reward}")
        print(f"Total de interacciones: {total_interactions}")
    
    # Desconectar el robot
    rob.disconnect()
    return best_q_table

def test_robot(q_table, sim):
    """
    Prueba el comportamiento del robot con la tabla Q aprendida
    """
    # Inicializar la conexión con Robobo
    print("\nConectando con Robobo para prueba final...")
    rob = Robobo('localhost')  # Cambiar por la IP correcta
    rob.connect()
    
    # Resetear la simulación
    sim.resetSimulation()
    time.sleep(1.5)  # Dar tiempo suficiente a que se reinicie
    
    print("\n" + "="*70)
    print("PRUEBA FINAL CON LA POLÍTICA APRENDIDA")
    print("="*70)
    
    # Variables para seguimiento
    goal_reached = False
    steps = 0
    max_steps = 100  # Aumentamos el número máximo de pasos para la prueba
    total_reward = 0
    
    # Imprimir la tabla Q final
    print("\nTabla Q final utilizada para la prueba:")
    state_names = ["Muy izquierda", "Poco izquierda", "Centro", "Poco derecha", "Muy derecha", "No válido"]
    action_names = ["Girar mucho der", "Girar poco der", "Avanzar", "Girar poco izq", "Girar mucho izq"]
    
    for s in range(6):
        actions_values = q_table[s]
        best_action = np.argmax(actions_values)
        print(f"  Estado {s} ({state_names[s]}): {actions_values} -> Mejor acción: {action_names[best_action]}")
    
    print("\nIniciando recorrido hacia la luz...")
    
    while not goal_reached and steps < max_steps:
        # Determinar el estado actual
        state = get_state(rob)
        print(f"\n--- Paso {steps+1} ---")
        print(f"Estado actual: {state} ({state_names[state] if state < len(state_names) else 'Desconocido'})")
        
        # Seleccionar la mejor acción (sin exploración)
        action = np.argmax(q_table[state])
        
        # Ejecutar la acción
        print(f"Ejecutando acción: {action} ({action_names[action]})")
        reward, goal_reached = take_action(rob, action)
        total_reward += reward
        
        steps += 1
        
        # Si se alcanzó el objetivo, mostrar mensaje
        if goal_reached:
            print(f"\n¡OBJETIVO ALCANZADO en el paso {steps}! 🎉")
            print(f"Total de recompensa acumulada: {total_reward}")
            break
    
    if not goal_reached:
        print("\nNo se pudo alcanzar el objetivo en el número máximo de pasos.")
        print(f"Total de recompensa acumulada: {total_reward}")
    
    # Desconectar el robot
    rob.disconnect()

def save_q_table(q_table, filename='q_table.npy'):
    """
    Guarda la tabla Q en un archivo
    
    Args:
        q_table: La tabla Q a guardar
        filename: Nombre del archivo donde guardar la tabla (por defecto 'q_table.npy')
    """
    try:
        np.save(filename, q_table)
        print(f"Tabla Q guardada exitosamente en '{filename}'")
    except Exception as e:
        print(f"Error al guardar la tabla Q: {e}")

def load_q_table(filename='q_table.npy', shape=(6, 5)):
    """
    Carga una tabla Q desde un archivo
    
    Args:
        filename: Nombre del archivo desde donde cargar la tabla (por defecto 'q_table.npy')
        shape: Forma esperada de la tabla Q (por defecto (6, 5) para 6 estados y 5 acciones)
    
    Returns:
        La tabla Q cargada o una nueva si no se pudo cargar el archivo
    """
    try:
        # Intentar cargar la tabla desde el archivo
        q_table = np.load(filename)
        print(f"Tabla Q cargada exitosamente desde '{filename}'")
        
        # Verificar que la forma sea correcta
        if q_table.shape != shape:
            print(f"Advertencia: La tabla cargada tiene forma {q_table.shape}, pero se esperaba {shape}")
            print("Se inicializará una nueva tabla Q con la forma correcta")
            q_table = initialize_q_table()
        
        return q_table
    except FileNotFoundError:
        print(f"No se encontró el archivo '{filename}'. Se inicializará una nueva tabla Q")
        return initialize_q_table()
    except Exception as e:
        print(f"Error al cargar la tabla Q: {e}")
        print("Se inicializará una nueva tabla Q")
        return initialize_q_table()

if __name__ == "__main__":
    
    sim = RoboboSim("localhost")
    sim.connect()
    sim.resetSimulation()

    try:
        # Cargar una tabla Q existente o crear una nueva
        q_table = load_q_table()
        
        # Entrenar al robot
        print("=== ENTRENAMIENTO DEL ROBOT ===")
        learned_q_table = train_robot(sim, q_table)  # Pasar la tabla Q cargada
        
        # Guardar la tabla Q aprendida
        save_q_table(learned_q_table, 'q_table_final.npy')
        
        # Probar el comportamiento aprendido
        print("\n=== PRUEBA FINAL DEL ROBOT ===")
        test_robot(learned_q_table, sim)
        
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
        # Guardar la tabla Q incluso si se interrumpe
        if 'learned_q_table' in locals():
            save_q_table(learned_q_table, 'q_table_interrupted.npy') # type: ignore
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\nPrograma finalizado.")

