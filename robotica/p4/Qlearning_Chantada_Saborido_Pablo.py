import random
import time
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
from utils import load_q_table, save_q_table, print_q_table

ALPHA = 0.6         # Tasa de aprendizaje
GAMMA = 0.7         # Factor de descuento
EPSILON = 0.2       # Probabilidad de exploración 
MAX_LUX = 350       
NUM_EPISODES = 5    
NUM_EPOCHS = 100
SPEED = 30  

def initialize_q_table():
    """
    Inicializa la tabla Q con valores aleatorios pequeños
    """
    # Inicializamos con valores entre -0.1 y 0.1 para mejor exploración inicial
    q_table = np.random.random((6, 5)) * 0.2 - 0.1
    return q_table

def update_q_table(q_table, state, action, reward, new_state):
    """
    Actualiza la tabla Q usando la ecuación de Bellman
    Usa valores negativos para penalizar malas acciones
    """
    # Calcular el valor Q máximo para el nuevo estado
    max_future_q = np.max(q_table[new_state])
    
    # Valor Q actual
    current_q = q_table[state, action]
    
    # Aplicar la ecuación 
    new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
    
    # Actualizar la tabla  
    q_table[state, action] = new_q
    
    return q_table

def take_action(rob, action):
    """
    Ejecuta la acción seleccionada 
    """
    try:
        initial_brightness = rob.readBrightnessSensor()
        
        # # Si la luz inicial es muy baja, intentamos reorientarnos antes de actuar
        # if initial_brightness < 10:
        #     print("Brillo inicial muy bajo, buscando mejor orientación...")
        #     new_state = detailed_scan_for_light(rob)
        #     if new_state != 5:  # Si encontramos luz
        #         print(f"Encontrada mejor orientación, nuevo estado: {new_state}")
        #         # Adaptamos la acción según la dirección detectada
        #         if new_state == 0 or new_state == 1:  # Luz a la izquierda
        #             action = 4 if new_state == 0 else 3  # Girar hacia la izquierda
        #         elif new_state == 3 or new_state == 4:  # Luz a la derecha
        #             action = 0 if new_state == 4 else 1  # Girar hacia la derecha
         
        # Comprobar caidas
        if check_fall_risk(rob):
            front_l = rob.readIRSensor(IR.FrontL)
            front_r = rob.readIRSensor(IR.FrontR)

            rob.moveWheelsByTime(-SPEED, -SPEED, 1.25)
            
            # Girar para cambiar de dirección
            if front_l > front_r:
                print("Caida a la izquierda, girando a la derecha")
                rob.moveWheelsByTime(-SPEED, SPEED, 0.4)
            else:
                print("Caida a la derecha, girando a la izquierda")
                rob.moveWheelsByTime(SPEED, -SPEED, 0.4)
            time.sleep(0.1)
            
            return -0.1, False  # Recompensa negativa por evitar caída
        
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
            
            # Tiempo máximo de movimiento en segundos
            max_move_time = 3.0          
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
                    
                time.sleep(0.1)
            
            rob.stopMotors()
            
        elif action == 3:  # Girar poco a la izquierda
            print("Girando poco a la izquierda")
            rob.moveWheelsByTime(SPEED // 2, -SPEED // 2, 0.4)
            
        elif action == 4:  # Girar mucho a la izquierda
            print("Girando mucho a la izquierda")
            rob.moveWheelsByTime(SPEED, -SPEED, 0.7)
        
        time.sleep(0.3)  
        
        # Leer el nuevo nivel de brillo
        new_brightness = rob.readBrightnessSensor()
        print(f"Brillo inicial: {initial_brightness}, Brillo nuevo: {new_brightness}")
        
        # Si perdimos significativamente la luz, intentar recuperarla
        # if new_brightness < 10 and initial_brightness > 20:
        #     print("La luz se ha perdido, intentando recuperarla...")
        #     recover_state = detailed_scan_for_light(rob)
        #     if recover_state != 5:  # Si encontramos luz
        #         print("Luz recuperada")
        #         new_brightness = rob.readBrightnessSensor()
        #         print(f"Nuevo brillo tras recuperación: {new_brightness}")

        # Calcular la recompensa basada en el cambio de brillo
        brightness_change = new_brightness - initial_brightness
        
        if brightness_change > 50:  # Mejora muy significativa
            reward = 1.0
        elif brightness_change > 20:  # Mejora significativa
            reward = 0.5
        elif brightness_change > 0:  # Mejora leve
            reward = 0.2
        elif brightness_change >= -5:  # Cambio mínimo 
            reward = -0.1
        else:  # Empeoramiento
            reward = -0.2
        
        # Comprobar si se ha alcanzado el objetivo
        if new_brightness >= MAX_LUX:
            reward = 1.0  
            return reward, True
        
        return reward, False

    except Exception as e:
        print(f"Error en take_action: {e}")
        return -0.1, False

def get_state(rob):
    """
    Determina el estado actual según la lectura del sensor de luz
    """
    try:
        # Leer los sensores de luz inicialmente
        # brightness = rob.readBrightnessSensor()
        # 
        # # Si no hay suficiente luz, realizamos un barrido completo para buscarla
        # if brightness < 5:
        #     print("Luz inicial muy baja, realizando barrido completo...")
        #     return detailed_scan_for_light(rob)
            
        # Rcogemos los valores de los sensores en 5 posiciones
        pan_positions = [-90, -45, 0, 45, 90]
        brightness_readings = []
        
        for angle in pan_positions:
            # print(f"Moviendo pan a {angle} grados...")
            rob.movePanTo(angle, 100)  
            time.sleep(0.5)  
            reading = rob.readBrightnessSensor()
            brightness_readings.append(reading)
            # print(f"Ángulo {angle}°: Brillo = {reading}")
        
        # Reiniciar el pan a la posición central
        rob.movePanTo(0, 100)  
        
        print(f"Lecturas de brillo: {brightness_readings}")
        
        # # Si todas las lecturas son muy bajas, realizamos un barrido completo
        # if max(brightness_readings) <= 40:
        #     print("Todas las lecturas son muy bajas, realizando barrido completo...")
        #     return detailed_scan_for_light(rob)

        # Determinar el estado basado en qué posición tiene mayor brillo
        state_index = brightness_readings.index(max(brightness_readings))
        
        state_names = ["Muy izquierda", "Poco izquierda", "Centro", "Poco derecha", "Muy derecha"]
        print(f"Estado: {state_index} - Luz {state_names[state_index]}")
        return state_index

    except Exception as e:
        print(f"Error en get_state: {e}")
        return 5  # Estado no válido en caso de error

# def detailed_scan_for_light(rob):
#     """
#     Realiza un barrido completo para encontrar la luz con mayor precisión
#     Devuelve un estado entre 0-5 
#     """
#     try:
#         print("Iniciando barrido detallado para buscar la luz...")
#         max_brightness = 0
#         best_state = 5  # Estado no válido por defecto
#         
#         # Realizamos un giro de 360 grados, parando cada 90
#         for i in range(4):  # 0, 90, 180, 270 grados
#             # Girar 90 grados
#             rob.moveWheelsByTime(SPEED // 2, -SPEED // 2, 1.0)
#             time.sleep(0.5)  # Esperar a que se estabilice
#             
#             # Obtenemos los valores de los sensores en 5 posiciones
#             pan_positions = [-90, -45, 0, 45, 90]
#             
#             for j, angle in enumerate(pan_positions):
#                 rob.movePanTo(angle, 100)  
#                 time.sleep(0.1)
#                 
#                 brightness = rob.readBrightnessSensor()
#                 # print(f"Rotación {i*90}°, Pan {angle}°: Brillo = {brightness}")
#                 
#                 if brightness > max_brightness:
#                     max_brightness = brightness
#                     # El indice de la posición del pan, se corresponde con el estado
#                     best_state = j
#                     
#         # Reiniciar las posicion del pan
#         rob.movePanTo(0, 100)  
#         
#         if max_brightness < 10:
#             print("No se encontró suficiente luz en el barrido completo")
#             return 5  # Estado no válido
#         else:
#             state_names = ["Muy izquierda", "Poco izquierda", "Centro", "Poco derecha", "Muy derecha"]
#             print(f"Luz encontrada! Estado: {best_state} ({state_names[best_state]}), Brillo máximo: {max_brightness}")
#             return best_state

#     except Exception as e:
#         print(f"Error en detailed_scan_for_light: {e}")
#         return 5  # Estado no válido en caso de error

def check_collision_risk(rob):
    """
    Comprueba si hay riesgo de colisión y determina la dirección del obstáculo
    
    Returns:
        tuple: 
            - riesgo: True si hay riesgo de colisión, False si es seguro
            - dirección: 'izquierda', 'derecha', 'centro' o None si no hay obstáculo
    """
    try:
        # Comprobar sensores IR frontales y laterales
        front_c = rob.readIRSensor(IR.FrontC)
        front_l = rob.readIRSensor(IR.FrontL)
        front_r = rob.readIRSensor(IR.FrontR)
        front_ll = rob.readIRSensor(IR.FrontLL)
        front_rr = rob.readIRSensor(IR.FrontRR)
        
        collision_threshold = 100
        
        left_intensity = max(front_l, front_ll)
        right_intensity = max(front_r, front_rr)
        center_intensity = front_c
        
        # Determinar si hay riesgo
        if front_c > collision_threshold or front_l > collision_threshold or front_r > collision_threshold:
            obstruction_direction = None
            
            # IZQUIERDA
            if left_intensity > right_intensity and left_intensity > center_intensity:
                obstruction_direction = 'izquierda'
            
            # DERECHA 
            elif right_intensity > left_intensity and right_intensity > center_intensity:
                obstruction_direction = 'derecha'
            
            # CENTRO
            else:
                obstruction_direction = 'centro'
            
            print(f"Riesgo de colisión detectado: {obstruction_direction}")
            return True, obstruction_direction
                
        return False, None

    except Exception as e:
        print(f"Error en check_collision_risk: {e}")
        return False, None
    
def handle_collision(rob, obstruction_direction):
    """
    Maneja la evasión de obstáculos según la dirección detectada

    Args:
        rob:  Robobo
        obstruction_direction: Dirección del obstáculo ('izquierda', 'derecha', 'centro' o None)
    """
    try:
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

        # Retroceder 
        rob.moveWheelsByTime(-SPEED, -SPEED, 0.5)
        time.sleep(0.3)

        # Girar en la dirección opuesta al obstáculo
        if obstruction_direction == 'izquierda':
            print("Obstáculo a la izquierda: girando a la DERECHA")
            rob.moveWheelsByTime(-SPEED, SPEED, 0.8)
        elif obstruction_direction == 'derecha':
            print("Obstáculo a la derecha: girando a la IZQUIERDA")
            rob.moveWheelsByTime(SPEED, -SPEED, 0.8)
        else:  
            print("Obstáculo al centro: evasión completa")
            # Retroceder un poco más
            rob.moveWheelsByTime(-SPEED, -SPEED, 0.3)
            time.sleep(0.2)
            
            # Giro en dirección aleatoria
            if random.random() < 0.5:
                rob.moveWheelsByTime(SPEED, -SPEED, 1.0)  # Girar a la izquierda
            else:
                rob.moveWheelsByTime(-SPEED, SPEED, 1.0)  # Girar a la derecha
        
        time.sleep(0.3)  # Pausa para estabilizar
                
    except Exception as e:
        print(f"Error en handle_collision: {e}")

def check_fall_risk(rob):
    """
    Comprueba si hay riesgo de caída usando los sensores IR frontales
    """
    try:
        # Leer los sensores IR frontales directamente
        front_l = rob.readIRSensor(IR.FrontL)
        front_r = rob.readIRSensor(IR.FrontR)
        
        # Imprimir los valores para diagnóstico
        # print(f"Sensores frontales - Izquierdo: {front_l}, Derecho: {front_r}")
        
        # Valor mínimo de los dos sensores frontales
        min_front = min(front_l, front_r)
        
        # Umbral de distancia segura
        safe_distance = 10
        
        if min_front < safe_distance:
            print(f"¡Riesgo de caída detectado! Distancia mínima frontal: {min_front}")
            return True
                
        return False

    except Exception as e:
        print(f"Error en check_fall_risk: {e}")
        return False
    

def select_action(q_table, state, epsilon):
    """
    Selecciona una acción según el valor de epsilon 
    """
    if state == 5:  # Estado no válido
        return random.randint(0, 4)
    
    # Con probabilidad epsilon, exploramos 
    if random.random() < epsilon:
        return random.randint(0, 4)
    
    # Con probabilidad (1-epsilon), explotamos 
    return np.argmax(q_table[state])


def train_robot(sim, initial_q_table=None):
    """
    Entrenamiento por epocas usando Q-learning
    
    Args:
        sim: simulador
        initial_q_table: Tabla Q inicial (opcional)     
    Returns:
        La mejor tabla Q aprendida
    """
    # Inicializar la conexión con Robobo
    print("Conectando con Robobo para entrenamiento...")
    rob = Robobo('localhost')
    rob.connect()
    
    # Inicializar la tabla Q o usar la proporcionada
    q_table = initial_q_table if initial_q_table is not None else initialize_q_table()
    
    # Variables para seguimiento
    episode_rewards = []
    total_interactions = 0
    best_episode_reward = -1000 # Placeholder con valor extremadamente bajo
    best_q_table = None
    
    print("Iniciando entrenamiento...")    

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
        
        while not goal_reached and step < NUM_EPOCHS:
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
            print_q_table(state, action, reward, q_table)
                
            step += 1
            
            # Si se alcanzó el objetivo, mostrar mensaje
            if goal_reached:
                print(f"\nObjetivo alcanzado en el paso {step}")
                break
        
        # Guardar la recompensa del episodio
        episode_rewards.append(episode_reward)
        
        # Guardar la mejor tabla Q 
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_q_table = q_table.copy() 
        
        print(f"\nEpisodio {episode+1} completado.")
        print(f"Recompensa total del episodio: {episode_reward}")
        print(f"Mejor recompensa hasta ahora: {best_episode_reward}")
        print(f"Total de interacciones: {total_interactions}")
    
    # Desconectar el robot
    rob.disconnect()
    return best_q_table

if __name__ == "__main__":
    
    sim = RoboboSim("localhost")
    sim.connect()
    sim.resetSimulation()
    
    try:
        # Cargar una tabla Q existente o crear una nueva
        q_table = load_q_table()
        
        # Entrenar al robot
        print("=== ENTRENAMIENTO DEL ROBOT ===")
        learned_q_table = train_robot(sim, q_table)              

        # Guardar la mejor tabla Q aprendida
        save_q_table(learned_q_table)
        
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        sim.disconnect()
