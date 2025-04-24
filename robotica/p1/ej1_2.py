import time
import random
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR  # Importación correcta de la enumeración IR
from robobosim.RoboboSim import RoboboSim

# Constantes
ALPHA = 0.4  # Tasa de aprendizaje
GAMMA = 0.5  # Factor de descuento
EPSILON = 0.2  # Probabilidad de exploración (vs explotación)
MAX_LUX = 350  # Nivel de luz para considerar que se ha llegado al objetivo
NUM_EPISODES = 100  # Número de episodios de entrenamiento
SPEED = 10  # Velocidad de movimiento
# Ya no usamos MIN_IR_SAFE pues cambió la lógica de detección de caídas

# Definición de estados
# Estado 0: más luz a la derecha
# Estado 1: más luz a la izquierda
# Estado 2: más luz en el centro/frente
# Estado -1: lectura no válida

# Definición de acciones
# Acción 0: Girar a la derecha
# Acción 1: Girar a la izquierda
# Acción 2: Avanzar recto

def initialize_q_table():
    """
    Inicializa la tabla Q con valores aleatorios pequeños
    """
    # Definimos 4 estados (-1, 0, 1, 2) y 3 acciones (0, 1, 2)
    q_table = np.zeros((4, 3))
    # Inicializamos con valores pequeños aleatorios
    q_table = np.random.random((4, 3)) * 0.1
    return q_table

def get_state(rob):
    """
    Determina el estado actual basado en las lecturas de luz
    """
    try:
        # Leer los sensores de luz
        brightness = rob.readBrightnessSensor()
        
        # Si no hay suficiente luz o lecturas erróneas, consideramos estado no válido
        if brightness < 5:
            return -1
            
        rob.movePanTo(-90, 50) 
        time.sleep(0.5)  # Dar tiempo a que complete el movimiento
        left_brightness = rob.readBrightnessSensor()
        
        rob.movePanTo(0, 50)  # Volver al centro
        time.sleep(0.5)  # Dar tiempo a que complete el movimiento
        center_brightness = rob.readBrightnessSensor()
        
        rob.movePanTo(90, 50)
        time.sleep(0.5)  # Dar tiempo a que complete el movimiento
        right_brightness = rob.readBrightnessSensor()

        # Volver a la posición central
        rob.movePanTo(0, 50)
        time.sleep(0.5)  # Dar tiempo a que complete el movimiento
        
        print(f"Lecturas de brillo - Izquierda: {left_brightness}, Centro: {center_brightness}, Derecha: {right_brightness}")
        
        # Determinar el estado basado en la comparación de lecturas
        if right_brightness > left_brightness and right_brightness > center_brightness:
            return 0  # Más luz a la derecha
        elif left_brightness > right_brightness and left_brightness > center_brightness:
            return 1  # Más luz a la izquierda
        else:
            return 2  # Más luz al frente
            
    except Exception as e:
        print(f"Error al leer sensores: {e}")
        return -1  # Estado no válido en caso de error

def check_fall_risk(rob):
    """
    Comprueba si hay riesgo de caída usando los sensores IR frontales
    Devuelve True si hay riesgo, False si es seguro
    """
    try:
        # Leer los sensores IR frontales directamente
        front_l = rob.readIRSensor(IR.FrontL)
        front_r = rob.readIRSensor(IR.FrontR)
        
        # Imprimir los valores para diagnóstico
        print(f"Sensores frontales - Izquierdo: {front_l}, Derecho: {front_r}")
        
        # Valor mínimo de los dos sensores frontales
        min_front = min(front_l, front_r)
        
        # Umbral de distancia segura (ajustar según entorno)
        safe_distance = 5
        
        # Si alguno de los sensores frontales detecta una distancia muy corta (o ninguna superficie),
        # consideramos que hay riesgo de caída
        if min_front < safe_distance:
            print(f"¡Riesgo de caída detectado! Distancia mínima frontal: {min_front}")
            return True
                
        return False
        
    except Exception as e:
        print(f"Error al comprobar riesgo de caída: {e}")
        return False

def take_action(rob, action):
    """
    Ejecuta la acción seleccionada y devuelve la recompensa
    """
    initial_brightness = rob.readBrightnessSensor()
    
    # Comprobar riesgo de caída antes de moverse
    if check_fall_risk(rob):
        print("¡Precaución! Detectado posible borde. Retrocediendo...")
        # Retroceder un poco para evitar la caída
        rob.moveWheelsByTime(-SPEED, -SPEED, 0.5)
        time.sleep(0.3)
        
        # Girar para cambiar de dirección
        rob.moveWheelsByTime(SPEED, -SPEED, 0.4)
        time.sleep(0.3)
        
        return -1, False  # Recompensa negativa menor por evitar caída
    
    # Ejecutar la acción
    if action == 0:  # Girar a la derecha
        print("Girando a la derecha")
        rob.moveWheelsByTime(SPEED, -SPEED, 0.5)
    elif action == 1:  # Girar a la izquierda
        print("Girando a la izquierda")
        rob.moveWheelsByTime(-SPEED, SPEED, 0.5)
    elif action == 2:  # Avanzar recto
        print("Avanzando recto")
        
        # Implementar una versión simplificada de avoid_falling dentro del avance
        rob.moveWheels(SPEED, SPEED)  # Iniciar movimiento continuo
        
        # Monitorear sensores durante un tiempo máximo
        max_move_time = 1.0  # Tiempo máximo de movimiento en segundos
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
            
            # Breve pausa para no saturar los sensores
            time.sleep(0.1)
        
        # Asegurarse de que los motores se detengan
        rob.stopMotors()
    
    time.sleep(0.3)  # Pequeña pausa para estabilizar
    
    # Leer el nuevo nivel de brillo
    new_brightness = rob.readBrightnessSensor()
    print(f"Brillo inicial: {initial_brightness}, Brillo nuevo: {new_brightness}")
    
    # Calcular la recompensa basada en el cambio de brillo
    brightness_change = new_brightness - initial_brightness
    
    if brightness_change > 5:  # Mejora significativa
        reward = 2
    elif brightness_change > 0:  # Mejora leve
        reward = 1
    else:  # Empeoramiento
        reward = -1
    
    # Comprobar si se ha alcanzado el objetivo
    if new_brightness >= MAX_LUX:
        reward = 10  # Recompensa especial por alcanzar el objetivo
        return reward, True
    
    return reward, False
    
    # Ejecutar la acción
    if action == 0:  # Girar a la derecha
        print("Girando a la derecha")
        rob.moveWheelsByTime(SPEED, -SPEED, 0.5)
    elif action == 1:  # Girar a la izquierda
        print("Girando a la izquierda")
        rob.moveWheelsByTime(-SPEED, SPEED, 0.5)
    elif action == 2:  # Avanzar recto
        print("Avanzando recto")
        # Movemos por más tiempo para asegurar que avanza lo suficiente
        rob.moveWheelsByTime(SPEED, SPEED, 1.0)
    
    time.sleep(0.5)  # Esperar a que se complete la acción
    
    # Leer el nuevo nivel de brillo
    new_brightness = rob.readBrightnessSensor()
    print(f"Brillo inicial: {initial_brightness}, Brillo nuevo: {new_brightness}")
    
    # Calcular la recompensa
    if new_brightness > initial_brightness:
        reward = 1  # Recompensa positiva si aumenta el brillo
    else:
        reward = -1  # Recompensa negativa si disminuye el brillo
    
    # Comprobar si se ha alcanzado el objetivo
    if new_brightness >= MAX_LUX:
        reward = 10  # Recompensa especial por alcanzar el objetivo
        return reward, True
    
    return reward, False

def select_action(q_table, state, epsilon):
    """
    Selecciona una acción usando la política epsilon-greedy
    """
    if state == -1:  # Si el estado no es válido, realizamos una acción aleatoria
        return random.randint(0, 2)
    
    # Con probabilidad epsilon, exploramos (acción aleatoria)
    if random.random() < epsilon:
        return random.randint(0, 2)
    
    # Con probabilidad (1-epsilon), explotamos (mejor acción conocida)
    # Convertimos el estado negativo a índice positivo para la tabla
    state_idx = state if state >= 0 else 3  # El estado -1 lo mapeamos al índice 3
    return np.argmax(q_table[state_idx])

def update_q_table(q_table, state, action, reward, new_state):
    """
    Actualiza la tabla Q usando la ecuación de Bellman
    """
    # Convertimos estados negativos a índices positivos para la tabla
    state_idx = state if state >= 0 else 3
    new_state_idx = new_state if new_state >= 0 else 3
    
    # Calcular el valor Q máximo para el nuevo estado
    max_future_q = np.max(q_table[new_state_idx])
    
    # Valor Q actual
    current_q = q_table[state_idx, action]
    
    # Aplicar la ecuación de actualización de Q-learning
    new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
    
    # Actualizar la tabla Q
    q_table[state_idx, action] = new_q
    
    return q_table

def train_robot(sim):
    """
    Entrenamiento principal del robot usando Q-learning
    """
    # Inicializar la conexión con Robobo
    print("Conectando con Robobo...")
    rob = Robobo('localhost')  # Cambiar por la IP correcta
    rob.connect()
    
    # Inicializar la tabla Q
    q_table = initialize_q_table()
    
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
            for s in range(4):
                state_name = s if s < 3 else "-1"
                actions_values = q_table[s]
                print(f"  Estado {state_name}: {actions_values}")
            
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
        
        print(f"\nEpisodio {episode+1} completado.")
        print(f"Recompensa total del episodio: {episode_reward}")
        print(f"Mejor recompensa hasta ahora: {best_episode_reward}")
        print(f"Total de interacciones: {total_interactions}")
    
    # Desconectar el robot
    rob.disconnect()
    
    # Retornar la mejor tabla Q aprendida
    return best_q_table  # Devolvemos la tabla Q del mejor episodio

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
    max_steps = 50  # Aumentamos el número máximo de pasos para la prueba
    total_reward = 0
    
    # Imprimir la tabla Q final
    print("\nTabla Q final utilizada para la prueba:")
    for s in range(4):
        state_name = s if s < 3 else "-1"
        actions_values = q_table[s]
        best_action = np.argmax(actions_values)
        action_names = ["girar derecha", "girar izquierda", "avanzar"]
        print(f"  Estado {state_name}: {actions_values} -> Mejor acción: {action_names[best_action]}")
    
    print("\nIniciando recorrido hacia la luz...")
    
    while not goal_reached and steps < max_steps:
        # Determinar el estado actual
        state = get_state(rob)
        print(f"\n--- Paso {steps+1} ---")
        print(f"Estado actual: {state}")
        
        # Seleccionar la mejor acción (sin exploración)
        state_idx = state if state >= 0 else 3
        action = np.argmax(q_table[state_idx])
        
        # Ejecutar la acción
        action_names = ["girar derecha", "girar izquierda", "avanzar"]
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

# Programa principal
if __name__ == "__main__":
    
    sim = RoboboSim("localhost")
    sim.connect()
    sim.resetSimulation()

    try:
        # Entrenar al robot
        print("=== ENTRENAMIENTO DEL ROBOT ===")
        learned_q_table = train_robot(sim)
        
        # Probar el comportamiento aprendido
        print("\n=== PRUEBA FINAL DEL ROBOT ===")
        test_robot(learned_q_table, sim)
        
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\nPrograma finalizado.")
