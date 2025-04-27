import numpy as np

def initialize_q_table():
    """
    Inicializa la tabla Q con valores aleatorios entre 0 y 1
    """
    q_table = np.random.random((6, 5)) * 0.1  # Valores entre 0 y 0.1
    return q_table

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

def load_q_table(filename='q_table.npy'):
    """
    Carga una tabla Q desde un archivo
    
    Args:
        filename: Nombre del archivo desde donde cargar la tabla (por defecto 'q_table.npy')
    Returns:
        La tabla Q cargada o una nueva si no se pudo cargar el archivo
    """
    try:
        # Intentar cargar la tabla desde el archivo
        q_table = np.load(filename)
        print(f"Tabla Q cargada exitosamente desde '{filename}'")
        return q_table

    except FileNotFoundError:
        print(f"No se encontró el archivo '{filename}'. Se inicializará una nueva tabla Q")
        return initialize_q_table()

    except Exception as e:
        print(f"Error al cargar la tabla Q: {e}")
        print("Se inicializará una nueva tabla Q")
        return initialize_q_table()

def print_q_table(state, action, reward, q_table, test=False):
    """
    Imprime el estado actual y la tabla Q de forma formateada
    """
    # Imprimir información de seguimiento
    if not test:
        print(f"Estado actual: {state}, Acción: {action}, Recompensa: {reward}")

    print("\nEstado de la tabla Q:")
    state_names = ["Muy izquierda", "Poco izquierda", "Centro", "Poco derecha", "Muy derecha", "No válido"]
    print("-" * 70)
    print(f"| {'Estado':^10} | {'Nombre':^15} | {'Valores Q':^40} |")
    print("-" * 70)
    
    for s in range(6):
        actions_values = q_table[s]
        formatted_values = [f"{value:.3f}" for value in actions_values]
        formatted_str = "[" + " ".join(formatted_values) + "]"
        
        print(f"| {s:^10} | {state_names[s]:^15} | {formatted_str:^40} |")
    print("-" * 70)

