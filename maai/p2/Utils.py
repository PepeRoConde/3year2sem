import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import visualkeras

class DatasetProcess:
    """
    Clase que proporciona métodos para cargar y procesar conjuntos de datos.
    
    Esta clase ofrece funcionalidades para cargar el conjunto de datos CIFAR-100
    y prepararlo para diferentes enfoques de aprendizaje (supervisado, semi-supervisado, etc.)
    """
    @staticmethod
    def load():
        """
        Carga el conjunto de datos CIFAR-100.
        
        Returns:
            tuple: Tupla que contiene (datos_entrenamiento, datos_prueba)
        """
        return keras.datasets.cifar100.load_data()
        
    @staticmethod
    def hold_out(train, test, validation_size=2000):
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.
        
        Divide además los datos de entrenamiento en etiquetados y no etiquetados
        para permitir enfoques semi-supervisados.
        
        Args:
            train: Datos de entrenamiento (x_train, y_train)
            test: Datos de prueba (x_test, y_test)
            validation_size (int): Tamaño del conjunto de validación
            
        Returns:
            tuple: Contiene (train_sets, val_sets, test_sets)
                  - train_sets: (x_train_no_labeled, x_train_labeled, y_train_labeled)
                  - val_sets: (x_val, y_val)
                  - test_sets: (x_test, y_test)
        """
        (x_train, y_train) = train
        (x_test, y_test) = test
        
        # Separación de datos no etiquetados (80% del conjunto de entrenamiento)
        x_train_no_labeled = x_train[:40000]  # Sin etiquetas
        
        # Los 10000 restantes del conjunto de entrenamiento
        x_train_labeled_all = x_train[40000:]
        y_train_labeled_all = y_train[40000:]
        
        # Separación para validación
        x_val = x_train_labeled_all[:validation_size]
        y_val = y_train_labeled_all[:validation_size]
        
        # Los datos etiquetados restantes para entrenamiento
        x_train_labeled = x_train_labeled_all[validation_size:]
        y_train_labeled = y_train_labeled_all[validation_size:]
        
        train_sets = (x_train_no_labeled, x_train_labeled, y_train_labeled)
        val_sets = (x_val, y_val)
        test_sets = (x_test, y_test)
        
        return train_sets, val_sets, test_sets
    
    @staticmethod
    def load_dataset():
        """
        Método ampliado para cargar y preparar el conjunto de datos CIFAR-100.
        
        Divide los datos en conjuntos para entrenamiento semi-supervisado y realiza
        la normalización y codificación one-hot de las etiquetas.
        
        Returns:
            tuple: Múltiples conjuntos de datos procesados y preparados
        """
        labeled_data = 0.2  
        np.random.seed(42)
        
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        
        # Shuffle the indices of the training data
        indexes = np.arange(len(x_train))
        np.random.shuffle(indexes)
        
        # Define the number of labeled data to use
        ntrain_data = int(labeled_data * len(x_train))
        
        # Split the data into unlabeled, train, and validation sets
        unlabeled_train = x_train[indexes[ntrain_data:]] / 255
        x_train = x_train[indexes[:ntrain_data]] / 255
        x_test = x_test / 255
        y_train = y_train[indexes[:ntrain_data]]
        
        # Split the labeled training data further into training and validation
        validation_split = 0.1
        nvalidation_data = int(validation_split * len(x_train))
        
        x_val = x_train[-nvalidation_data:]
        y_val = y_train[-nvalidation_data:]
        x_train = x_train[:-nvalidation_data]
        y_train = y_train[:-nvalidation_data]
        
        # One-hot encoding for labels
        one_hot_train = np.zeros((y_train.size, len(np.unique(y_train))), dtype=int)
        for vector, y in zip(one_hot_train, y_train):
            vector[y] = 1
        
        one_hot_val = np.zeros((y_val.size, len(np.unique(y_val))), dtype=int)
        for vector, y in zip(one_hot_val, y_val):
            vector[y] = 1
        
        one_hot_test = np.zeros((y_test.size, len(np.unique(y_test))), dtype=int)
        one_hot_test[np.arange(y_test.size), y_test] = 1
        
        return unlabeled_train, x_train, y_train, x_val, y_val, x_test, y_test, one_hot_train, one_hot_val, one_hot_test


def reconstruction_plot(autoencoder, x_test):
    """
    Visualiza la reconstrucción de una imagen por un autoencoder.
    
    Esta función selecciona aleatoriamente una imagen del conjunto de prueba,
    la pasa por el autoencoder y muestra la imagen original junto con la reconstruida.
    
    Args:
        autoencoder: Modelo autoencoder
        x_test: Conjunto de datos de prueba
    """
    index = np.random.randint(len(x_test))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(x_test[index].reshape(32, 32, 3))
    axes[0].set_title("Imagen Original")
    axes[0].axis('off') 
    
    reconstructed_image = autoencoder(x_test[index].reshape(1, 32, 32, 3))

    if len(reconstructed_image) == 2: # esto es para usarlo con el autoencoder de una o dos cabezas
        reconstructed_image = reconstructed_image['decoder'].reshape(32, 32, 3)
    else:
        reconstructed_image = reconstructed_image.reshape(32, 32, 3)
    
    axes[1].imshow(reconstructed_image)
    axes[1].set_title("Imagen Reconstruida")
    axes[1].axis('off')
    
    plt.show()

def plot_model(model, name):
    """
    Visualiza la arquitectura del modelo y la guarda como imagen.
    
    Args:
        model: Puede ser un modelo de Keras directo o un objeto ConvModel
        name: Nombre del archivo de salida
    """
    # Comprobar si el modelo es una instancia de modelo Keras
    if hasattr(model, 'model'):
        keras_model = model.model
    else:
        # Si ya es un modelo Keras, usarlo directamente
        keras_model = model
    
    # Visualizar el modelo usando visualkeras
    visualkeras.layered_view(keras_model, to_file=name, legend=True, draw_volume=True)    
    print(f"Modelo guardado como '{name}'")
 
def anomaly_report(model, unlabeled_train):
    """
    Genera un informe sobre detección de anomalías en datos no etiquetados.
    
    Esta función utiliza un modelo de detección de anomalías para identificar
    qué ejemplos en el conjunto no etiquetado son típicos y cuáles son atípicos.
    
    Args:
        model: Modelo detector de anomalías
        unlabeled_train: Conjunto de datos no etiquetados
        
    Returns:
        tuple: (filtered_unlabeled_train, is_typical)
               - filtered_unlabeled_train: Datos no etiquetados filtrados (solo típicos)
               - is_typical: Máscara booleana que indica qué ejemplos son típicos
    """
    # Primero predecimos qué datos son típicos en el conjunto no etiquetado
    unlabeled_predictions = model.predict(unlabeled_train)
    r_value = model.model.r.numpy()
    is_typical = unlabeled_predictions > r_value
    
    # Filtramos para obtener solo los datos típicos
    filtered_unlabeled_train = np.array([unlabeled_train[i] for i in range(len(unlabeled_train)) if is_typical[i]])
    percetage = float(sum(is_typical) * 100 / len(is_typical))
    print(f"Porcentaje de datos no etiquetados etiquetados como típicos: {percetage:.2f}%")
    print(f"Porcentaje de datos no etiquetados etiquetados como atípicos: {100 - percetage:.2f}%")
    print(f"Datos originales no etiquetados: {unlabeled_train.shape}")
    print(f"Datos filtrados no etiquetados (solo típicos): {filtered_unlabeled_train.shape}")
    print(f"Se eliminaron {unlabeled_train.shape[0] - filtered_unlabeled_train.shape[0]} muestras atípicas")

    return filtered_unlabeled_train, is_typical


def plot_atipicos(is_typical, unlabeled_train, title="Ejemplo Atípico", random_state=None):
    """
    Visualiza un ejemplo atípico del conjunto de datos.
    
    Args:
        is_typical: Máscara booleana que indica qué ejemplos son típicos
        unlabeled_train: Conjunto de datos no etiquetados
        title (str): Título para la visualización
        random_state: Semilla para reproducibilidad
        
    Returns:
        int: Índice del ejemplo atípico visualizado
    """
    # Obtener índices de ejemplos atípicos
    is_atipica = ~is_typical
    indices_atipicos = np.where(is_atipica)[0]
    
    # Verificar si hay ejemplos atípicos
    if len(indices_atipicos) == 0:
        print("No se encontraron ejemplos atípicos")
        return
    
    # Seleccionar un ejemplo aleatorio con reproducibilidad opcional
    if random_state is not None:
        np.random.seed(random_state)
    indice_atipico = np.random.choice(indices_atipicos)
    
    # Visualizar
    plt.figure(figsize=(8, 8))
    plt.imshow(unlabeled_train[indice_atipico])
    plt.title(f"{title} (índice: {indice_atipico})")
    plt.axis('off')
    plt.show()
    
    return indice_atipico