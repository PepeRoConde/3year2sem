import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

class DatasetProcess:
    def load():
        return keras.datasets.cifar100.load_data()
        
    def hold_out(train, test, validation_size=2000):
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
    
    def alt():
        
        labeled_data = 0.33  
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
        validation_split = 0.5  # Use 50% of labeled data for validation
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
	index = np.random.randint(len(x_test))
	
	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	
	axes[0].imshow(x_test[index].reshape(32, 32, 3))
	axes[0].set_title("Original Image")
	axes[0].axis('off') 
	
	reconstructed_image = autoencoder(x_test[index].reshape(1, 32, 32, 3))

	if  len(reconstructed_image) == 2: # esto es para usarlo con el autoencoder de una o dos cabezas
		reconstructed_image = reconstructed_image['decoder'].reshape(32, 32, 3)
	else:
		reconstructed_image = reconstructed_image.reshape(32, 32, 3)
	
	axes[1].imshow(reconstructed_image)
	axes[1].set_title("Reconstructed Image")
	axes[1].axis('off')
	
	plt.show()


def anomaly_report(model,unlabeled_train):
    # Primero predecimos qué datos son típicos en el conjunto no etiquetado
    unlabeled_predictions = model.predict(unlabeled_train)
    r_value = model.model.r.numpy()
    is_typical = unlabeled_predictions > r_value
    
    # Filtramos para obtener solo los datos típicos
    filtered_unlabeled_train = np.array([unlabeled_train[i]  for i in range(len(unlabeled_train)) if is_typical[i]])
    percetage = float(sum(is_typical) * 100 / len(is_typical))
    print(f"Porcentaje de datos no etiquetados etiquetados como típicos: {percetage:.2f}%")
    print(f"Porcentaje de datos no etiquetados etiquetados como atípicos: {100 - percetage:.2f}%")
    print(f"Datos originales no etiquetados: {unlabeled_train.shape}")
    print(f"Datos filtrados no etiquetados (solo típicos): {filtered_unlabeled_train.shape}")
    print(f"Se eliminaron {unlabeled_train.shape[0] - filtered_unlabeled_train.shape[0]} muestras atípicas")

    return filtered_unlabeled_train



    
def plot_atipicos():
    # Seleccionar un ejemplo de imagen típica
    # Tomamos una imagen aleatoria del conjunto filtrado (típicas)
    idx_tipica = np.random.randint(0, filtered_unlabeled_train.shape[0])
    img_tipica = filtered_unlabeled_train[idx_tipica]
    
    # Seleccionar un ejemplo de imagen atípica
    # Creamos una máscara para las imágenes atípicas
    is_atipica = ~is_typical
    atipicas = unlabeled_train[is_atipica]
    
    # Verificamos que haya imágenes atípicas
    if atipicas.shape[0] > 0:
        idx_atipica = np.random.randint(0, atipicas.shape[0])
        img_atipica = atipicas[idx_atipica]
    else:
        # Si no hay imágenes atípicas, tomamos la que tenga la puntuación más baja
        idx_atipica = np.argmin(unlabeled_predictions)
        img_atipica = unlabeled_train[idx_atipica]
    
    # Calcular las puntuaciones de anomalía para ambas imágenes
    score_tipica = model.predict(img_tipica.reshape(1, 32, 32, 3))[0]
    score_atipica = model.predict(img_atipica.reshape(1, 32, 32, 3))[0]
    
    # Visualizar ambas imágenes lado a lado
    plt.figure(figsize=(12, 5))
    
    # Imagen típica
    plt.subplot(1, 2, 1)
    plt.imshow(img_tipica)
    plt.title(f'Imagen Típica\nPuntuación: {score_tipica:.4f}\nr_value: {r_value:.4f}')
    plt.axis('off')
    
    # Imagen atípica
    plt.subplot(1, 2, 2)
    plt.imshow(img_atipica)
    plt.title(f'Imagen Atípica\nPuntuación: {score_atipica:.4f}\nr_value: {r_value:.4f}')
    plt.axis('off')
    
    plt.suptitle('Comparación de Imágenes Típicas vs Atípicas')
    plt.tight_layout()
    plt.show()

# Adicionalmente, podemos visualizar más ejemplos de imágenes atípicas
def mostrar_mas_ejemplos_atipicos(num_ejemplos=5):
    if atipicas.shape[0] > 0:
        plt.figure(figsize=(15, 3))
        indices = np.random.choice(atipicas.shape[0], min(num_ejemplos, atipicas.shape[0]), replace=False)
        
        for i, idx in enumerate(indices):
            plt.subplot(1, num_ejemplos, i + 1)
            plt.imshow(atipicas[idx])
            score = model.predict(atipicas[idx].reshape(1, 32, 32, 3))[0]
            plt.title(f'Score: {score:.4f}')
            plt.axis('off')
            
        plt.suptitle(f'Ejemplos adicionales de imágenes atípicas')
        plt.tight_layout()
        plt.show()
    else:
        print("No hay suficientes ejemplos de imágenes atípicas para mostrar.")

# Descomentar para mostrar más ejemplos
# mostrar_mas_ejemplos_atipicos()

# También podemos ver la distribución de puntuaciones
def visualizar_distribucion_puntuaciones():
    plt.figure(figsize=(10, 6))
    
    # Obtener todas las predicciones
    todas_predicciones = model.predict(unlabeled_train)
    
    # Visualizar histograma
    plt.hist(todas_predicciones, bins=50, alpha=0.7)
    plt.axvline(x=r_value, color='r', linestyle='--', label=f'r_value = {r_value:.4f}')
    plt.xlabel('Puntuación de Anomalía')
    plt.ylabel('Número de Muestras')
    plt.title('Distribución de Puntuaciones de Anomalía')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Descomentar para visualizar la distribución
# visualizar_distribucion_puntuaciones()