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
		
		labeled_data = 0.01 # Vamos a usar el etiquetado de sólo el 1% de los datos
		np.random.seed(42)
		
		(x_train, y_train), (x_test, y_test), = keras.datasets.cifar100.load_data()
		
		indexes = np.arange(len(x_train))
		np.random.shuffle(indexes)
		ntrain_data = int(labeled_data*len(x_train))
		unlabeled_train = x_train[indexes[ntrain_data:]] /255
		x_train = x_train[indexes[:ntrain_data]] /255
		x_test = x_test /255
		y_train = y_train[indexes[:ntrain_data]]

		one_hot_train = np.zeros((y_train.size, len(np.unique(y_train))), dtype=int)
		for vector, y in zip(one_hot_train, y_train):
			vector[y] = 1
		
		one_hot_test = np.zeros((y_test.size, len(np.unique(y_test))), dtype=int)
		one_hot_test[np.arange(y_test.size), y_test ] = 1
		
		return unlabeled_train, x_train, y_train, x_test, y_test, one_hot_train, one_hot_test


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