import keras

class DatasetProcess:
	def load():
		return keras.datasets.cifar100.load_data()
	def separate_data(train, test):
		(x_train, y_train) = train
		(x_test, y_test) = test
  
		x_train_no_labeled = x_train[:40000]  # Sin etiquetas
		x_train_labeled = x_train[40000:]     # Con etiquetas
		y_train_labeled = y_train[40000:]     # Etiquetas correspondientes
  
		return (x_train_no_labeled, x_train_labeled, y_train_labeled),(x_test, y_test)