from tensorflow.keras import layers, optimizers, models, Model, regularizers, losses
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

class ContrastiveLoss():
	"""
	Esta función de pérdida fomenta embeddings similares para diferentes vistas aumentadas
	de la misma entrada y separa los embeddings de diferentes entradas.
	
	Atributos:
		temperature (float): Factor de escala que controla la nitidez de la distribución softmax
	"""
	def __init__(self, temperature=0.5):
		"""
		Args:
			temperature (float): Parámetro de temperatura para escalar puntuaciones de similitud
		"""
		self.temperature = temperature
		
	def __call__(self, M):
		"""
		Calcula la pérdida contrastiva a partir de la matriz de similitud.
		
		Args:
			M: Matriz de similitud entre representaciones codificadas
			
		Returns:
			float: Valor de pérdida contrastiva calculado
		"""
		logits = M / self.temperature # temperatura
		
		batch_size = tf.shape(M)[0]
		I = tf.eye(batch_size)  # matriz identidad

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(I, logits))
		
		return loss

class ClusteringLoss():
	"""
	Implementación de pérdida de clustering para aprendizaje auto-supervisado.
	
	Esta función de pérdida fomenta que el modelo produzca asignaciones de cluster
	consistentes para diferentes vistas aumentadas de la misma entrada, mientras
	también fomenta que la distribución de cluster sea más definida (menos ambigua).
	"""
	def __init__(self):
		"""
		Inicializa ClusteringLoss.
		"""
		pass
		
	def __call__(self, cX_1comp, cX_2comp):
		"""
		Calcula la pérdida de clustering a partir de dos conjuntos de asignaciones de cluster.
		
		Args:
			cX_1comp: Asignaciones de cluster de la primera vista aumentada
			cX_2comp: Asignaciones de cluster de la segunda vista aumentada
			
		Returns:
			float: Valor de pérdida de clustering calculado
		"""
		# Fomentar distribuciones definidas (minimizar entropía)
		entropy_1 = -tf.reduce_mean(tf.reduce_sum(cX_1comp * tf.math.log(cX_1comp + 1e-8), axis=1))
		entropy_2 = -tf.reduce_mean(tf.reduce_sum(cX_2comp * tf.math.log(cX_2comp + 1e-8), axis=1))
		
		# Asegurar consistencia entre vistas
		consistency = tf.reduce_mean(tf.reduce_sum(tf.square(cX_1comp - cX_2comp), axis=1))
		
		return entropy_1 + entropy_2 + consistency

class ContrastiveModel():
	"""
	Modelo de aprendizaje auto-supervisado usando pérdidas contrastivas y de clustering.
	
	Este modelo aprende representaciones a partir de datos no etiquetados comparando
	diferentes vistas aumentadas de las mismas imágenes. Utiliza un enfoque de aprendizaje
	contrastivo combinado con un término de clustering para mejorar la calidad de las
	representaciones aprendidas sin supervisión.
	
	Atributos:
		input_shape: Forma de las imágenes de entrada
		output_dim (int): Número de dimensiones de salida (clusters)
		lambda_param (float): Peso para la pérdida de clustering
		contrastive_loss: Función de pérdida para aprendizaje contrastivo
		clustering_loss: Función de pérdida para clustering
		data_augmentation_1: Primera pipeline de aumentación de datos
		data_augmentation_2: Segunda pipeline de aumentación de datos
		encoder: Red neuronal para extracción de características
		cluster: Red neuronal para asignación de clusters
		loss_history (dict): Historial de valores de pérdida durante el entrenamiento
	"""
	def __init__(self, input_shape, lambda_param=0.5, temperature=0.5, learning_rate=0.0005, l2_lambda=0.01, dropout_prob=0.01):
		"""
		Args:
			input_shape: Forma de las imágenes de entrada
			lambda_param (float): Factor de peso para la pérdida de clustering
			temperature (float): Parámetro de temperatura para la pérdida contrastiva
			learning_rate (float): Tasa de aprendizaje inicial
			l2_lambda (float): Factor de regularización L2
			dropout_prob (float): Probabilidad de dropout
		"""
		self.input_shape = input_shape
		self.output_dim = 100
		self.lambda_param = lambda_param
		self.contrastive_loss = ContrastiveLoss(temperature=temperature)
		self.clustering_loss = ClusteringLoss()

		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=learning_rate,
			decay_steps=30,
			decay_rate=0.8)
		
		self.optimicer = optimizers.AdamW(
			learning_rate=lr_schedule,
			clipnorm=5,
		)
		
		# Definición de dos pipelines de aumentación de datos diferentes
		self.data_augmentation_1 = models.Sequential([
				layers.RandomFlip("horizontal"), 
				layers.RandomGaussianBlur(factor=(0,.5)),
				layers.RandomColorJitter(value_range=(0,1),hue_factor=(0.1, 0.1)),
				layers.RandomRotation(0.05),
				layers.RandomTranslation(0.15, 0.15),
				layers.RandomZoom(.15),
			])
	
		self.data_augmentation_2 = tf.keras.models.Sequential([
				layers.RandomFlip("horizontal"),  
				layers.RandomTranslation(0.15, 0.15),
				layers.RandomGaussianBlur(factor=.5),
				layers.RandomRotation(.15),
				layers.Resizing(38, 38), 
				layers.RandomCrop(32, 32), 
			])
			
		# Definir modelo convolucional
		input_layer = layers.Input(shape=self.input_shape)

		# Encoder - Primer bloque convolucional
		x = layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(input_layer)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Dropout(dropout_prob / 2)(x)
	
		# Encoder - Segundo bloque convolucional
		x = layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Dropout(dropout_prob / 2)(x)
	
		# Encoder - Tercer bloque convolucional
		x = layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		code = layers.BatchNormalization()(x)
		
		# Capa de clustering
		flatten_layer = layers.Flatten()(code)
		
		clustering = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(flatten_layer)
		clustering = layers.BatchNormalization()(clustering)
		clustering = layers.Dropout(dropout_prob)(clustering)
		
		cluster_layer = layers.Dense(self.output_dim, activation="softmax", name='classifier')(clustering)

		# Modelo final
		self.encoder = Model(input_layer, outputs=flatten_layer)
		self.cluster = Model(flatten_layer, outputs=cluster_layer)
		
		# Historial de pérdidas para graficar
		self.loss_history = {
			'total_loss': [],
			'contrastive_loss': [],
			'clustering_loss': []
		}
	
	def train_step(self, data, temperature):
		"""
		Paso de entrenamiento para el modelo contrastivo.
		
		Realiza un paso de entrenamiento utilizando pérdidas contrastivas y de clustering
		sobre datos aumentados con dos transformaciones diferentes.
		
		Args:
			data: Datos de entrada para el paso de entrenamiento
			temperature: Parámetro de temperatura para la pérdida contrastiva
			
		Returns:
			dict: Diccionario con valores de pérdida para este paso
		"""
		if isinstance(data, tuple):
			X = data[0]
		else:
			X = data
			
		batch_size = tf.shape(X)[0]
		
		# Aplicar las dos transformaciones de data augmentation
		augX_1 = self.data_augmentation_1(X)
		augX_2 = self.data_augmentation_2(X)
		
		with tf.GradientTape() as tape:
			# Normalizar las representaciones del encoder
			augX_1comp = tf.nn.l2_normalize(self.encoder(augX_1), axis=1)
			augX_2comp = tf.nn.l2_normalize(self.encoder(augX_2), axis=1)

			# Obtener asignaciones de clusters
			cX_1comp = self.cluster(augX_1comp)
			cX_2comp = self.cluster(augX_2comp)
			
			# Calcular matriz de similitud
			M = tf.matmul(augX_1comp, augX_2comp, transpose_b=True)
			
			# Calcular pérdidas
			loss_M = self.contrastive_loss(M) 
			loss_C = self.clustering_loss(cX_1comp, cX_2comp)
			total_loss = loss_M + self.lambda_param * loss_C
			
		# Calcular gradientes y actualizar pesos
		gradients = tape.gradient(total_loss, self.encoder.trainable_variables + self.cluster.trainable_variables)

		grad_norm = tf.linalg.global_norm(gradients)
		
		self.optimicer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.cluster.trainable_variables))
		
		return {"loss": total_loss, "contrastive_loss": loss_M, "clustering_loss": loss_C}
	
	def mini_batches(self, X, batch_size):
		"""
		Args:
			X: Datos de entrada
			batch_size: Tamaño de los mini-batches
			
		Yields:
			Batches de datos para entrenamiento
		"""
		for start in range(0, X.shape[0], batch_size):
			# Yield each mini-batch
			end = min(start + batch_size, X.shape[0])
			yield X[start:end]
			
	def train(self, dataset, epochs=10, batch_size=128, temperature=0.5):
		"""
		Args:
			dataset: Conjunto de datos para entrenamiento
			epochs (int): Número de épocas de entrenamiento
			batch_size (int): Tamaño de los lotes para entrenamiento
			temperature (float): Temperatura para la pérdida contrastiva
		"""
		# Reiniciar el historial de pérdida si comenzamos un nuevo entrenamiento
		self.loss_history = {
			'total_loss': [],
			'contrastive_loss': [],
			'clustering_loss': []
		}
		
		for epoch in range(epochs):
			epoch_total_loss = 0
			epoch_contrastive_loss = 0
			epoch_clustering_loss = 0
			batch_count = 0
			
			for data in self.mini_batches(dataset, batch_size=batch_size):
				print('.',end='')
				loss_dict = self.train_step(data, temperature=temperature)
				epoch_total_loss += loss_dict["loss"]
				epoch_contrastive_loss += loss_dict["contrastive_loss"]
				epoch_clustering_loss += loss_dict["clustering_loss"]
				batch_count += 1
			
			# Calcular promedios para la época
			avg_total_loss = epoch_total_loss / batch_count
			avg_contrastive_loss = epoch_contrastive_loss / batch_count
			avg_clustering_loss = epoch_clustering_loss / batch_count
			
			# Guardar en el historial
			self.loss_history['total_loss'].append(avg_total_loss.numpy())
			self.loss_history['contrastive_loss'].append(avg_contrastive_loss.numpy())
			self.loss_history['clustering_loss'].append(avg_clustering_loss.numpy())
			
			print(f"Epoch {epoch+1}/{epochs}, Total Loss: {avg_total_loss:.4f}, "
				  f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
				  f"Clustering Loss: {avg_clustering_loss:.4f}")
			
	def get_encoded_data(self,X):
		"""
		Obtiene las representaciones codificadas para los datos de entrada.
		
		Args:
			X: Datos de entrada
			
		Returns:
			Representaciones codificadas generadas por el encoder
		"""
		return self.encoder(X)
	
	def plot_training_history(self, figsize=(12, 6)):
		"""
		Visualiza el historial de pérdidas durante el entrenamiento.
		
		Args:
			figsize (tuple): Tamaño de la figura para visualización
		"""
		import matplotlib.pyplot as plt
		
		epochs = range(1, len(self.loss_history['total_loss']) + 1)
		
		plt.figure(figsize=figsize)
		
		# Gráfico de pérdida total
		plt.subplot(1, 2, 1)
		plt.plot(epochs, self.loss_history['total_loss'], 'b-', label='Pérdida Total')
		plt.title('Pérdida Total')
		plt.xlabel('Épocas')
		plt.ylabel('Pérdida')
		plt.grid(True)
		plt.legend()
		
		# Gráfico comparativo de pérdidas
		plt.subplot(1, 2, 2)
		plt.plot(epochs, self.loss_history['contrastive_loss'], 'r-', label='Pérdida Contrastiva')
		plt.plot(epochs, self.loss_history['clustering_loss'], 'g-', label='Pérdida de Clustering')
		plt.title('Pérdidas por Componente')
		plt.xlabel('Épocas')
		plt.ylabel('Pérdida')
		plt.grid(True)
		plt.legend()
		
		plt.tight_layout()
		plt.show()
	
	def __call__(self, X):
		"""
		Permite llamar al modelo directamente como una función.
		
		Args:
			X: Datos de entrada
			
		Returns:
			tuple: (características, clusters) extraídos de los datos de entrada
		"""
		features = self.encoder(X)
		clusters = self.cluster(features)
		return features, clusters

	def plot_similarity_matrix(self, X, n_samples=10):
		"""
		Visualiza la matriz de similitud para un conjunto de muestras.
		
		Args:
			X: Datos de entrada
			n_samples (int): Número de muestras a visualizar
			
		Returns:
			numpy.ndarray: Matriz de similitud calculada
		"""
		if n_samples < X.shape[0]:
			indices = np.random.choice(X.shape[0], n_samples, replace=False)
			samples = X[indices]
		else:
			samples = X
			
		augX_1 = self.data_augmentation_1(samples) # Aplicar data augmentation
		augX_2 = self.data_augmentation_2(samples)
		
		augX_1comp = self.encoder(augX_1) # Obtener representaciones
		augX_2comp = self.encoder(augX_2)
		
		M = tf.matmul(augX_1comp, augX_2comp, transpose_b=True).numpy()
		
		# Visualizar
		plt.figure(figsize=(8, 8))
		plt.imshow(M, cmap='viridis')
		plt.colorbar()
		plt.title('Matriz de Similitud')
		plt.xlabel('Aumentación 2')
		plt.ylabel('Aumentación 1')
		plt.tight_layout()
		plt.show()
		
		return M


#---

class SupervisedLoss():
	"""
	Implementación de pérdida supervisada para clasificación.
	
	Esta clase encapsula la función de pérdida de entropía cruzada
	para tareas de clasificación supervisada.
	"""
	def __init__(self):
		"""
		Inicializa SupervisedLoss.
		"""
		self.loss_fn = losses.SparseCategoricalCrossentropy()
		
	def __call__(self, y_true, y_pred, sample_weights=None):
		"""
		Calcula la pérdida supervisada.
		
		Args:
			y_true: Etiquetas verdaderas
			y_pred: Predicciones del modelo
			sample_weights: Pesos opcionales para las muestras
			
		Returns:
			float: Valor de pérdida calculado
		"""
		return self.loss_fn(y_true, y_pred, sample_weight=sample_weights)

class SemiSupervisedContrastiveModel():
	"""
	Modelo contrastivo semi-supervisado que combina aprendizaje contrastivo
	con clasificación supervisada.
	
	Este modelo extiende el enfoque contrastivo para aprovechar tanto datos
	etiquetados como no etiquetados, combinando pérdidas supervisadas y no supervisadas.
	
	Atributos:
		input_shape: Forma de las imágenes de entrada
		output_dim (int): Número de dimensiones de salida (clases)
		lambda_param (float): Peso para la pérdida de clustering
		lambda_supervised (float): Peso para la pérdida supervisada
		contrastive_loss: Función de pérdida para aprendizaje contrastivo
		clustering_loss: Función de pérdida para clustering
		supervised_loss: Función de pérdida para aprendizaje supervisado
		optimizer: Optimizador para el entrenamiento
		data_augmentation_1: Primera pipeline de aumentación de datos
		data_augmentation_2: Segunda pipeline de aumentación de datos
		encoder: Red neuronal para extracción de características
		cluster: Red neuronal para asignación de clusters/clases
		loss_history (dict): Historial de valores de pérdida durante el entrenamiento
	"""
	def __init__(self, input_shape, lambda_param=0.5, lambda_supervised=1.0, temperature=0.5, 
				 learning_rate=0.0005, l2_lambda=0.01, dropout_prob=0.01):
		"""
		Inicializa SemiSupervisedContrastiveModel con los parámetros especificados.
		
		Args:
			input_shape: Forma de las imágenes de entrada
			lambda_param (float): Factor de peso para la pérdida de clustering
			lambda_supervised (float): Factor de peso para la pérdida supervisada
			temperature (float): Parámetro de temperatura para la pérdida contrastiva
			learning_rate (float): Tasa de aprendizaje inicial
			l2_lambda (float): Factor de regularización L2
			dropout_prob (float): Probabilidad de dropout
		"""
		self.input_shape = input_shape
		self.output_dim = 100
		self.lambda_param = lambda_param
		self.lambda_supervised = lambda_supervised
		self.contrastive_loss = ContrastiveLoss(temperature=temperature)
		self.clustering_loss = ClusteringLoss()
		self.supervised_loss = SupervisedLoss()

		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=learning_rate,
			decay_steps=30,
			decay_rate=0.8)
		
		self.optimizer = optimizers.AdamW(
			learning_rate=lr_schedule,
			clipnorm=5,
		)
		self.data_augmentation_1 = models.Sequential([
				layers.RandomFlip("horizontal"), 
				layers.RandomGaussianBlur(factor=(0,.5)),
				layers.RandomColorJitter(value_range=(0,1),hue_factor=(0.1, 0.1)),
				layers.RandomRotation(0.05),
				layers.RandomTranslation(0.15, 0.15),
				layers.RandomZoom(.15),
			])
	
		self.data_augmentation_2 = tf.keras.models.Sequential([
				layers.RandomFlip("horizontal"),  
				layers.RandomTranslation(0.15, 0.15),
				layers.RandomGaussianBlur(factor=.5),
				layers.RandomRotation(.15),
				layers.Resizing(38, 38), 
				layers.RandomCrop(32, 32), 
			])
			
		# Definir modelo convolucional
		input_layer = layers.Input(shape=self.input_shape)

		# Data augmentation layers
		x = layers.RandomFlip("horizontal")(input_layer)
		x = layers.RandomRotation(0.2)(x)
		x = layers.RandomZoom(0.2)(x)
		x = layers.RandomTranslation(0.1, 0.1)(x)
		x = layers.RandomGaussianBlur(factor=0.5)(x)
		x = layers.RandomContrast(0.3)(x)
	
		# Encoder - Primer bloque convolucional
		x = layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Dropout(dropout_prob / 2)(x)
	
		# Encoder - Segundo bloque convolucional
		x = layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Dropout(dropout_prob / 2)(x)
	
		# Encoder - Tercer bloque convolucional
		x = layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda))(x)
		code = layers.BatchNormalization()(x)
		
		# Capa de clustering
		flatten_layer = layers.Flatten()(code)
		
		clustering = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(flatten_layer)
		clustering = layers.BatchNormalization()(clustering)
		clustering = layers.Dropout(dropout_prob)(clustering)
		
		cluster_layer = layers.Dense(self.output_dim, activation="softmax", name='classifier')(clustering)

		# Modelo final
		self.encoder = Model(input_layer, outputs=flatten_layer)
		self.cluster = Model(flatten_layer, outputs=cluster_layer)
		
		# Historial de pérdidas para graficar
		self.loss_history = {
			'total_loss': [],
			'contrastive_loss': [],
			'clustering_loss': [],
			'supervised_loss': []
		}
	
	def train_step(self, data, temperature):	
		"""
		Paso de entrenamiento que maneja datos etiquetados y no etiquetados.
		
		Args:
			data: Tupla de (X, y, is_labeled) donde:
				  - X son los datos de entrada
				  - y son las etiquetas (puede ser None para datos no etiquetados)
				  - is_labeled es una máscara booleana que indica qué muestras están etiquetadas
			temperature: Parámetro de temperatura para la pérdida contrastiva
			
		Returns:
			dict: Diccionario con valores de pérdida para este paso
		"""
		X, y, is_labeled = data
		batch_size = tf.shape(X)[0]
		
		# Aplicar las dos transformaciones de data augmentation
		augX_1 = self.data_augmentation_1(X)
		augX_2 = self.data_augmentation_2(X)
		
		with tf.GradientTape() as tape:
			# Obtener embeddings normalizados
			augX_1comp = tf.nn.l2_normalize(self.encoder(augX_1), axis=1)
			augX_2comp = tf.nn.l2_normalize(self.encoder(augX_2), axis=1)

			# Obtener asignaciones de cluster
			cX_1comp = self.cluster(augX_1comp)
			cX_2comp = self.cluster(augX_2comp)
			
			# Calcular matriz de similitud
			M = tf.matmul(augX_1comp, augX_2comp, transpose_b=True)
			
			# Calcular pérdidas no supervisadas
			loss_M = self.contrastive_loss(M)
			loss_C = self.clustering_loss(cX_1comp, cX_2comp)
			
			# Inicializar pérdida supervisada a 0
			loss_S = 0.0
			
			# Aplicar pérdida supervisada solo si hay muestras etiquetadas
			if tf.reduce_sum(tf.cast(is_labeled, tf.float32)) > 0:
				# Crear pesos de muestra basados en qué muestras están etiquetadas
				sample_weights = tf.cast(is_labeled, tf.float32)
				
				# Usar el promedio de las dos vistas para la predicción
				avg_pred = (cX_1comp + cX_2comp) / 2.0
				
				# Calcular pérdida supervisada con pesos de muestra
				loss_S = self.supervised_loss(y, avg_pred, sample_weights)
			
			# Pérdida combinada
			total_loss = loss_M + self.lambda_param * loss_C + self.lambda_supervised * loss_S
			
		# Calcular gradientes y actualizar pesos
		gradients = tape.gradient(total_loss, self.encoder.trainable_variables + self.cluster.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.cluster.trainable_variables))
		
		return {
			"loss": total_loss, 
			"contrastive_loss": loss_M, 
			"clustering_loss": loss_C,
			"supervised_loss": loss_S
		}
	
	def prepare_data_batch(self, X_unlabeled, X_labeled=None, y_labeled=None):
		"""
		Prepara un batch combinado de datos etiquetados y no etiquetados.
		
		Args:
			X_unlabeled: Datos no etiquetados
			X_labeled: Datos etiquetados (opcional)
			y_labeled: Etiquetas para los datos etiquetados (opcional)
			
		Returns:
			tuple: (X, y, is_labeled) donde is_labeled es una máscara booleana
		"""
		if X_labeled is not None and y_labeled is not None:
			# Combinar datos etiquetados y no etiquetados
			X_combined = np.concatenate([X_unlabeled, X_labeled], axis=0)
			
			# Crear etiquetas para todos los datos (establecer etiquetas ficticias para datos no etiquetados)
			y_combined = np.zeros(X_combined.shape[0], dtype=np.int32)
			y_combined[X_unlabeled.shape[0]:] = y_labeled.squeeze()
			
			# Crear máscara que indica qué muestras están etiquetadas
			is_labeled = np.zeros(X_combined.shape[0], dtype=bool)
			is_labeled[X_unlabeled.shape[0]:] = True
			
			return X_combined, y_combined, is_labeled
		else:
			# Solo datos no etiquetados
			dummy_labels = np.zeros(X_unlabeled.shape[0], dtype=np.int32)
			is_labeled = np.zeros(X_unlabeled.shape[0], dtype=bool)
			return X_unlabeled, dummy_labels, is_labeled
	
	def mini_batches(self, X_unlabeled, X_labeled=None, y_labeled=None, batch_size=128):
		"""
		Genera mini-batches a partir de datos etiquetados y no etiquetados.
		
		Args:
			X_unlabeled: Datos no etiquetados
			X_labeled: Datos etiquetados (opcional)
			y_labeled: Etiquetas para los datos etiquetados (opcional)
			batch_size: Tamaño del batch
			
		Yields:
			tuple: Batch de datos preparados para entrenamiento
		"""
		n_unlabeled = X_unlabeled.shape[0]
		n_labeled = 0 if X_labeled is None else X_labeled.shape[0]
		
		# Mezclar índices para datos etiquetados y no etiquetados
		unlabeled_indices = np.random.permutation(n_unlabeled)
		labeled_indices = np.random.permutation(n_labeled) if n_labeled > 0 else None
		
		# Calcular cuántas muestras etiquetadas incluir en cada batch
		# Usar una proporción para mantener un equilibrio entre etiquetadas y no etiquetadas
		if n_labeled > 0:
			# Apuntar a aproximadamente 25% de muestras etiquetadas en cada batch, si están disponibles
			labeled_per_batch = min(n_labeled, max(1, int(batch_size * 0.25)))
			unlabeled_per_batch = batch_size - labeled_per_batch
		else:
			labeled_per_batch = 0
			unlabeled_per_batch = batch_size
		
		# Generar batches
		for i in range(0, n_unlabeled, unlabeled_per_batch):
			# Obtener índices no etiquetados para este batch
			end_idx = min(i + unlabeled_per_batch, n_unlabeled)
			batch_unlabeled_indices = unlabeled_indices[i:end_idx]
			batch_X_unlabeled = X_unlabeled[batch_unlabeled_indices]
			
			# Si tenemos datos etiquetados, incluir algunos en el batch
			if n_labeled > 0:
				# Ciclar a través de datos etiquetados si es necesario
				start_labeled = (i // unlabeled_per_batch * labeled_per_batch) % n_labeled
				end_labeled = min(start_labeled + labeled_per_batch, n_labeled)
				
				# Si damos la vuelta, tomar algunos del principio
				if end_labeled - start_labeled < labeled_per_batch:
					remaining = labeled_per_batch - (end_labeled - start_labeled)
					batch_labeled_indices = np.concatenate([
						labeled_indices[start_labeled:end_labeled],
						labeled_indices[:remaining]
					])
				else:
					batch_labeled_indices = labeled_indices[start_labeled:end_labeled]
				
				batch_X_labeled = X_labeled[batch_labeled_indices]
				batch_y_labeled = y_labeled[batch_labeled_indices]
				
				# Preparar el batch combinado
				batch_data = self.prepare_data_batch(batch_X_unlabeled, batch_X_labeled, batch_y_labeled)
			else:
				batch_data = self.prepare_data_batch(batch_X_unlabeled)
				
			yield batch_data
	
	def train(self, X_unlabeled, X_labeled=None, y_labeled=None, epochs=10, batch_size=128, temperature=0.5):
		"""
		Entrenamiento del modelo con datos etiquetados y no etiquetados.
		
		Args:
			X_unlabeled: Datos no etiquetados para entrenamiento
			X_labeled: Datos etiquetados para entrenamiento (opcional)
			y_labeled: Etiquetas correspondientes a X_labeled (opcional)
			epochs: Número de épocas de entrenamiento
			batch_size: Tamaño del batch para entrenamiento
			temperature: Parámetro de temperatura para la pérdida contrastiva
		"""
		# Reiniciar el historial de pérdida si comenzamos un nuevo entrenamiento
		self.loss_history = {
			'total_loss': [],
			'contrastive_loss': [],
			'clustering_loss': [],
			'supervised_loss': []
		}
		
		for epoch in range(epochs):
			epoch_total_loss = 0
			epoch_contrastive_loss = 0
			epoch_clustering_loss = 0
			epoch_supervised_loss = 0
			batch_count = 0
			
			for batch_data in self.mini_batches(X_unlabeled, X_labeled, y_labeled, batch_size=batch_size):
				print('.',end='')
				loss_dict = self.train_step(batch_data, temperature=temperature)
				
				epoch_total_loss += loss_dict["loss"]
				epoch_contrastive_loss += loss_dict["contrastive_loss"]
				epoch_clustering_loss += loss_dict["clustering_loss"]
				epoch_supervised_loss += loss_dict["supervised_loss"]
				batch_count += 1
			
			# Calcular promedios para la época
			avg_total_loss = epoch_total_loss / batch_count
			avg_contrastive_loss = epoch_contrastive_loss / batch_count
			avg_clustering_loss = epoch_clustering_loss / batch_count
			avg_supervised_loss = epoch_supervised_loss / batch_count
			
			# Guardar en el historial
			self.loss_history['total_loss'].append(avg_total_loss.numpy())
			self.loss_history['contrastive_loss'].append(avg_contrastive_loss.numpy())
			self.loss_history['clustering_loss'].append(avg_clustering_loss.numpy())
			self.loss_history['supervised_loss'].append(avg_supervised_loss.numpy())
			
			print(f"\nEpoch {epoch+1}/{epochs}, Total Loss: {avg_total_loss:.4f}, "
				  f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
				  f"Clustering Loss: {avg_clustering_loss:.4f}, "
				  f"Supervised Loss: {avg_supervised_loss:.4f}")
	
	def plot_training_history(self, figsize=(15, 6)):
		"""
		Visualiza el historial de pérdidas durante el entrenamiento.
		
		Args:
			figsize (tuple): Tamaño de la figura para visualización
		"""
		import matplotlib.pyplot as plt
		
		epochs = range(1, len(self.loss_history['total_loss']) + 1)
		
		plt.figure(figsize=figsize)
		
		# Gráfico de pérdida total
		plt.subplot(1, 3, 1)
		plt.plot(epochs, self.loss_history['total_loss'], 'b-', label='Pérdida Total')
		plt.title('Pérdida Total')
		plt.xlabel('Épocas')
		plt.ylabel('Pérdida')
		plt.grid(True)
		plt.legend()
		
		# Gráfico comparativo de pérdidas sin supervisión
		plt.subplot(1, 3, 2)
		plt.plot(epochs, self.loss_history['contrastive_loss'], 'r-', label='Pérdida Contrastiva')
		plt.plot(epochs, self.loss_history['clustering_loss'], 'g-', label='Pérdida de Clustering')
		plt.title('Pérdidas No Supervisadas')
		plt.xlabel('Épocas')
		plt.ylabel('Pérdida')
		plt.grid(True)
		plt.legend()
		
		# Gráfico de pérdida supervisada
		plt.subplot(1, 3, 3)
		plt.plot(epochs, self.loss_history['supervised_loss'], 'c-', label='Pérdida Supervisada')
		plt.title('Pérdida Supervisada')
		plt.xlabel('Épocas')
		plt.ylabel('Pérdida')
		plt.grid(True)
		plt.legend()
		
		plt.tight_layout()
		plt.show()
	
	def __call__(self, X):
		"""
		Permite llamar al modelo directamente como una función.
		
		Args:
			X: Datos de entrada
			
		Returns:
			tuple: (características, clusters) extraídos de los datos de entrada
		"""
		features = self.encoder(X)
		clusters = self.cluster(features)
		return features, clusters

	def extract_features(self, X):
		"""
		Extrae características del modelo encoder.
		
		Args:
			X: Datos de entrada
			
		Returns:
			numpy.ndarray: Características extraídas
		"""
		return self.encoder(X).numpy()
	
	def predict(self, X):
		"""
		Predice asignaciones de cluster para los datos de entrada.
		
		Args:
			X: Datos de entrada
			
		Returns:
			numpy.ndarray: Asignaciones de cluster predichas
		"""
		features = self.encoder(X)
		clusters = self.cluster(features)
		return tf.argmax(clusters, axis=1).numpy()

	def score(self, x, y):
		"""
		Calcula la precisión del modelo en los datos proporcionados.
		
		Args:
			x: Datos de entrada
			y: Etiquetas verdaderas
			
		Returns:
			float: Precisión del modelo
		"""
		return accuracy_score(self.predict(x), y)
		
	
	def plot_similarity_matrix(self, X, n_samples=10):
		"""
		Visualiza la matriz de similitud para un conjunto de muestras.
		
		Args:
			X: Datos de entrada
			n_samples (int): Número de muestras a visualizar
			
		Returns:
			numpy.ndarray: Matriz de similitud calculada
		"""
		if n_samples < X.shape[0]:
			indices = np.random.choice(X.shape[0], n_samples, replace=False)
			samples = X[indices]
		else:
			samples = X
			
		augX_1 = self.data_augmentation_1(samples) # Aplicar data augmentation
		augX_2 = self.data_augmentation_2(samples)
		
		augX_1comp = self.encoder(augX_1) # Obtener representaciones
		augX_2comp = self.encoder(augX_2)
		
		M = tf.matmul(augX_1comp, augX_2comp, transpose_b=True).numpy()
		
		# Visualizar
		plt.figure(figsize=(8, 8))
		plt.imshow(M, cmap='viridis')
		plt.colorbar()
		plt.title('Matriz de Similitud')
		plt.xlabel('Aumentación 2')
		plt.ylabel('Aumentación 1')
		plt.tight_layout()
		plt.show()
		
		return M