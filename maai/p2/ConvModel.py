from tensorflow.keras import layers, models, regularizers, optimizers, losses, callbacks, backend
import numpy as np
import matplotlib.pyplot as plt

class ConvModel:
	def __init__(self, learning_rate=0.0005, dropout_prob=0.3, l2_lambda=0.003):
		self.output_dim = 100
		self.model = models.Sequential([
			# Data augmentation layers - aumentados ligeramente
			layers.RandomFlip("horizontal_and_vertical", input_shape=(32, 32, 3)),
			layers.RandomRotation(0.2),
			layers.RandomZoom(0.2),
			layers.RandomTranslation(0.1, 0.1),  # Añadido traslación
			layers.RandomGaussianBlur(factor=0.5),
			layers.RandomContrast(0.3),
			
			# Primer bloque convolucional - más filtros
			layers.Conv2D(96, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3), 
						  kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
			layers.Dropout(dropout_prob/2),
			
			# Segundo bloque convolucional - más filtros
			layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
			layers.Dropout(dropout_prob/2),
   
			# Tercer bloque convolucional
			layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
						  kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
			layers.Dropout(dropout_prob),
   
			# Global Average Pooling en lugar de Flatten
			layers.GlobalAveragePooling2D(),
   
			# Capas fully connected
			layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.Dropout(dropout_prob),
   
			# Capa extra intermedia
			layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.Dropout(dropout_prob/2),
			
			layers.Dense(self.output_dim, activation="softmax")
		])
		
		# Learning rate scheduler con CosineDecay
		lr_schedule = optimizers.schedules.CosineDecay(
			initial_learning_rate=learning_rate,
			decay_steps=2000,
			alpha=0.1  # learning rate mínima al final
		)
		
		self.optimizer = optimizers.AdamW(
			learning_rate=lr_schedule,
			clipnorm=1,
			weight_decay=1e-4
		)

		# Loss con label smoothing
		self.loss = losses.SparseCategoricalCrossentropy()
		
		self.model.compile(
			loss=self.loss,
			optimizer=self.optimizer,
			metrics=['accuracy']
		)
	
	def fit(self, X, y, validation_data=None, batch_size=64, epochs=100, patience=8, sample_weight=None):
		# Callbacks mejorados
		callback_list = [
			callbacks.EarlyStopping(
				monitor="val_loss" if validation_data else "loss",
				patience=patience,
				restore_best_weights=True,
				verbose=1
			)
		]
		
		history = self.model.fit(
				X, y, 
				batch_size=batch_size, 
				epochs=epochs,
				validation_data=validation_data,
				callbacks=callback_list,
				verbose=1,
				sample_weight=sample_weight  
		)
		return history
	
	# El método self_training_v2 permanece igual
	@staticmethod
	def self_training_v2(model_func, x_train, y_train, unlabeled_data, validation_data=None, thresh=0.8, train_epochs=3):
		# Make copies of the input data
		train_data = np.array(x_train, copy=True)
		train_label = np.array(y_train, copy=True)
		current_unlabeled = np.array(unlabeled_data, copy=True)
		
		# Initialize sample weights
		sample_weights = np.ones(len(train_label)) * 2.0
		
		for i in range(train_epochs):
			if len(current_unlabeled) == 0:
				print("No more unlabeled data left")
				break
				
			# Create and train new model
			model = model_func()
			model.fit(train_data, train_label, sample_weight=sample_weights)
			
			# Predict on unlabeled data
			y_pred = model.predict_proba(current_unlabeled)
			y_class = np.argmax(y_pred, axis=1)
			y_value = np.max(y_pred, axis=1)
			
			# Select high confidence predictions
			high_confidence = y_value > thresh
			
			if np.any(high_confidence):
				# Get confident predictions
				new_data = current_unlabeled[high_confidence]
				new_labels = y_class[high_confidence]
				new_probs = y_value[high_confidence]
				
				# Add to training set
				train_data = np.vstack([train_data, new_data])
				train_label = np.append(train_label, new_labels)
				sample_weights = np.append(sample_weights, new_probs)
				
				# Remove used examples from unlabeled data
				current_unlabeled = current_unlabeled[~high_confidence]
				
				print(f"Epoch {i+1}: Added {len(new_data)} samples, {len(current_unlabeled)} remaining")
			else:
				print(f"Epoch {i+1}: No samples added")
		
		# Train final model
		final_model = model_func()
		final_model.fit(train_data, train_label, validation_data=validation_data, sample_weight=sample_weights)
		
		return final_model
	
	def predict(self, X):
		return np.argmax(self.model.predict(X), axis=1)
	
	def predict_proba(self, X):
		return self.model.predict(X)
	
	def score(self, X, y):
		_, acc = self.model.evaluate(X, y)
		return acc
	
	def plot(self, history):
		plt.figure(figsize=(12, 5))
	
		# Subplot 1: Loss
		plt.subplot(1, 2, 1)
		plt.plot(history.history['loss'], label='Training Loss')
		if 'val_loss' in history.history:
			plt.plot(history.history['val_loss'], label='Validation Loss')
		plt.title('Model Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend(loc='upper right')
	
		# Subplot 2: Accuracy
		plt.subplot(1, 2, 2)
		plt.plot(history.history['accuracy'], label='Training Accuracy')
		if 'val_accuracy' in history.history:
			plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
		plt.title('Model Accuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend(loc='lower right')
	
		# Show plots
		plt.tight_layout()
		plt.show()
	
	def __call__(self, X):
		return self.model.predict(X)
	
	def summary(self):
		return self.model.summary()
	
	def __del__(self):
		del self.model
		backend.clear_session()  # Liberar memoria en GPU