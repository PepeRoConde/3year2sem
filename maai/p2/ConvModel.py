import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ConvModel:
	def __init__(self, learning_rate = 0.001, dropout_prob = 0.2):

        self.output_dim = 100
        
		self.model = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D((2, 2)),
			
			tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D((2, 2)),
			
			tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D((2, 2)),
			
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(256, activation='relu'),
			tf.keras.layers.Dropout(dropout_prob),
			tf.keras.layers.Dense(self.output_dim, activation="softmax")
		])
		
		self.optimicer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )

		self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
		self.model.compile(
			loss=self.loss,
			optimizer=self.optimizer,
			metrics=['accuracy']
		)
	
	
	def fit(self, X, y, batch_size=32, epochs=50, patience, sample_weight=None):
		history = self.model.fit(X, y, 
                                 sample_weight=sample_weight, 
                                 batch_size=batch_size, 
								 epochs=epochs, 
                                 verbose=1, 
								 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", 
                                                                            patience=patience)])
		return history
	
	@staticmethod
	def self_training_v2(model_func, x_train, y_train, unlabeled_data, x_test=None, y_test=None, thresh=0.8, train_epochs=3):
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
		final_model.fit(train_data, train_label, sample_weight=sample_weights)
		
		return final_model

	def predict(self, X):
		return np.argmax(self.model.predict(X), axis=1)
	
	def predict_proba(self, X):
		return self.model.predict(X)
	
	def score(self, X, y):
		_, acc = self.model.evaluate(X, y)
		return acc
	
	def plot(self, history):
		plt.figure(figsize=(10, 5))

		# Subplot 1: Loss
		plt.subplot(1, 2, 1)
		plt.plot(history.history['loss'], label='Training Loss')
		plt.title('Model Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend(loc='upper right')

		# Subplot 2: Accuracy
		plt.subplot(1, 2, 2)
		plt.plot(history.history['accuracy'], label='Training Accuracy')
		plt.title('Model Accuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend(loc='lower right')

		# Show plots
		plt.tight_layout()
		plt.show()

    def __call__(self, X):
        return self.model.predict(X)

	def __del__(self):
		del self.model
		tf.keras.backend.clear_session()  # Liberar memoria en GPU