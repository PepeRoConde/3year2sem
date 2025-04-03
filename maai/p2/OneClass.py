from tensorflow.keras import layers, callbacks, regularizers, backend, optimizers, models
import tensorflow as tf
import numpy as np

class ChangeRCallback(callbacks.Callback):
   def __init__(self, train_data, delta=.025, steps=3):
       super().__init__()
       self.train_data = train_data
       self.delta = delta
       self.steps = steps
       self.cont = 0

   def on_epoch_end(self, epoch, logs=None):
       sorted_values = np.sort(self.model.predict(self.train_data).flatten())
       new_value = sorted_values[int(len(sorted_values) * (1. - self.model.nu))]
       old_value = self.model.r.numpy()
       print('Cambiando r a', new_value, ', max:', sorted_values.max(), ', min:', sorted_values.min())
       self.model.r.assign(new_value)
       if np.abs(old_value - new_value) < self.delta:
            self.cont += 1
            if self.cont >= self.steps:
                print('Convergencia obtenida. Finalizando el entrenamiento.')
                self.model.stop_training = True
       else:
            self.cont = 0

#---------------------------------------
#---------------------------------------

class AnomalyDetector:

    def __init__(self, input_shape, nu=.5, l2_lambda=.0001, learning_rate=0.001, dropout_prob=0.1):

        self.model = models.Sequential([
			# Data augmentation layers - aumentados ligeramente
			layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
			layers.RandomRotation(0.2),
			layers.RandomZoom(0.2),
			layers.RandomTranslation(0.1, 0.1),  # Añadido traslación
			layers.RandomGaussianBlur(factor=0.5),
			
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
   
			# Capa extra intermedia
			layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
			layers.BatchNormalization(),
			layers.Dropout(dropout_prob/2),
			
			layers.Dense(1, activation="sigmoid")
        ])


        self.model.r = tf.Variable(1.0, trainable=False, name='r', dtype=tf.float32)
        self.model.nu = tf.Variable(nu, trainable=False, name='nu', dtype=tf.float32)

        self.optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )
		
	  
    def loss_function(self, y_true, y_pred):
        r = self.model.r
        nu = self.model.nu
        
        # Primera parte: regularización L2 (ya incluida por los regularizadores de capa)
        # Segunda parte: término de error basado en el margen r
        # max(0, r - y_pred) para cada predicción
        margin_error = tf.maximum(0.0, r - y_pred)
        # Calculamos la media y aplicamos el factor 1/nu
        margin_loss = tf.reduce_mean(margin_error) / nu
        
        # La función de pérdida total (la regularización ya está incluida en las capas)
        return margin_loss
	
    def fit(self, X, y=None, sample_weight=None, batch_size=64, epochs=50, delta=.025, steps=3):

        dummy_y = np.zeros((len(X), 1)) # Necesario pasar como salida para que keras no de un error
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function)
        self.model.fit(X, dummy_y, callbacks=[ChangeRCallback(X,delta=delta,steps=steps)], batch_size=batch_size, epochs=epochs, verbose=1)
        return self.model
		
    def predict(self, X):
        return self.model.predict(X)
    
    def get_encoded_data(self, X):
        return self.model.layers[1].predict(X)
        
    def __del__(self):
        del self.model
        backend.clear_session() # Necesario para liberar la memoria en GPU

    def plot_confusion_matrix(self, x_test, y_test):
        y_pred = self.predict_class(x_test)
        
        if len(y_true) == 1:
            y_true = y_test
        else:
            y_true = np.argmax(y_test, axis=1)  
            
        cm = confusion_matrix(y_true, y_pred) # Compute confusion matrix
        
        # Plot confusion matrix using seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(self.num_classes), yticklabels=np.arange(self.num_classes))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()