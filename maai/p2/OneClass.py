from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, callbacks, regularizers, backend, optimizers, models
import tensorflow as tf
import numpy as np
import seaborn as sns

class ChangeRCallback(callbacks.Callback):
    """
    Este callback recalcula el valor apropiado de 'r' después de cada época basándose en las predicciones
    actuales del modelo. También monitorea la convergencia y puede detener el entrenamiento cuando el valor
    de 'r' se estabiliza.
    
    Atributos:
        train_data: Los datos de entrenamiento utilizados para calcular el nuevo valor de 'r'
        delta (float): Umbral para el cambio en 'r' para considerar que el modelo ha convergido
        steps (int): Número de épocas consecutivas con cambios pequeños necesarios para la convergencia
        cont (int): Contador de épocas consecutivas con cambios pequeños
    """
    def __init__(self, train_data, delta=.025, steps=3):
        """
        Args:
            train_data: Datos de entrenamiento para calcular los nuevos valores de 'r'
            delta (float): Cambio máximo en 'r' para ser considerado pequeño (para detección de convergencia)
            steps (int): Número de épocas consecutivas con cambios pequeños necesarios para convergencia
        """
        super().__init__()
        self.train_data = train_data
        self.delta = delta
        self.steps = steps
        self.cont = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Recalcula el valor de 'r' al final de cada época y verifica la convergencia.
        
        El valor de 'r' se establece en el punto donde la fracción de puntos dentro del límite
        es igual al parámetro 'nu' especificado.
        
        Args:
            epoch: Número de época actual
            logs: Diccionario de registros (no utilizado)
        """
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

class AnomalyDetector:
    """
    Esta implementación utiliza una red neuronal para mapear los datos de entrada a un espacio de características
    donde se puede establecer un límite de hiperesfera para separar instancias normales de anomalías.
    El modelo se entrena solo con instancias normales y aprende a identificar valores atípicos basándose en
    una función de pérdida basada en margen.
    
    Atributos:
        model (keras.Model): El modelo de red neuronal subyacente
        optimizer (keras.optimizers): El optimizador utilizado para el entrenamiento
    """
    def __init__(self, input_shape, nu=.5, l2_lambda=.0001, learning_rate=0.001, dropout_prob=0.1):
        """
        Inicializa el modelo detector de anomalías.
        
        Args:
            input_shape (tuple): La forma de las imágenes de entrada (alto, ancho, canales)
            nu (float): El límite superior en la fracción de valores atípicos y límite inferior en la fracción de vectores de soporte
            l2_lambda (float): Coeficiente de regularización L2
            learning_rate (float): Tasa de aprendizaje para el optimizador
            dropout_prob (float): Probabilidad de dropout para regularización
        """
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

        # Variables no entrenables para aprendizaje de una clase
        self.model.r = tf.Variable(1.0, trainable=False, name='r', dtype=tf.float32)
        self.model.nu = tf.Variable(nu, trainable=False, name='nu', dtype=tf.float32)

        self.optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )
      
    def loss_function(self, y_true, y_pred):
        """
        Implementa una pérdida basada en margen similar a la utilizada en One-Class SVM.
        Penaliza los puntos que caen fuera del límite de la hiperesfera definida por el radio r.
        
        Args:
            y_true: No utilizado (requerido por la API de Keras)
            y_pred: Predicciones del modelo
            
        Returns:
            El valor de pérdida calculado
        """
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
        """
        Args:
            X: Datos de entrenamiento (solo instancias normales)
            y: No utilizado, incluido para compatibilidad con la API
            sample_weight: Pesos de muestra opcionales
            batch_size (int): Tamaño del lote para entrenamiento
            epochs (int): Número máximo de épocas de entrenamiento
            delta (float): Cambio máximo en 'r' para considerar convergencia
            steps (int): Número de épocas consecutivas con cambios pequeños necesarios para convergencia
            
        Returns:
            El modelo entrenado
        """
        dummy_y = np.zeros((len(X), 1)) # Necesario pasar como salida para que keras no de un error
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function)
        self.model.fit(X, dummy_y, callbacks=[ChangeRCallback(X,delta=delta,steps=steps)], batch_size=batch_size, epochs=epochs, verbose=1)
        return self.model
        
    def predict(self, X):
        """
        Genera predicciones del modelo para los datos de entrada.
        
        Valores más altos indican instancias más normales, mientras que valores más bajos indican posibles anomalías.
        
        Args:
            X: Datos de entrada para predecir
            
        Returns:
            Predicciones del modelo
        """
        return self.model.predict(X)
    
    def get_encoded_data(self, X):
        """
        Extrae representaciones codificadas del modelo.
        
        Args:
            X: Datos de entrada
            
        Returns:
            Representaciones codificadas de una capa intermedia
        """
        return self.model.layers[1].predict(X)
        
    def __del__(self):
        """
        Limpia los recursos cuando el objeto es eliminado.
        """
        del self.model
        backend.clear_session() # Necesario para liberar la memoria en GPU

    def plot_confusion_matrix(self, x_test, y_test):
        """
        Grafica la matriz de confusión para la evaluación del modelo.
        
        Args:
            x_test: Datos de entrada de prueba
            y_test: Etiquetas verdaderas para los datos de prueba
        """
        y_pred = self.predict_class(x_test)
        
        if len(y_true) == 1:
            y_true = y_test
        else:
            y_true = np.argmax(y_test, axis=1)  
            
        cm = confusion_matrix(y_true, y_pred) 
        
        # Plot confusion matrix 
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(self.num_classes), yticklabels=np.arange(self.num_classes))
        plt.xlabel('Etiquetas Predichas')
        plt.ylabel('Etiquetas Verdaderas')
        plt.title('Matriz de Confusión')
        plt.show()