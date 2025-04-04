from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, optimizers, backend, regularizers, callbacks
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os
import seaborn as sns

os.environ['KERAS_BACKEND'] = 'JAX'

class TwoStepAutoEncoder():
    """
    Autoencoder implementado en dos pasos separados: codificador y decodificador.
    
    Este modelo aprende representaciones no supervisadas de los datos de entrada
    utilizando una arquitectura de autoencoder. Está diseñado para trabajar con
    imágenes y utiliza bloques convolucionales para la extracción de características.
    
    Atributos:
        input_shape: Forma de las imágenes de entrada
        encoder: Modelo para codificar imágenes en representaciones latentes
        decoder: Modelo para decodificar representaciones latentes a imágenes
        autoencoder: Modelo completo que combina encoder y decoder
        optimicer: Optimizador configurado para el entrenamiento
    """
    def __init__(self, input_shape, learning_rate=0.001, l2_lambda=0.01, dropout_prob=0.1):
        """
        Args:
            input_shape: Forma de las imágenes de entrada
            learning_rate (float): Tasa de aprendizaje para el optimizador
            l2_lambda (float): Factor de regularización L2
            dropout_prob (float): Probabilidad de dropout
        """
        self.input_shape = input_shape
        
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(32, 32, 3)),
    
            # Data augmentation layers
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomGaussianBlur(factor=0.5),
            layers.RandomContrast(0.3),
    
            # Encoder - First convolutional block
            layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_prob / 2),
    
            # Encoder - Second convolutional block
            layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_prob / 2),
    
            # Encoder - Third convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
        ])
    
        # Decoder Sequential Model
        self.decoder = models.Sequential([
            layers.Conv2DTranspose(256, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
    
            layers.Conv2DTranspose(192, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(192, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
    
            layers.Conv2DTranspose(96, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(96, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),
    
            # Output layer to reconstruct the image
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding="same"),
        ])
        
        self.autoencoder = models.Sequential([self.encoder, self.decoder])

        self.optimicer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )

        self.autoencoder.compile(optimizer=self.optimicer, loss='mse')
    
    def fit(self, X, y=None, validation_data=None, sample_weight=None, batch_size=60_000, epochs=100):
        """
        Entrena el autoencoder.
        
        Args:
            X: Datos de entrenamiento (imágenes)
            y: No utilizado, incluido para compatibilidad API
            validation_data: Datos de validación opcional (X_val, X_val)
            sample_weight: Pesos opcionales para las muestras
            batch_size (int): Tamaño del batch para entrenamiento
            epochs (int): Número de épocas de entrenamiento
            
        Returns:
            history: Historial de entrenamiento
        """
        return self.autoencoder.fit(X, X, 
                             validation_data=validation_data,
                             batch_size=batch_size, 
                             epochs=epochs, 
                             sample_weight=sample_weight)

    def get_encoded_data(self, X):
        """
        Obtiene representaciones codificadas para los datos de entrada.
        
        Args:
            X: Datos de entrada
            
        Returns:
            numpy.ndarray: Representaciones codificadas
        """
        return self.encoder.predict(X)

    def get_encoded_data_batched(self, X, batch_size=128):
        """
        Obtiene representaciones codificadas para los datos de entrada usando procesamiento por batches.
        
        Útil para grandes conjuntos de datos que no caben en memoria.
        
        Args:
            X: Datos de entrada
            batch_size (int): Tamaño del batch para procesamiento
            
        Returns:
            numpy.ndarray: Representaciones codificadas combinadas
        """
        results = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            batch_encoded = self.encoder(batch)
            results.append(batch_encoded)
        return np.concatenate(results, axis=0)

    def __call__(self, X):
        """
        Permite usar el autoencoder como una función.
        
        Args:
            X: Datos de entrada
            
        Returns:
            numpy.ndarray: Reconstrucción de los datos de entrada
        """
        return self.autoencoder.predict(X)
        
    def __del__(self):
        """
        Limpia recursos cuando se elimina el objeto.
        """
        backend.clear_session() # Necesario para liberar la memoria en GPU

#--------------------------------------------------------------------------------------------#

class TwoStepClassifier:
    """
    Clasificador que opera en dos pasos, utilizando características extraídas por un autoencoder.
    
    Esta clase implementa un clasificador que se entrena sobre representaciones codificadas,
    permitiendo utilizar características aprendidas de manera no supervisada para tareas de clasificación.
    
    Atributos:
        output_dim (int): Dimensión de salida (número de clases)
        classifier: Modelo de clasificación que opera sobre características extraídas
        optimizer: Optimizador configurado para el entrenamiento
    """
    def __init__(self, dropout_prob=0.1, l2_lambda=0.0005, learning_rate=0.01):
        """
        Inicializa el clasificador de dos pasos.
        
        Args:
            dropout_prob (float): Probabilidad de dropout para regularización
            l2_lambda (float): Factor de regularización L2
            learning_rate (float): Tasa de aprendizaje para el optimizador
        """
        self.output_dim = 100

        self.classifier = models.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_prob),
   
            # Capa extra intermedia
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_prob/2),
            
            layers.Dense(self.output_dim, activation="softmax")
        ])

        self.optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )

        self.classifier.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    def fit(self, X, y, validation_data=None, sample_weight=None, batch_size=60_000, epochs=350):
        """
        Entrena el clasificador.
        
        Args:
            X: Características codificadas para entrenamiento
            y: Etiquetas para entrenamiento
            validation_data: Datos de validación opcional (X_val, y_val)
            sample_weight: Pesos opcionales para las muestras
            batch_size (int): Tamaño del batch para entrenamiento
            epochs (int): Número de épocas de entrenamiento
            
        Returns:
            history: Historial de entrenamiento
        """
        return self.classifier.fit(X, y, 
                            validation_data=validation_data,
                            batch_size=batch_size, 
                            epochs=epochs, 
                            sample_weight=sample_weight)

    def predict(self, X):
        """
        Predice la clase para nuevas características.
        
        Args:
            X: Características codificadas para predecir
            
        Returns:
            int: Clase predicha
        """
        return np.argmax(self.predict_proba(X)) + 1
    
    def predict_proba(self, X):
        """
        Calcula probabilidades para cada clase.
        
        Args:
            X: Características codificadas para predecir
            
        Returns:
            numpy.ndarray: Probabilidades para cada clase
        """
        return self.classifier.predict(X)
    
    def score(self, X, y):
        """
        Evalúa el rendimiento del clasificador.
        
        Args:
            X: Características codificadas para evaluación
            y: Etiquetas verdaderas
            
        Returns:
            float: Precisión del clasificador
        """
        return self.classifier.evaluate(X, y)[1]

    def __del__(self):
        """
        Limpia recursos cuando se elimina el objeto.
        """
        backend.clear_session() # Necesario para liberar la memoria en GPU


#--------------------------------------------------------------------------------------------#

def TwoStepTraining(autoencoder, classifier, x_train, y_train, unlabeled_train, validation_data=None, batch_size_autoencoder=1024, epochs_autoencoder=100, batch_size_classifier=1024, epochs_classifier=100, contrastive=False):
    """
    Función para entrenar un sistema de dos pasos: autoencoder y clasificador.
    
    Este enfoque permite primero aprender representaciones no supervisadas con un autoencoder
    (utilizando tanto datos etiquetados como no etiquetados), y luego entrenar un clasificador
    sobre esas representaciones utilizando solo los datos etiquetados.
    
    Args:
        autoencoder: Modelo autoencoder
        classifier: Modelo clasificador
        x_train: Datos etiquetados para entrenamiento
        y_train: Etiquetas para entrenamiento
        unlabeled_train: Datos no etiquetados para entrenamiento
        validation_data: Datos de validación (x_val, y_val)
        batch_size_autoencoder (int): Tamaño del batch para entrenar el autoencoder
        epochs_autoencoder (int): Épocas para entrenar el autoencoder
        batch_size_classifier (int): Tamaño del batch para entrenar el clasificador
        epochs_classifier (int): Épocas para entrenar el clasificador
        contrastive (bool): Si es True, usa entrenamiento contrastivo en lugar de reconstrucción
    """
    all_x = np.vstack((x_train, unlabeled_train))
    if contrastive:
        autoencoder.train(unlabeled_train, epochs=epochs_autoencoder, batch_size=batch_size_autoencoder)
    else:
        autoencoder.fit(all_x, batch_size=batch_size_autoencoder, epochs=epochs_autoencoder, validation_data=(validation_data[0],validation_data[0]))
    x_coded = autoencoder.get_encoded_data(x_train)
    x_val, y_val = validation_data
    x_val_coded = autoencoder.get_encoded_data(x_val)
    classifier.fit(x_coded, y_train, batch_size=batch_size_classifier, epochs=epochs_classifier, validation_data=(x_val_coded,y_val))


#--------------------------------------------------------------------------------------------#

class OneStepAutoencoder:
    """
    Autoencoder que integra codificación y clasificación en un único modelo.
    
    Este modelo realiza simultáneamente aprendizaje de representaciones (autoencoder)
    y clasificación, compartiendo el codificador entre ambas tareas. Permite aprovechar
    tanto datos etiquetados como no etiquetados durante el entrenamiento.
    
    Atributos:
        input_shape: Forma de las imágenes de entrada
        output_dim (int): Número de clases para clasificación 
        model: Modelo con dos salidas (reconstrucción y clasificación)
    """
    def __init__(self, input_shape, learning_rate=0.001, decoder_extra_loss_weight=0.3, l2_lambda=0.001, dropout_prob=0.05):
        """
        Inicializa el autoencoder de un paso con funciones de codificación y clasificación.
        
        Args:
            input_shape: Forma de las imágenes de entrada
            learning_rate (float): Tasa de aprendizaje para el optimizador
            decoder_extra_loss_weight (float): Peso adicional para la pérdida del decodificador
            l2_lambda (float): Factor de regularización L2
            dropout_prob (float): Probabilidad de dropout
        """
        self.input_shape = input_shape
        self.output_dim = 100

        input_layer = layers.Input(shape=self.input_shape)

        # Data augmentation layers
        x = layers.RandomFlip("horizontal_and_vertical")(input_layer)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomTranslation(0.1, 0.1)(x)
        x = layers.RandomGaussianBlur(factor=0.5)(x)
        x = layers.RandomContrast(0.3)(x)
    
        # Encoder - First convolutional block
        x = layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_prob / 2)(x)
    
        # Encoder - Second convolutional block
        x = layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_prob / 2)(x)
    
        # Encoder - Third convolutional block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_prob)(x)
    
        # Now, the bottleneck (encoded representation) is the output of this encoder.
        encoded = x
    
        # Decoder - Upsampling and convolutional layers
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding="same")(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2DTranspose(192, (3, 3), activation='relu', padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(192, (3, 3), activation='relu', padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        x = layers.Conv2DTranspose(96, (3, 3), activation='relu', padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(96, (3, 3), activation='relu', padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
    
        # Output layer to reconstruct the image
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding="same")(x)
        decoded = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='decoder')(decoded)

        # Clasificador
        classifier = layers.Flatten()(encoded)
        classifier = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(classifier)
        classifier = layers.BatchNormalization()(classifier)
        classifier = layers.Dropout(dropout_prob)(classifier)
        
        classifier = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(classifier)
        classifier = layers.BatchNormalization()(classifier)
        classifier = layers.Dropout(dropout_prob/2)(classifier)
        
        classifier_output = layers.Dense(self.output_dim, activation="softmax", name='classifier')(classifier)

        #--------------------------------------------------------------------------------------------#
        # Modelo
        self.model = models.Model(input_layer, 
                                  {'decoder': decoded, 'classifier': classifier_output})

        self.optimicer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )
        
        self.model.compile(optimizer=self.optimicer,
                           loss={'decoder': 'mse', 'classifier': 'categorical_crossentropy'},
                           loss_weights={'decoder': 1.0 + decoder_extra_loss_weight, 'classifier': 1.0 - decoder_extra_loss_weight},  # Ajustar pesos de pérdida si es necesario
                           metrics={'decoder': [], 'classifier': ['accuracy']})
    
    def fit(self, X, y, unlabeled_train, batch_size, epochs, patience, validation_data=None):
        """
        Entrena el autoencoder de un paso utilizando datos etiquetados y no etiquetados.
        
        Esta función permite entrenar el modelo combinado, utilizando diferentes estrategias
        para los datos etiquetados (para el codificador y el clasificador) y no etiquetados
        (solo para el codificador).
        
        Args:
            X: Datos etiquetados para entrenamiento
            y: Etiquetas para entrenamiento
            unlabeled_train: Datos no etiquetados para entrenamiento
            batch_size (int): Tamaño del batch para entrenamiento
            epochs (int): Número máximo de épocas
            patience (int): Épocas a esperar sin mejora antes de detener entrenamiento
            validation_data: Datos de validación opcional
            
        Returns:
            history: Historial de entrenamiento
        """
        # Convertir a float32
        X = np.array(X, dtype=np.float32)
        unlabeled_train = np.array(unlabeled_train, dtype=np.float32)

        # Lo que vamos a usar
        all_x = np.vstack((X, unlabeled_train))
        y_zeros = np.zeros((unlabeled_train.shape[0], y.shape[1]))
        all_y = np.vstack((y, y_zeros))

        # Coeficientes para los ejemplos
        weight_autoencoder = np.ones((all_x.shape[0], all_x.shape[1], all_x.shape[2]))
        weight_classifier = np.array([1.0]*len(y) + [0.0]*len(y_zeros))

        
        h = self.model.fit(all_x,
                       {'decoder': all_x, 'classifier': all_y},
                       sample_weight={'decoder': weight_autoencoder, 'classifier': weight_classifier},
                       epochs=epochs, 
                       batch_size=batch_size,
                       validation_data=validation_data,
                       verbose=1,
                       callbacks=[callbacks.EarlyStopping(monitor="loss", 
                                                                    patience=patience)])
        return h

    def predict_class(self, X):
        """
        Predice las clases para las muestras en X.
        
        Args:
            X: Datos de entrada
            
        Returns:
            numpy.ndarray: Clases predichas
        """
        class_label, _ = self.model.predict(X)
        return class_label.argmax(axis=1)
    
    def predict_image(self, X):
        """
        Genera reconstrucciones de las imágenes de entrada.
        
        Args:
            X: Datos de entrada
            
        Returns:
            numpy.ndarray: Imágenes reconstruidas
        """
        _, image = self.model.predict(X)
        return image
    
    def score(self, X, y):
        """
        Evalúa el rendimiento del clasificador.
        
        Args:
            X: Datos de entrada
            y: Etiquetas verdaderas
            
        Returns:
            float: Precisión del clasificador
        """
        y_pred = np.argmax(self.model.predict(X)['classifier'], axis=1)
        return accuracy_score(y, y_pred)

    def __call__(self, X):
        """
        Permite usar el modelo como una función.
        
        Args:
            X: Datos de entrada
            
        Returns:
            dict: Diccionario con predicciones del clasificador y reconstrucciones
        """
        return self.model.predict(X)

    def __del__(self):
        """
        Limpia recursos cuando se elimina el objeto.
        """
        backend.clear_session() # Necesario para liberar la memoria en GPU

    def plot_confusion_matrix(self, x_test, y_test):
        """
        Grafica la matriz de confusión para evaluación del modelo.
        
        Args:
            x_test: Datos de prueba
            y_test: Etiquetas verdaderas para los datos de prueba
        """
        y_pred = self.predict_class(x_test)
        
        if len(y_true) == 1:
            y_true = y_test
        else:
            y_true = np.argmax(y_test, axis=1)  
            
        cm = confusion_matrix(y_true, y_pred) # Compute confusion matrix
        
        # Plot confusion 
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(self.num_classes), yticklabels=np.arange(self.num_classes))
        plt.xlabel('Etiquetas Predichas')
        plt.ylabel('Etiquetas Verdaderas')
        plt.title('Matriz de Confusión')
        plt.show()

#--------------------------------------------------------------------------------------------#

def OneStepTraining(model, x_train, y_train, unlabeled_train, batch_size=60_000, epochs=1000, patience=5):
    """
    Función auxiliar para entrenar el modelo OneStepAutoencoder.
    
    Args:
        model: Modelo OneStepAutoencoder
        x_train: Datos etiquetados para entrenamiento
        y_train: Etiquetas para entrenamiento
        unlabeled_train: Datos no etiquetados para entrenamiento
        batch_size (int): Tamaño del batch para entrenamiento
        epochs (int): Número máximo de épocas
        patience (int): Épocas a esperar sin mejora antes de detener entrenamiento
        
    Returns:
        history: Historial de entrenamiento
    """
    h = model.fit(x_train, y_train, unlabeled_train, batch_size=batch_size, epochs=epochs, patience=patience)
    return h