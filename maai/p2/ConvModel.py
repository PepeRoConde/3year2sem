from tensorflow.keras import layers, models, regularizers, optimizers, losses, callbacks, backend
import numpy as np
import matplotlib.pyplot as plt

class ConvModel:
    """
	Clase ConvModel para crear y entrenar un modelo de red neuronal convolucional.
    
    Atributos:
        output_dim (int): Dimensión de la capa de salida (número de clases)
        model (keras.Model): El modelo de keras subyacente
        optimizer (keras.optimizers): Optimizador configurado
        loss (keras.losses): Función de pérdida
    """
    def __init__(self, learning_rate=0.0005, dropout_prob=0.3, l2_lambda=0.003):
        """
        Args:
            learning_rate (float): Tasa de aprendizaje inicial para el optimizador
            dropout_prob (float): Probabilidad de dropout para regularización
            l2_lambda (float): Factor de regularización L2 para los kernels convolucionales
        """
        self.output_dim = 100
        self.model = models.Sequential([
            # Capas de aumento de datos para mejorar generalización
            layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomGaussianBlur(factor=0.5),
            
            # Primer bloque: extracción de características básicas
            layers.Conv2D(96, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3), 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_prob/2),
            
            # Segundo bloque: extracción de características más complejas
            layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Conv2D(192, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_prob/2),
   
            # Tercer bloque: características de alto nivel
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_prob),
   
            # GlobalAveragePooling2D reduce parámetros y mejora generalización
            layers.GlobalAveragePooling2D(),
   
            # Capas densas para clasificación
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_prob),
   
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_prob/2),
            
            # Capa de salida con activación softmax para clasificación multiclase
            layers.Dense(self.output_dim, activation="softmax")
        ])
        
        # Tasa de aprendizaje con decaimiento coseno
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=2000,
            alpha=0.1  # Valor mínimo al que decaerá la tasa
        )
        
        # AdamW combina Adam con decaimiento de pesos adecuado
        self.optimizer = optimizers.AdamW(
            learning_rate=lr_schedule,
            clipnorm=1,        # Previene explosión de gradientes
            weight_decay=1e-4  # Decaimiento de pesos para regularización
        )

        # Función de pérdida estándar para clasificación multiclase
        self.loss = losses.SparseCategoricalCrossentropy()
        
        # Compilación del modelo
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )
    
    def fit(self, X, y, validation_data=None, batch_size=128, epochs=100, patience=4, sample_weight=None, verbose=1):
        """
        Args:
            X (numpy.ndarray): Datos de entrenamiento
            y (numpy.ndarray): Etiquetas de entrenamiento
            validation_data (tuple): Datos y etiquetas de validación (opcional)
            batch_size (int): Tamaño del lote para entrenamiento
            epochs (int): Número máximo de épocas
            patience (int): Épocas a esperar sin mejora antes de detener entrenamiento
            sample_weight (numpy.ndarray): Pesos por muestra para el entrenamiento
            verbose (int): Nivel de feedback
            
        Returns:
            history: Historial de entrenamiento con métricas por época
        """
        # Early stopping para evitar sobreajuste
        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            )
        ]
        
        history = self.model.fit(
                X, y, 
                batch_size=batch_size, 
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callback_list,
                verbose=verbose,
                sample_weight=sample_weight  
        )
        return history
    
    @staticmethod
    def self_training_v2(model_func, x_train, y_train, unlabeled_data, validation_data=None, thresh=0.8, train_epochs=3, verbose=1):
        """
        Implementa self-training para aprendizaje semi-supervisado.
        
        1. Se entrena un modelo con datos etiquetados
        2. Se hacen predicciones sobre datos no etiquetados
        3. Predicciones de alta confianza se añaden al conjunto de entrenamiento
        4. Se repite el proceso con los datos aumentados
        5. Se entrena un modelo final con todos los datos
        
        Args:
            model_func (callable): Función que devuelve una instancia del modelo
            x_train (numpy.ndarray): Datos de entrenamiento iniciales (etiquetados)
            y_train (numpy.ndarray): Etiquetas iniciales
            unlabeled_data (numpy.ndarray): Datos no etiquetados
            validation_data (tuple): Datos de validación (opcional)
            thresh (float): Umbral de confianza para seleccionar nuevas muestras
            train_epochs (int): Número de iteraciones del proceso de self-training
            verbose (int): Nivel de feedback
            
        Returns:
            ConvModel: Modelo final entrenado con todos los datos
        """
        # Copias para evitar modificar los datos originales
        train_data = np.array(x_train, copy=True)
        train_label = np.array(y_train, copy=True)
        current_unlabeled = np.array(unlabeled_data, copy=True)
        
        # Inicialización de pesos: mayor peso a muestras originales (etiquetadas)
        sample_weights = np.ones(len(train_label)) * 2.0
        
        for i in range(1, train_epochs):
            if len(current_unlabeled) == 0:
                print("No more unlabeled data left")
                break
                
            # Crear y entrenar nuevo modelo en cada iteración
            model = model_func()
            model.fit(train_data, train_label, validation_data=validation_data, sample_weight=sample_weights, verbose=verbose)
            
            # Obtener predicciones y confianza
            y_pred = model.predict_proba(current_unlabeled)
            y_class = np.argmax(y_pred, axis=1)
            y_value = np.max(y_pred, axis=1)
            
            # Filtrar predicciones con alta confianza
            high_confidence = y_value > thresh
            
            if np.any(high_confidence):
                # Obtener muestras con alta confianza
                new_data = current_unlabeled[high_confidence]
                new_labels = y_class[high_confidence]
                new_probs = y_value[high_confidence]
                
                # Augmentar conjunto de entrenamiento
                train_data = np.vstack([train_data, new_data])
                train_label = np.append(train_label, new_labels)
                sample_weights = np.append(sample_weights, new_probs)
                
                # Eliminar ejemplos usados del conjunto no etiquetado
                current_unlabeled = current_unlabeled[~high_confidence]
                
                print(f"Epoch {i}: Added {len(new_data)} samples, {len(current_unlabeled)} remaining")
            else:
                print(f"Epoch {i}: No samples added")
        
        # Entrenamiento del modelo final con todos los datos pseudo-etiquetados
        final_model = model_func()
        history = final_model.fit(train_data, train_label, validation_data=validation_data, sample_weight=sample_weights, epochs=train_epochs)
        print(f"Final model trained with {len(train_data)} samples")
        final_model.plot(history)
        return final_model
    
    def predict(self, X):
        """
        Realiza predicciones de clase para las muestras en X.
        
        Args:
            X (numpy.ndarray): Datos de entrada
            
        Returns:
            numpy.ndarray: Etiquetas de clase predichas
        """
        return np.argmax(self.model.predict(X), axis=1)
    
    def predict_proba(self, X):
        """
        Calcula probabilidades de cada clase para las muestras en X.
        
        Args:
            X (numpy.ndarray): Datos de entrada
            
        Returns:
            numpy.ndarray: Probabilidades para cada clase
        """
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        Evalúa el rendimiento del modelo.
        
        Args:
            X (numpy.ndarray): Datos de entrada
            y (numpy.ndarray): Etiquetas verdaderas
            
        Returns:
            float: Precisión (accuracy) del modelo
        """
        _, acc = self.model.evaluate(X, y)
        return acc
    
    def plot(self, history):
        """
        Visualiza las métricas de entrenamiento (pérdida y precisión).
        
        Args:
            history: Objeto history devuelto por el método fit
        """
        plt.figure(figsize=(12, 5))
    
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Pérdida de Validación')
        plt.title('Pérdida del Modelo')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend(loc='upper right')
    
        # Gráfico de precisión
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
        plt.title('Precisión del Modelo')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend(loc='lower right')
    
        plt.tight_layout()
        plt.show()
    
    def __call__(self, X):
        """
        Permite llamar al objeto como una función para realizar predicciones.
        
        Args:
            X (numpy.ndarray): Datos de entrada
            
        Returns:
            numpy.ndarray: Predicciones del modelo
        """
        return self.model.predict(X)
    
    def summary(self):
        """
        Devuelve un resumen de la arquitectura del modelo.
        
        Returns:
            str: Resumen de las capas y parámetros del modelo
        """
        return self.model.summary()
    
    def __del__(self):
        """
        Libera recursos al eliminar la instancia del modelo.
        """
        del self.model
        backend.clear_session()  # Liberar memoria en GPU