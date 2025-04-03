from tensorflow.keras import layers, models, optimizers, backend, regularizers, callbacks
from sklearn.metrics import accuracy_score
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'JAX'

class TwoStepAutoEncoder():

    def __init__(self, input_shape, learning_rate=0.001, l2_lambda=0.01, dropout_prob=0.1):
        
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
        self.autoencoder.fit(X, X, 
                             validation_data=validation_data,
                             batch_size=batch_size, 
                             epochs=epochs, 
                             sample_weight=sample_weight)

    def get_encoded_data(self, X):
        return self.encoder.predict(X)


    def __call__(self, X):
        return self.autoencoder.predict(X)
        
    def __del__(self):
        backend.clear_session() # Necesario para liberar la memoria en GPU

#--------------------------------------------------------------------------------------------#

class TwoStepClassifier:

    def __init__(self,dropout_prob=0.1,l2_lambda=0.0005,learning_rate=0.01):

        self.output_dim=100

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
        self.classifier.fit(X, y, 
                            validation_data=validation_data,
                             batch_size=batch_size, 
                             epochs=epochs, 
                             sample_weight=sample_weight)

    def predict(self, X):
        return np.argmax(self.predict_proba(X)) + 1
    
    def predict_proba(self, X):
        return self.classifier.predict(X)
    
    def score(self, X, y):
        return self.classifier.evaluate(X, y)[1]

    def __del__(self):
        backend.clear_session() # Necesario para liberar la memoria en GPU


#--------------------------------------------------------------------------------------------#

def TwoStepTraining(autoencoder, classifier, x_train, y_train, unlabeled_train, validation_data=None, batch_size_autoencoder=1024, epochs_autoencoder=100, batch_size_classifier=1024, epochs_classifier=100):

    all_x = np.vstack((x_train, unlabeled_train))
    autoencoder.fit(all_x, batch_size=batch_size_autoencoder, epochs=epochs_autoencoder, validation_data=(validation_data[0],validation_data[0]))
    x_coded = autoencoder.get_encoded_data(x_train)
    x_val, y_val = validation_data
    x_val_coded = autoencoder.get_encoded_data(x_val)
    classifier.fit(x_coded, y_train, batch_size=batch_size_classifier, epochs=epochs_classifier, validation_data=(x_val_coded,y_val))


#--------------------------------------------------------------------------------------------#

class OneStepAutoencoder:

    def __init__(self, input_shape,learning_rate=0.001, decoder_extra_loss_weight = 0.3, l2_lambda=0.001, dropout_prob=0.05):
       
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

        # classifier

        classifier = layers.Flatten()(encoded)
        classifier = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(classifier)
        classifier = layers.BatchNormalization()(classifier)
        classifier = layers.Dropout(dropout_prob)(classifier)
        
        classifier = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(classifier)
        classifier = layers.BatchNormalization()(classifier)
        classifier = layers.Dropout(dropout_prob/2)(classifier)
        
        classifier_output = layers.Dense(self.output_dim, activation="softmax", name='classifier')(classifier)

        #--------------------------------------------------------------------------------------------#
        # model
        self.model = models.Model(input_layer, 
                                  {'decoder': decoded, 'classifier': classifier_output})

        self.optimicer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )
        
        self.model.compile(optimizer=self.optimicer,
                           loss = {'decoder': 'mse', 'classifier': 'categorical_crossentropy'},
                           loss_weights={'decoder': 1.0 + decoder_extra_loss_weight, 'classifier': 1.0 - decoder_extra_loss_weight},  # Adjust loss weights if needed
                           metrics={'decoder': [], 'classifier': ['accuracy']})
    
    def fit(self, X, y, unlabeled_train, batch_size,  epochs, patience, validation_data=None):
        # a float32
        X = np.array(X, dtype=np.float32)
        unlabeled_train = np.array(unlabeled_train, dtype=np.float32)

        # lo que vamos a usar
        all_x = np.vstack((X, unlabeled_train))
        y_zeros = np.zeros((unlabeled_train.shape[0],y.shape[1]))
        all_y = np.vstack((y,y_zeros))

        # coeficientes para los ejemplos
        weight_autoencoder = np.ones((all_x.shape[0],all_x.shape[1],all_x.shape[2]))
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
        class_label, _ = self.model.predict(X)
        return class_label.argmax(axis=1)
    
    def predict_image(self, X):
        _, image = self.model.predict(X)
        return image
    
    def score(self, X, y):
        y_pred = np.argmax(self.model.predict(X)['classifier'],axis=1)
        return accuracy_score(y, y_pred)

    def __call__(self, X):
        return self.model.predict(X)

    def __del__(self):
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

#--------------------------------------------------------------------------------------------#

def OneStepTraining(model, x_train, y_train, unlabeled_train, batch_size=60_000, epochs = 1000, patience=5):
    h = model.fit(x_train, y_train, unlabeled_train, batch_size=batch_size, epochs = epochs, patience=patience)
    return h