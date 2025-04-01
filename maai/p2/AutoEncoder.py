from tensorflow.keras import layers, models, optimizers, backend
import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'JAX'


class TwoStepAutoEncoder():

    def __init__(self, input_shape, learning_rate=0.001):
        
        self.input_shape = input_shape
        
        self.encoder = models.Sequential([
            #layers.Reshape((32, 32, 3), input_shape=input_shape),
			layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=self.input_shape),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
			
			layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
		
			layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
			
			#layers.Flatten(),
			#layers.Dense(256, activation='relu'),
			#layers.Dropout(0.5),
			#layers.Dense(100, activation="softmax")
		])
        
        self.decoder = models.Sequential([
            # First fully connected layer (Dense) from the encoder
            #layers.Dense(128 * 32 * 32, activation='relu', input_shape=(100,)),
            #layers.Reshape((4, 4, 128)),  # Reshape to the appropriate size for the first transposed conv layer
            
            # First transposed convolution layer (upsampling)
            layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),  # Upsample to double the size
            
            # Second transposed convolution layer (upsampling)
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),  # Upsample again
            
            # Third transposed convolution layer (upsampling)
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),  # Upsample again
            
            # Final convolution layer to get the output with 3 channels (RGB image)
            layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')  # Output 32x32x3
        ])
        
        self.autoencoder = models.Sequential([self.encoder, self.decoder])

        self.optimicer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )


        self.autoencoder.compile(optimizer=self.optimicer, loss='mse')
    
    def fit(self, X, y=None, sample_weight=None, batch_size=60_000, epochs=100):
        # TODO: entrena el modelo. Escoge el tamaño de batch y el número de epochs que quieras
        
        self.autoencoder.fit(X, X, 
                             batch_size=batch_size, 
                             epochs=epochs, 
                             sample_weight=sample_weight)

    def get_encoded_data(self, X):
        # TODO: devuelve la salida del encoder (code)
        return self.encoder.predict(X)


    def __call__(self, X):
        return self.autoencoder.predict(X)
        
    def __del__(self):
        # elimina todos los modelos que hayas creado
        backend.clear_session() # Necesario para liberar la memoria en GPU

#--------------------------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------#

class TwoStepClassifier:

    def __init__(self):
        # TODO : define el modelo y compílalo
        
        self.input_shape = (4 , 4 , 128)
        
        self.classifier = models.Sequential()
        self.classifier.add(layers.InputLayer(shape=self.input_shape))
        self.classifier.add(layers.Flatten())
        self.classifier.add(layers.Dense(128, activation='relu'))
        self.classifier.add(layers.Dense(100, activation='sigmoid'))

        self.optimicer = optimizers.AdamW(
            learning_rate=0.001,
            clipnorm=1,
        )

        self.classifier.compile(optimizer=self.optimicer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    def fit(self, X, y, sample_weight=None, batch_size=60_000, epochs=350):
        # TODO: entrena el modelo. Escoge el tamaño de batch y el número de epochs que quieras
        self.classifier.fit(X, y, 
                             batch_size=batch_size, 
                             epochs=epochs, 
                             sample_weight=sample_weight)

    def predict(self, X):
        # TODO: devuelve la clase ganadora

        return np.argmax(self.predict_proba(X)) + 1
    
    def predict_proba(self, X):
        
        return self.classifier.predict(X)
    
    def score(self, X, y):
        
        return self.classifier.evaluate(X, y)[1]

    def __del__(self):
        # elimina todos los modelos que hayas creado
        backend.clear_session() # Necesario para liberar la memoria en GPU


#--------------------------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------#


def TwoStepTraining(autoencoder, classifier, x_train, y_train, unlabeled_train, batch_size_autoencoder=1024, epochs_autoencoder=100, batch_size_classifier=1024, epochs_classifier=100):

    all_x = np.vstack((x_train, unlabeled_train))
    autoencoder.fit(all_x, batch_size=batch_size_autoencoder, epochs=epochs_autoencoder)
    x_coded = autoencoder.get_encoded_data(x_train)
    classifier.fit(x_coded, y_train, batch_size=batch_size_classifier, epochs=epochs_classifier)


#--------------------------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------#


class OneStepAutoencoder:

    def __init__(self, input_shape,learning_rate=0.001):
        # TODO : define el modelo y compílalo
        
        self.input_shape = input_shape
        self.num_classes = 100
        
        input_layer = layers.Input(shape=self.input_shape)

        '''
        self.encoder = models.Sequential([
			layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=self.input_shape),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
			
			layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
		
			layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
			layers.BatchNormalization(),
			layers.MaxPooling2D((2, 2)),
		])
        
        self.decoder = models.Sequential([
            # First transposed convolution layer (upsampling)
            layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),  # Upsample to double the size
            
            # Second transposed convolution layer (upsampling)
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),  # Upsample again
            
            # Third transposed convolution layer (upsampling)
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling2D((2, 2)),  # Upsample again
            
            # Final convolution layer to get the output with 3 channels (RGB image)
            layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')  # Output 32x32x3
        ])
        '''

        conv = layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=self.input_shape)(input_layer)
        conv = layers.BatchNormalization()(conv)
        conv = layers.MaxPooling2D((2, 2))(conv)
        
        conv = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.MaxPooling2D((2, 2))(conv)
    
        conv = layers.Conv2D(128, (3, 3), activation='relu', padding="same")(conv)
        conv = layers.BatchNormalization()(conv)
        code = layers.MaxPooling2D((2, 2))(conv)

        # First transposed convolution layer (upsampling)
        decoded = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(code)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.UpSampling2D((2, 2))(decoded)  # Upsample to double the size
        
        # Second transposed convolution layer (upsampling)
        decoded = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.UpSampling2D((2, 2))(decoded)  # Upsample again
        
        # Third transposed convolution layer (upsampling)
        decoded = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.UpSampling2D((2, 2))(decoded)  # Upsample again
        
        # Final convolution layer to get the output with 3 channels (RGB image)
        decoded = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='decoder')(decoded)  # Output 32x32x3

        

        
        #self.autoencoder = models.Sequential([self.encoder, self.decoder])

        #code = self.encoder(input_layer)
        #decoded = self.decoder()(code)
        #decoded = layers.Reshape(self.input_shape)(decoded)

        classifier = layers.Flatten()(code)
        #classifier = layers.Dense(100, activation='relu')(classifier)
        #classifier = layers.Dense(128, activation='relu')(classifier)
        classifier_output = layers.Dense(self.num_classes, activation='softmax',name='classifier')(classifier)

        
        #self.classifier = models.Model(code, classifier_output)
        #self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        
        
        # Combined model with two outputs: one for autoencoder (reconstruction) and one for classifier (classification)
        self.model = models.Model(input_layer, 
                                  [classifier_output, decoded])
                                  #decoded)
                                  #classifier_output)

        self.optimicer = optimizers.AdamW(
            learning_rate=learning_rate,
            clipnorm=1,
        )
        
        self.model.compile(optimizer=self.optimicer,
                           #loss=['categorical_crossentropy', 'mse'],
                           loss = {'decoder': 'mse', 'classifier': 'categorical_crossentropy'},
                           #loss='mse',
                           #loss='categorical_crossentropy',
                # >>>>>>>  #loss_weights=[.8, 1.2],  # Adjust loss weights if needed
                           metrics={'decoder': [], 'classifier': ['accuracy']},
                           #metrics=['accuracy'])
                          )
    
    def fit(self, X, y, unlabeled_train, batch_size,  epochs):
        # TODO: entrena el modelo. Escoge el tamaño de batch y el número de epochs que quieras, y define bien el sample_weight

        X = np.array(X, dtype=np.float32)
        unlabeled_train = np.array(unlabeled_train, dtype=np.float32)

        all_x = np.vstack((X, unlabeled_train))
        y_zeros = np.zeros((unlabeled_train.shape[0],y.shape[1]))
        all_y = np.vstack((y,y_zeros))
        weight_autoencoder = np.ones(len(all_x))
        #weight_autoencoder = None DA ERROR
        weight_classifier = np.array([1]*len(y) + [0]*len(y_zeros))
        
        h = self.model.fit(all_x,
                       #[all_y, all_x],
                       {'decoder': all_x, 'classifier': all_y},
                       #all_x,
                       #all_y,
                       #sample_weight={'decoder': weight_autoencoder, 'classifier': weight_classifier},  # da un key error
                       #sample_weight=[weight_autoencoder, weight_classifier],  # You should provide one `sample_weight` array per output in `y`. The two structures did not match: - y: {'decoder': array([[[[0.8509804 , 0.8156863 , 0.6313726 ],
                       #sample_weight=weight_classifier,
                       epochs=epochs, 
                       batch_size=batch_size, 
                       verbose=1)
        return h

    def predict_class(self, X):
        # TODO: devuelve la clase ganadora del clasificador
        class_label, _ = self.model.predict(X)
        return class_label.argmax(axis=1)
    
    def predict_image(self, X):
        # TODO: devuelve la probabilidad del clasificador
        _, image = self.model.predict(X)
        return image
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def __call__(self, X):
        return self.model.predict(X)

    def __del__(self):
        # elimina todos los modelos que hayas creado
        backend.clear_session() # Necesario para liberar la memoria en GPU


#--------------------------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------#

def OneStepTraining(model, x_train, y_train, unlabeled_train, batch_size=60_000, epochs = 1000):
    h = model.fit(x_train, y_train, unlabeled_train, batch_size=batch_size, epochs = epochs)
    return h

