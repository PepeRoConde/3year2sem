from tensorflow.keras import layers, optimizers, models, Model, regularizers, losses
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

class ContrastiveLoss():
    def __init__(self, temperature=0.5):
        self.temperature = temperature
        
    def __call__(self, M):
        logits = M / self.temperature # temperatura
        
        batch_size = tf.shape(M)[0]
        I = tf.eye(batch_size)  # matriz identidad

        loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(I, logits))

        
        return loss

class ClusteringLoss():
    def __init__(self):
        pass
        
    def __call__(self, cX_1comp, cX_2comp):

        # Encourage peaky distributions
        entropy_1 = -tf.reduce_mean(tf.reduce_sum(cX_1comp * tf.math.log(cX_1comp + 1e-8), axis=1))
        entropy_2 = -tf.reduce_mean(tf.reduce_sum(cX_2comp * tf.math.log(cX_2comp + 1e-8), axis=1))
        
        # Ensure consistency between views
        consistency = tf.reduce_mean(tf.reduce_sum(tf.square(cX_1comp - cX_2comp), axis=1))
        
        return entropy_1 + entropy_2 + consistency

class ContrastiveModel():
    def __init__(self, input_shape, lambda_param = 0.5, temperature = 0.5, learning_rate=0.0005, l2_lambda=0.01, dropout_prob=0.01):
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
                layers.Resizing(38, 38), # para CIFAR, para MNIST usar 40 en lugar de 48
                layers.RandomCrop(32, 32), # para CIFAR, para MNIST usar 28 en lugar de 32
            ])
            
        # Definir modelo convolucional
        input_layer = layers.Input(shape=self.input_shape)


    
        # Encoder - First convolutional block
        x = layers.Conv2D(96, (3, 3), activation='relu', padding="same", 
                          kernel_regularizer=regularizers.l2(l2_lambda))(input_layer)
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
        if isinstance(data, tuple):
            X = data[0]
        else:
            X = data
            
        batch_size = tf.shape(X)[0]
        
        # Aplicar las dos transformaciones de data augmentation
        augX_1 = self.data_augmentation_1(X)
        augX_2 = self.data_augmentation_2(X)
        
        with tf.GradientTape() as tape:
            
            #augX_1comp = self.encoder(augX_1) # representaciones del encoder
            #augX_2comp = self.encoder(augX_2)

            augX_1comp = tf.nn.l2_normalize(self.encoder(augX_1), axis=1)
            augX_2comp = tf.nn.l2_normalize(self.encoder(augX_2), axis=1)

            cX_1comp = self.cluster(augX_1comp) # salidas del clustering
            cX_2comp = self.cluster(augX_2comp)
            
            M = tf.matmul(augX_1comp, augX_2comp, transpose_b=True) # matriz de similitud M
            
            loss_M = self.contrastive_loss(M) 
            loss_C = self.clustering_loss(cX_1comp, cX_2comp)
            total_loss = loss_M + self.lambda_param * loss_C
            
        # Calcular gradientes y actualizar pesos
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables + self.cluster.trainable_variables)

        grad_norm = tf.linalg.global_norm(gradients)
        
        self.optimicer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.cluster.trainable_variables))
        
        return {"loss": total_loss, "contrastive_loss": loss_M, "clustering_loss": loss_C}
    
    def mini_batches(self, X, batch_size):
        for start in range(0, X.shape[0], batch_size):
            # Yield each mini-batch
            end = min(start + batch_size, X.shape[0])
            yield X[start:end]
    
    def train(self, dataset, epochs=10, batch_size=128, temperature=0.5):
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
        return self.encoder(X)
    
    def plot_training_history(self, figsize=(12, 6)):
        """
        Visualiza el historial de pérdidas durante el entrenamiento.
        """
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(self.loss_history['total_loss']) + 1)
        
        plt.figure(figsize=figsize)
        
        # Gráfico de pérdida total
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.loss_history['total_loss'], 'b-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Gráfico comparativo de pérdidas
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.loss_history['contrastive_loss'], 'r-', label='Contrastive Loss')
        plt.plot(epochs, self.loss_history['clustering_loss'], 'g-', label='Clustering Loss')
        plt.title('Component Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def __call__(self, X):
        features = self.encoder(X)
        clusters = self.cluster(features)
        return features, clusters

    def plot_similarity_matrix(self, X, n_samples=10):
        """
        Visualiza la matriz de similitud para un conjunto de muestras.
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
        plt.title('Similarity Matrix')
        plt.xlabel('Augmentation 2')
        plt.ylabel('Augmentation 1')
        plt.tight_layout()
        plt.show()
        
        return M


#---

class SupervisedLoss():
    def __init__(self):
        self.loss_fn = losses.SparseCategoricalCrossentropy()
        
    def __call__(self, y_true, y_pred, sample_weights=None):
        return self.loss_fn(y_true, y_pred, sample_weight=sample_weights)

class SemiSupervisedContrastiveModel():
    def __init__(self, input_shape, lambda_param=0.5, lambda_supervised=1.0, temperature=0.5, 
                 learning_rate=0.0005, l2_lambda=0.01, dropout_prob=0.01):
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
                layers.Resizing(38, 38), # para CIFAR, para MNIST usar 40 en lugar de 48
                layers.RandomCrop(32, 32), # para CIFAR, para MNIST usar 28 en lugar de 32
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
        Train step method that handles both labeled and unlabeled data.
        
        data: tuple of (X, y, is_labeled) where:
              - X is the input data
              - y is the labels (can be None for unlabeled data)
              - is_labeled is a boolean mask indicating which samples are labeled
        """
        X, y, is_labeled = data
        batch_size = tf.shape(X)[0]
        
        # Aplicar las dos transformaciones de data augmentation
        augX_1 = self.data_augmentation_1(X)
        augX_2 = self.data_augmentation_2(X)
        
        with tf.GradientTape() as tape:
            # Get normalized embeddings
            augX_1comp = tf.nn.l2_normalize(self.encoder(augX_1), axis=1)
            augX_2comp = tf.nn.l2_normalize(self.encoder(augX_2), axis=1)

            # Get cluster assignments
            cX_1comp = self.cluster(augX_1comp)
            cX_2comp = self.cluster(augX_2comp)
            
            # Compute similarity matrix
            M = tf.matmul(augX_1comp, augX_2comp, transpose_b=True)
            
            # Compute unsupervised losses
            loss_M = self.contrastive_loss(M)
            loss_C = self.clustering_loss(cX_1comp, cX_2comp)
            
            # Initialize supervised loss to 0
            loss_S = 0.0
            
            # Apply supervised loss only if there are labeled samples
            if tf.reduce_sum(tf.cast(is_labeled, tf.float32)) > 0:
                # Create sample weights based on which samples are labeled
                sample_weights = tf.cast(is_labeled, tf.float32)
                
                # Use the average of the two views for prediction
                avg_pred = (cX_1comp + cX_2comp) / 2.0
                
                # Compute supervised loss with sample weights
                loss_S = self.supervised_loss(y, avg_pred, sample_weights)
            
            # Combined loss
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
        Prepare a combined batch of labeled and unlabeled data.
        
        Returns:
            tuple: (X, y, is_labeled) where is_labeled is a boolean mask
        """
        if X_labeled is not None and y_labeled is not None:
            # Combine unlabeled and labeled data
            X_combined = np.concatenate([X_unlabeled, X_labeled], axis=0)
            
            # Create labels for all data (set dummy labels for unlabeled data)
            y_combined = np.zeros(X_combined.shape[0], dtype=np.int32)
            y_combined[X_unlabeled.shape[0]:] = y_labeled.squeeze()
            
            # Create mask indicating which samples are labeled
            is_labeled = np.zeros(X_combined.shape[0], dtype=bool)
            is_labeled[X_unlabeled.shape[0]:] = True
            
            return X_combined, y_combined, is_labeled
        else:
            # Only unlabeled data
            dummy_labels = np.zeros(X_unlabeled.shape[0], dtype=np.int32)
            is_labeled = np.zeros(X_unlabeled.shape[0], dtype=bool)
            return X_unlabeled, dummy_labels, is_labeled
    
    def mini_batches(self, X_unlabeled, X_labeled=None, y_labeled=None, batch_size=128):
        """
        Generate mini-batches from unlabeled and labeled data.
        """
        n_unlabeled = X_unlabeled.shape[0]
        n_labeled = 0 if X_labeled is None else X_labeled.shape[0]
        
        # Shuffle indices for unlabeled and labeled data
        unlabeled_indices = np.random.permutation(n_unlabeled)
        labeled_indices = np.random.permutation(n_labeled) if n_labeled > 0 else None
        
        # Calculate how many labeled samples to include in each batch
        # Use a ratio to maintain a balance between labeled and unlabeled
        if n_labeled > 0:
            # Aim for roughly 25% labeled samples in each batch, if available
            labeled_per_batch = min(n_labeled, max(1, int(batch_size * 0.25)))
            unlabeled_per_batch = batch_size - labeled_per_batch
        else:
            labeled_per_batch = 0
            unlabeled_per_batch = batch_size
        
        # Generate batches
        for i in range(0, n_unlabeled, unlabeled_per_batch):
            # Get unlabeled indices for this batch
            end_idx = min(i + unlabeled_per_batch, n_unlabeled)
            batch_unlabeled_indices = unlabeled_indices[i:end_idx]
            batch_X_unlabeled = X_unlabeled[batch_unlabeled_indices]
            
            # If we have labeled data, include some in the batch
            if n_labeled > 0:
                # Cycle through labeled data if needed
                start_labeled = (i // unlabeled_per_batch * labeled_per_batch) % n_labeled
                end_labeled = min(start_labeled + labeled_per_batch, n_labeled)
                
                # If we wrap around, take some from the beginning
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
                
                # Prepare the combined batch
                batch_data = self.prepare_data_batch(batch_X_unlabeled, batch_X_labeled, batch_y_labeled)
            else:
                batch_data = self.prepare_data_batch(batch_X_unlabeled)
                
            yield batch_data
    
    def train(self, X_unlabeled, X_labeled=None, y_labeled=None, epochs=10, batch_size=128, temperature=0.5):
        """
        Train the model with both unlabeled and labeled data.
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
        """
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(self.loss_history['total_loss']) + 1)
        
        plt.figure(figsize=figsize)
        
        # Gráfico de pérdida total
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.loss_history['total_loss'], 'b-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Gráfico comparativo de pérdidas sin supervisión
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.loss_history['contrastive_loss'], 'r-', label='Contrastive Loss')
        plt.plot(epochs, self.loss_history['clustering_loss'], 'g-', label='Clustering Loss')
        plt.title('Unsupervised Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Gráfico de pérdida supervisada
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.loss_history['supervised_loss'], 'c-', label='Supervised Loss')
        plt.title('Supervised Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def __call__(self, X):
        features = self.encoder(X)
        clusters = self.cluster(features)
        return features, clusters

    def extract_features(self, X):
        """
        Extract features from the encoder network.
        """
        return self.encoder(X).numpy()
    
    def predict(self, X):
        """
        Predict cluster assignments for input data.
        """
        features = self.encoder(X)
        clusters = self.cluster(features)
        return tf.argmax(clusters, axis=1).numpy()

    def score(self, x, y):
        return accuracy_score(self.predict(x), y)
        
    
    def plot_similarity_matrix(self, X, n_samples=10):
        """
        Visualiza la matriz de similitud para un conjunto de muestras.
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
        plt.title('Similarity Matrix')
        plt.xlabel('Augmentation 2')
        plt.ylabel('Augmentation 1')
        plt.tight_layout()
        plt.show()
        
        return M