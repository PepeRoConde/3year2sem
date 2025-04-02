from tensorflow.keras import layers, optimizers, models, Model, regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
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
    def __init__(self, input_shape, lambda_param = 0.5, temperature = 0.5, learning_rate=0.0005, l2_lambda=0.01):
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
        input_layer = layers.Input(batch_shape=(None, 32, 32,3))  # Tamaño de imagen
        conv = layers.Conv2D(32, (3, 3), activation='relu', padding="same", kernel_regularizer=regularizers.l2(l2_lambda))(input_layer)
        conv = layers.BatchNormalization()(conv)
        #conv = layers.MaxPooling2D((2, 2))(conv)
        
        conv = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.MaxPooling2D((2, 2))(conv)

        conv = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(conv)
        conv = layers.BatchNormalization()(conv)
        #conv = layers.MaxPooling2D((2, 2))(conv)
    
        conv = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(conv)
        conv = layers.BatchNormalization()(conv)
        code = layers.MaxPooling2D((2, 2))(conv)
        
        flatten_layer = layers.Flatten()(code)
        
        # Capa de clustering
        cluster_layer = layers.Dense(self.output_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(flatten_layer)
        cluster_layer = layers.Dense(self.output_dim, activation='softmax', kernel_regularizer=regularizers.l2(l2_lambda))(cluster_layer)
        
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