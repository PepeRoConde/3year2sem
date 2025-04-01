import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

class DatasetProcess:
    def load():
    	return keras.datasets.cifar100.load_data()
        
    def separate_data(train, test):
        (x_train, y_train) = train
        (x_test, y_test) = test
        
        x_train_no_labeled = x_train[:40000]  # Sin etiquetas
        x_train_labeled = x_train[40000:]     # Con etiquetas
        y_train_labeled = y_train[40000:]     # Etiquetas correspondientes
        
        return (x_train_no_labeled, x_train_labeled, y_train_labeled),(x_test, y_test)

    def alt():
        
        labeled_data = 0.01 # Vamos a usar el etiquetado de sólo el 1% de los datos
        np.random.seed(42)
        
        (x_train, y_train), (x_test, y_test), = keras.datasets.cifar100.load_data()
        
        indexes = np.arange(len(x_train))
        np.random.shuffle(indexes)
        ntrain_data = int(labeled_data*len(x_train))
        unlabeled_train = x_train[indexes[ntrain_data:]] /255
        x_train = x_train[indexes[:ntrain_data]] /255
        x_test = x_test /255
        y_train = y_train[indexes[:ntrain_data]]
        
        #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2],3)) 
        #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2],3)) 
        #unlabeled_train = np.reshape(unlabeled_train, (unlabeled_train.shape[0], unlabeled_train.shape[1]*unlabeled_train.shape[2],3))
        
        one_hot_train = np.zeros((y_train.size, len(np.unique(y_train))), dtype=int)
        one_hot_train[np.arange(y_train.size), y_train ] = 1
        
        one_hot_test = np.zeros((y_test.size, len(np.unique(y_test))), dtype=int)
        one_hot_test[np.arange(y_test.size), y_test ] = 1
        
        return unlabeled_train, x_train, y_train, x_test, y_test, one_hot_train, one_hot_test


def reconstruction_plot(autoencoder, x_test):
    index = np.random.randint(len(x_test))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(x_test[index].reshape(32, 32, 3))
    axes[0].set_title("Original Image")
    axes[0].axis('off') 
    
    # Get the reconstructed image from the autoencoder
    reconstructed_image = autoencoder(x_test[index].reshape(1, 32, 32, 3)).reshape(32, 32, 3)
    
    # Plot the reconstructed image on the right
    axes[1].imshow(reconstructed_image)
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')
    
    # Display the plot
    plt.show()