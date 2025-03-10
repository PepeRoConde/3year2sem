from utils.mnist_reader import load_mnist
#import p1utils
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from  sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, MeanShift, OPTICS, estimate_bandwidth, AffinityPropagation, SpectralClustering
from sklearn.decomposition import PCA
import sklearn.manifold
from  sklearn.model_selection import GridSearchCV
import numpy as np
from time import time
import umap
import torch
from torch import nn, tensor, optim
from torch.utils.data import DataLoader
import pandas as pd

X_train, y_train = load_mnist('data/fashion', kind='train')
X_test, y_test = load_mnist('data/fashion', kind='t10k')
X_train, X_dev = X_train[:50000], X_train[50000:]
y_train, y_dev = y_train[:50000], y_train[50000:]

print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_dev: {X_dev.shape}, y_dev: {y_dev.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')


X_train_5D = umap.UMAP( n_components=5).fit_transform(X_train)
# vemos como queda el primer elemento
print(X_train_5D[0])
plt.imshow(np.resize(X_train[0],(28,28)));


class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    def predict(self,X):
      return self.labels_.astype(int)

class DBSCANWrapper(DBSCAN):
    def predict(self,X):
      return self.labels_.astype(int)

class HDBSCANWrapper(HDBSCAN):
    def predict(self,X):
      return self.labels_.astype(int)

class OPTICSWrapper(OPTICS):
    def predict(self,X):
        return self.labels_.astype(int)

class SpectralClusteringWrapper(SpectralClustering):
    def predict(self,X):
        return self.labels_.astype(int)

clustering_methods = {
    'KMeans': KMeans(),
    'DBSCAN': DBSCANWrapper(),
    'HDBSCAN': HDBSCANWrapper(),
    'MeanShift': MeanShift(),
    'OPTICS': OPTICSWrapper(),
    'AffinityPropagation': AffinityPropagation()
    'SpectralClustering': SpectralClusteringWrapper(),
    'Agglomerative': AgglomerativeClusteringWrapper()
    
}



param_grids = {
    'KMeans': {
        'n_clusters': [9],
        'init': ['k-means++', 'random'],
        #'max_iter': [300, 500]
    },
    'DBSCAN': {
        'eps': [10],
        'min_samples': [5, 50]
    },

    'HDBSCAN': {
        'min_cluster_size': [50, 100, 500], 
        'max_cluster_size': [4500, 5500, 6500], 
    },

    'MeanShift': {
        'bandwidth': [3.7,4.5]
    },

    'OPTICS': {
        'min_samples': [50, 100]
    },

    'AffinityPropagation': {
        'damping': [0.5]
    },
    
    'SpectralClustering': {
        'n_clusters': [9]
    },
    
    
    'AgglomerativeClustering': {
        'n_clusters': [ 9],
        'linkage': ['ward'] #, 'complete', 'average']
    }

}

# Function to evaluate each model and its hyperparameters
def try_clustering(X_train, y_train,  clustering_methods, param_grids):
    results = {}
    best_score = -float('inf')
    for name, model in clustering_methods.items():
        start = time()
        
        grid_search = GridSearchCV(model, param_grids[name], cv=2, scoring='adjusted_mutual_info_score', n_jobs=4)
        grid_search.fit(X_train, y_train)

        
        score = grid_search.best_score_
        if score > best_score:
            model = grid_search.best_estimator_
        params = grid_search.best_params_
        end = time()
        print(f'Modelo: {name}. \nAdjMutualInformation: {score}. \nParametros: {params}. \nTiempo: {round(end-start,2)}s (o {round(end-start,2)/60}min) .\n\n')
        
        results[name] = [params, score, round(end-start,2)]

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Mejores parámetros', 'Adjusted Mutual Info Score', 'Seconds'])
    
    # Reset index to make the model name a column
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'Algoritmo'}, inplace=True)
    
    # Sort the DataFrame by 'Adjusted Mutual Info Score' in descending order
    results_df = results_df.sort_values(by='Adjusted Mutual Info Score', ascending=False)


    return results_df, model


results, model = try_clustering(X_train_5D, y_train, clustering_methods, param_grids)
print(results)
print(model)
print('hola hemos llegado al final :3')