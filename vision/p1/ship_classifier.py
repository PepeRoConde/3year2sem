import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import time
import os
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import f1_score, confusion_matrix

class ShipClassifier:
    def __init__(self, pretrained=True, docked=True, mlp_head=True, device='mps'):

        self.pretrained = pretrained
        self.docked = docked
        self.mlp_head = mlp_head
        self.device = device
        
        self.model = None
        self.create_model(pretrained, docked)
        
    def create_model(self, pretrained=True, docked=True, mlp_head=True):
        """
        Crea una EfficientNet para la clasificacion de barcos
        
        Args:
            pretrained (bool): usar preentrenamiento o no
        """

        if pretrained:
            model = models.efficientnet_b3(weights='DEFAULT')
        else: 
            model = models.efficientnet_b3()

        if docked:
             n_outputs = 3
        else:
             n_outputs = 2
        
      
        for param in model.features.parameters():
            param.requires_grad = True
            
        
        if mlp_head:
            model.classifier[1] = nn.Linear(in_features=1536, out_features=64)
            model.classifier.append(nn.Linear(64, 32))
            model.classifier.append(nn.ReLU())
            model.classifier.append(nn.Linear(32, n_outputs))
        else:
            model.classifier[1] = nn.Linear(in_features=1536, out_features=n_outputs)
            
        model.classifier.append(nn.Softmax(dim=1))
        
        self.model = model

        return model
        
    def forward(self, x):
        x = self.model(x)
        return x

    def train_model(self, train_loader, optimizer=None, criterion=None, num_epochs=3, patience=5, lr_patience=2, l2_lambda=1e-2):
        """
        Entrena el modelo de clasificacion binaria de barcos
    
        Args:
            train_loader: DataLoader para los datos de entrenamiento
            optimizer: PyTorch optimizer, Adam por defecto 
            criterion: Loss function, CrossEntropyLoss por defecto
            num_epochs (int): Numero de epochs de entrenamiento
            patience (int): Paciencia del Early Stopping
            
        Returns:
            model: Modelo entrenado
            history: Diccionario con las metricas del modelo
        """
        if self.model is None:
            return "Please create a model (ShipClassifier.create_model()) before training"
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Add learning rate scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',      # Reduce LR when metric stops improving
            factor=0.5,      # Reduce learning rate by half
            patience=lr_patience,      # Wait 2 epochs before reducing LR
            verbose=True     # Print when LR changes
        )
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        print(f"Using device: {self.device}")
    
        self.model.to(self.device)
        scaler = GradScaler()
     
        # Update history to track learning rate
        history = {
            'train_loss': [],         # Perdida por Batch
            'train_acc': [],          # Accuracy por Batch
            'epoch_train_loss': [],   # Avg loss por Epoch
            'epoch_train_acc': [],    # Avg accuracy por Epoch
            'learning_rates': [],     # Track learning rates
            'training_time': 0.0      # Total training time
        }
    
        best_loss = np.inf  
        epochs_no_improve = 0       
    
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Training mode
            self.model.train()
            current_loss = 0.0
            current_corrects = 0
            total_samples = 0
            
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Reiniciar los gradientes
                optimizer.zero_grad()
                
                # Forward
                outputs = self.model(inputs) 
                loss = criterion(outputs, labels)  # Calculo de loss
                
                # L2 regularization
                l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                loss += l2_lambda * l2_norm

                # Backward + optimize
                scaler.scale(loss).backward()  # Escalar gradientes
                scaler.step(optimizer)  # Actualizar pesos
                scaler.update()  # Actualizar el escalador
                
                # Track current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                history['learning_rates'].append(current_lr)
                
                # Loss y Accuracy del Batch 
                history['train_loss'].append(loss.item())
                
                # Calcular accuracy
                _, preds = torch.max(outputs, 1)
                
                batch_acc = torch.sum(preds == labels.data) / inputs.size(0)
                history['train_acc'].append(batch_acc.item())
                
                # Acumular estadisitcas
                current_loss += loss.item() * inputs.size(0)  
                current_corrects += torch.sum(preds == labels.data)  
                total_samples += inputs.size(0)
                
                if i % 20 == 0:
                    print(f"  Batch {i}: Loss: {loss.item():.4f}, Acc: {batch_acc.item():.4f}")
            
            # Estadisiticas por epoch
            epoch_loss = current_loss / total_samples
            epoch_acc = current_corrects / total_samples
            
            history['epoch_train_loss'].append(epoch_loss)
            history['epoch_train_acc'].append(epoch_acc.item())
            
            print(f'  Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
            # Step the learning rate scheduler based on epoch loss
            lr_scheduler.step(epoch_loss)
    
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss  
                epochs_no_improve = 0  
            else:
                epochs_no_improve += 1 
                print(f'  No improvement in loss for {epochs_no_improve} epochs.')

                if epochs_no_improve == lr_patience:
                    print(f'  Learning rate early stopping triggered after {epoch+1} epochs.')
                
                if epochs_no_improve >= patience:
                    print(f'  Early stopping triggered after {epoch+1} epochs.')
                    break
    
        # Total training time
        time_elapsed = time.time() - start_time
        history['training_time'] = time_elapsed
    
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Final Accuracy: {epoch_acc:.4f}')
    
        return self.model, history

    def test_model(self, test_loader):
        """
        Evalua el modelo con un conjunto de entrenamiento
        
        Args:
            test_loader: DataLoader para los datos de entrenamiento
            
        Returns:
            test_acc: Test accuracy
            test_accuracies: Accuracy por Batch
            f1: F1 score
        """

        if self.model is None:
            return "Please create a model (ShipClassifier.create_model()) before training"

        print(f"Using device: {self.device}")

        self.model.to(self.device)  
        self.model.eval()  # Poner el modelo en modo evaluacion
        
        running_corrects = 0
        total_samples = 0
        test_accuracies = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                #preds = torch.argmax(outputs)

                batch_acc = torch.sum(preds == labels.data) / inputs.size(0)
                test_accuracies.append(batch_acc.item())
                
                # Guardar predicciones para calcular la F1-Score
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
       
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

        test_acc = running_corrects / total_samples
        print(f'Test Accuracy: {test_acc:.4f}')
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'F1 Score: {f1:.4f}')

        # Calcular la matriz de confusión
        cm = confusion_matrix(all_labels, all_preds)
        
        return test_acc.item(), test_accuracies, f1, cm
    
    def save_model(self, path):
        """
        Guarda el modelo en disco
        
        Args:
            path: direccion para guardar el modelo
        """
        if self.model is None:
            return "Please create a model (ShipClassifier.create_model()) before training"

        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """
        Carga el modelo desde disco
        
        Args:
            path: direcion donde se encuentra el modelo
        """
        if self.model is None:
            return "Please create a model (ShipClassifier.create_model()) before training"

        try:
            self.model.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
            
        except:
            
            new_state_dict = self.model.state_dict()
            
            for name, param in torch.load(path).items():
                if 'classifier.1' not in name:  # Skip the classifier layer
                    new_state_dict[name] = param
            self.model.load_state_dict(new_state_dict)


    def partial_load_model(self,original_model_path):
        # Load the original state dict
        original_state_dict = torch.load(original_model_path)
        
        # Create a new state dict for the new model
        new_state_dict = self.model.state_dict()
        
        # Copy all layers except the classifier layer
        for name, param in original_state_dict.items():
            if 'classifier.1' not in name:  # Skip the classifier layer
                new_state_dict[name] = param
        
        # Partially copy the classifier weights
        original_classifier_weight = original_state_dict['classifier.1.weight']
        original_classifier_bias = original_state_dict['classifier.1.bias']
        
        # Initialize the new classifier with the original weights
        new_state_dict['classifier.1.weight'][:2] = original_classifier_weight
        new_state_dict['classifier.1.bias'][:2] = original_classifier_bias
        
        # Load the modified state dict
        self.model.load_state_dict(new_state_dict)
        return self.model
    

    def plot_metrics(self, history, test_acc, cm, dataAugmentation, show=True):
        '''
        Grafica los resultados del entrenamiento y test
        
        Args:
            history: diccionario con las metricas a graficar
            test_acc: ultimo valor de accuracy en el test
        '''
        plt.figure(figsize=(12, 5))
        # ahora ya no hace falta
        #plt.title(f'Aumento de datos {dataAugmentation}, Preentrenado {self.pretrained}, Docked {self.docked} ')

        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title('Loss')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()

        # Gráfico de precisión
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Test Accuracy: {test_acc:.4f}')
        plt.title('Accuracy')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()

        plt.tight_layout()
        if show:
            plt.show()
        else:
            current_directory_path = os.getcwd()
            subfolder_path = os.path.join(current_directory_path, 'figures')
            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            file_name = f'LOSS__A_{dataAugmentation}_P_{self.pretrained}_D_{self.docked}_MLP_{self.mlp_head}'
            file_path = os.path.join(subfolder_path, file_name)
            
            plt.savefig(file_path)
        # Visualizar la matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=list(range(len(cm))), yticklabels=list(range(len(cm))))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        
        if show:
            plt.show()
        else:
            
            current_directory_path = os.getcwd()
            subfolder_path = os.path.join(current_directory_path, 'figures')
            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            file_name = f'CM__A_{dataAugmentation}_P_{self.pretrained}_D_{self.docked}_MLP_{self.mlp_head}'
            file_path = os.path.join(subfolder_path, file_name)
            
            plt.savefig(file_path)

    def plotgrid(self, trainset, dataAugmentation, cols=8, rows=4,  argmax=True, show=True):
    
        figure = plt.figure(figsize=(cols*2, rows*2))
        view = np.random.permutation(cols * rows)
       
        preds = []
        indices_aleatorios = np.random.choice(np.arange(len(trainset)),cols * rows)
        for i, j in zip(range(1, cols * rows + 1), indices_aleatorios):
            sample, label = trainset[j]
            im = torch.permute(torch.tensor(np.expand_dims(sample,0),dtype=torch.float32),(0,1,2,3)).to('mps')
    
            sample = torch.Tensor.permute(sample,(1,2,0)).numpy()
            sample -= np.min(sample)
            sample /= np.max(sample)
            
            figure.add_subplot(rows, cols, i)
            
            #pred = np.round(scipy.special.softmax(classifier.model(im)[0].cpu().detach().numpy()),2)
            pred = np.round(self.model(im)[0].cpu().detach().numpy(),2)
            preds.append(pred)
            
            if argmax:
                pred = np.argmax(pred)
            
    
            plt.title(f'y {label} - ŷ {pred}')
            #plt.title(label,pred)
            plt.axis("off")
            plt.imshow(sample, cmap="gray")
    
        if show:
            plt.show()
        else:
            current_directory_path = os.getcwd()
            subfolder_path = os.path.join(current_directory_path, 'figures')
            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            file_name = f'GRID__A_{dataAugmentation}_P_{self.pretrained}_D_{self.docked}_MLP_{self.mlp_head}'
            file_path = os.path.join(subfolder_path, file_name)
            
            plt.savefig(file_path)
            
        labels = np.array(trainset.labels)
        match len(np.unique(labels)):
            case 2:
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter([row[0] for row in preds], [row[1] for row in preds], 
                           c=labels[indices_aleatorios])
                plt.legend(handles=scatter.legend_elements()[0], 
                          labels=[f'Class {label}' for label in np.unique(labels)],
                          title="Classes")
                plt.xlabel('no barco')
                plt.ylabel('barco')
                plt.title('Prediction Distribution')
                plt.grid(True, linestyle='--', alpha=0.7)
            case 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                scatter = ax.scatter([row[0] for row in preds], 
                           [row[1] for row in preds],
                           [row[2] for row in preds],
                           c=labels[indices_aleatorios])
                legend1 = ax.legend(*scatter.legend_elements(),
                                   loc="upper right", 
                                   title="Classes")
                ax.add_artist(legend1)
                ax.set_xlabel('0 - no barco')
                ax.set_ylabel('1 - barco (undocked)')
                ax.set_zlabel('2 - barco (docked)')
                ax.set_title('3D Prediction Distribution')
                
        if show:
            plt.show()
        else:
            current_directory_path = os.getcwd()
            subfolder_path = os.path.join(current_directory_path, 'figures')
            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            file_name = f'3D__A_{dataAugmentation}_P_{self.pretrained}_D_{self.docked}_MLP_{self.mlp_head}'
            file_path = os.path.join(subfolder_path, file_name)
            
            plt.savefig(file_path)
    def test_single_images(self, image_paths, docked=True):
        """
        Test the classifier on individual images
        
        Args:
            classifier: Trained ShipClassifier
            image_paths: List of paths to images
        """
        self.model.to(self.device)
        self.model.eval()
        
        transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if docked:
            labels = ["No Ship", "Ship (Undocked)", "Ship (Docked)"]
        else:
            labels = ["No Ship", "Ship"]
        
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                _, pred = torch.max(output, 1)
                
            print(f"Image: {path}")
            print(f"Prediction: {labels[pred.item()]}")
            print(f"Confidence: {torch.nn.functional.softmax(output, dim=1)[0]}")
            print("-" * 30)
