import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.optim as optim
import matplotlib.pyplot as plt

class BoatClassifier:
    def __init__(self, pretrained=True):
        self.model = None
        self.create_model(pretrained)
        
    def create_model(self, pretrained=True):
        """
        Crea una EfficientNet para la clasificacion de barcos
        
        Args:
            pretrained (bool): usar preentrenamiento o no
        """

        model = models.efficientnet_b0(pretrained=pretrained)
        
        # SEGUN EL PREENTRENAMIENTO QUE HAGAMOS TENEMOS QUE CAMBIAR LA CAPA ORIGINAL
        # original_conv = model.features[0][0]
        # model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # En esta implementacion congelamos los pesos para hacer transfer learning
        for param in model.features.parameters():
            param.requires_grad = False
        # Usar el preentrenamiento
        # if pretrained:
        #     with torch.no_grad():
        #         # Average the weights across the 3 RGB channels to create weights for 1 channel
        #         model.features[0][0].weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)
        
        # Modificamos para usar dos clases (barco/no barco)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
        
        self.model = model
        return model

    def train_model(self, train_loader, optimizer=None, criterion=None, num_epochs=5, patience=2):
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
            return "Please create a model (BoatClassifier.create_model()) before training"
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {'GPU' if device == torch.device('cuda:0') else 'CPU'}")    

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {'MPS' if device.type == 'mps' else 'CPU'}")

        self.model.to(device)
        scaler = GradScaler()
     
        # History for metrics
        history = {
            'train_loss': [],         # Perdida por Batch
            'train_acc': [],          # Accuracy por Batch
            'epoch_train_loss': [],   # Avg loss por Epoch
            'epoch_train_acc': [],    # Avg accuracy por Epoch
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
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Reiniciar los gradientes
                optimizer.zero_grad()
                
                # Forward
                if device.type == 'cuda':
                    with autocast(device_type='cuda'): 
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)  # Calculo de loss
                else:
                    outputs = self.model(inputs) 
                    loss = criterion(outputs, labels)  # Calculo de loss
                
                # Backward + optimize
                scaler.scale(loss).backward()  # Escalar gradientes
                scaler.step(optimizer)  # Actualizar pesos
                scaler.update()  # Actualizar el escalador
                
                # Loss y Accuracy del Batch 
                history['train_loss'].append(loss.item())
                
                # Calcular accuracy
                _, predicted = torch.max(outputs, 1)
                batch_acc = torch.sum(predicted == labels.data) / inputs.size(0)
                history['train_acc'].append(batch_acc.item())
                
                # Acumular estadisitcas
                current_loss += loss.item() * inputs.size(0)  
                current_corrects += torch.sum(predicted == labels.data)  
                total_samples += inputs.size(0)
                
                if i % 20 == 0:
                    print(f"  Batch {i}: Loss: {loss.item():.4f}, Acc: {batch_acc.item():.4f}")
            
            # Estadisiticas por epoch
            epoch_loss = current_loss / total_samples
            epoch_acc = current_corrects / total_samples
            
            history['epoch_train_loss'].append(epoch_loss)
            history['epoch_train_acc'].append(epoch_acc.item())
            
            print(f'  Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss  
                epochs_no_improve = 0  
            else:
                epochs_no_improve += 1 
                print(f'  No improvement in loss for {epochs_no_improve} epochs.')
                
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
            return "Please create a model (BoatClassifier.create_model()) before training"

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {'MPS' if device.type == 'mps' else 'CPU'}")

        self.model.to(device)  
        self.model.eval()  # Poner el modelo en modo evaluacion
        
        running_corrects = 0
        total_samples = 0
        test_accuracies = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

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
        return test_acc.item(), test_accuracies, f1
    
    def save_model(self, path):
        """
        Guarda el modelo en disco
        
        Args:
            path: direccion para guardar el modelo
        """
        if self.model is None:
            return "Please create a model (BoatClassifier.create_model()) before training"

        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """
        Carga el modelo desde disco
        
        Args:
            path: direcion donde se encuentra el modelo
        """
        if self.model is None:
            return "Please create a model (BoatClassifier.create_model()) before training"

        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        
    def plot_metrics(self, history, test_acc):
        '''
        Grafica los resultados del entrenamiento y test
        
        Args:
            history: diccionario con las metricas a graficar
            test_acc: ultimo valor de accuracy en el test
        '''
        plt.figure(figsize=(12, 5))

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
        plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet B0 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)

    # DataLoaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    classifier = BoatClassifier(pretrained=True)
    # Guardar el modelo
    if classifier.model is None:
        print("Error: Model was not created properly.")
        classifier.create_model(pretrained=True)
        
        if classifier.model is None:
            print("Error: Failed to create model. Exiting.")
            exit(1)

    # Modificar para CIFAR10 la capa de salida
    classifier.model.classifier[1] = nn.Linear(in_features=1280, out_features=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([param for param in classifier.model.parameters() if param.requires_grad], 
                          lr=0.001, momentum=0.9)

    model, history = classifier.train_model(
        train_loader=trainloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=2,
        patience=3
    )

    test_acc, test_accuracies, f1 = classifier.test_model(testloader)
    classifier.plot_metrics(history, test_acc)
    classifier.save_model("./")
