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
    def __init__(self, pretrained=True, docked=True, mlp_head=True, arquitecture='efficientnet_b4', device='mps', figure_path='figures'):
        self.pretrained = pretrained
        self.docked = docked
        self.mlp_head = mlp_head
        self.device = device
        self.arquitecture = arquitecture
        self.figure_path = figure_path

        self.model = None
        self.create_model(pretrained, docked)
        
    def create_model(self, pretrained=True, docked=True, mlp_head=True):
        """
        Crea una EfficientNet para la clasificacion de barcos
        
        Args:
            pretrained (bool): usar preentrenamiento o no
        """

        if pretrained:
            match self.arquitecture:
                # EfficientNet family
                case 'efficientnet_b0':
                    model = models.efficientnet_b0(weights='DEFAULT')
                case 'efficientnet_b1':
                    model = models.efficientnet_b1(weights='DEFAULT')
                case 'efficientnet_b2':
                    model = models.efficientnet_b2(weights='DEFAULT')
                case 'efficientnet_b3':
                    model = models.efficientnet_b3(weights='DEFAULT')
                case 'efficientnet_b4':
                    model = models.efficientnet_b4(weights='DEFAULT')
                case 'efficientnet_b5':
                    model = models.efficientnet_b5(weights='DEFAULT')
                case 'efficientnet_b6':
                    model = models.efficientnet_b6(weights='DEFAULT')
                case 'efficientnet_b7':
                    model = models.efficientnet_b7(weights='DEFAULT')
                case 'efficientnet_v2_s':
                    model = models.efficientnet_v2_s(weights='DEFAULT')
                case 'efficientnet_v2_m':
                    model = models.efficientnet_v2_m(weights='DEFAULT')
                case 'efficientnet_v2_l':
                    model = models.efficientnet_v2_l(weights='DEFAULT')
                
                # ResNet family
                case 'resnet18':
                    model = models.resnet18(weights='DEFAULT')
                case 'resnet34':
                    model = models.resnet34(weights='DEFAULT')
                case 'resnet50':
                    model = models.resnet50(weights='DEFAULT')
                case 'resnet101':
                    model = models.resnet101(weights='DEFAULT')
                case 'resnet152':
                    model = models.resnet152(weights='DEFAULT')
                case 'wide_resnet50_2':
                    model = models.wide_resnet50_2(weights='DEFAULT')
                case 'wide_resnet101_2':
                    model = models.wide_resnet101_2(weights='DEFAULT')
                
                # MobileNet family
                case 'mobilenet_v2':
                    model = models.mobilenet_v2(weights='DEFAULT')
                case 'mobilenet_v3_small':
                    model = models.mobilenet_v3_small(weights='DEFAULT')
                case 'mobilenet_v3_large':
                    model = models.mobilenet_v3_large(weights='DEFAULT')
                
                # DenseNet family
                case 'densenet121':
                    model = models.densenet121(weights='DEFAULT')
                case 'densenet161':
                    model = models.densenet161(weights='DEFAULT')
                case 'densenet169':
                    model = models.densenet169(weights='DEFAULT')
                case 'densenet201':
                    model = models.densenet201(weights='DEFAULT')
                
                # VGG family
                case 'vgg11':
                    model = models.vgg11(weights='DEFAULT')
                case 'vgg13':
                    model = models.vgg13(weights='DEFAULT')
                case 'vgg16':
                    model = models.vgg16(weights='DEFAULT')
                case 'vgg19':
                    model = models.vgg19(weights='DEFAULT')
                case 'vgg11_bn':
                    model = models.vgg11_bn(weights='DEFAULT')
                case 'vgg13_bn':
                    model = models.vgg13_bn(weights='DEFAULT')
                case 'vgg16_bn':
                    model = models.vgg16_bn(weights='DEFAULT')
                case 'vgg19_bn':
                    model = models.vgg19_bn(weights='DEFAULT')
                
                # Vision Transformer family
                case 'vit_b_16':
                    model = models.vit_b_16(weights='DEFAULT')
                case 'vit_b_32':
                    model = models.vit_b_32(weights='DEFAULT')
                case 'vit_l_16':
                    model = models.vit_l_16(weights='DEFAULT')
                case 'vit_l_32':
                    model = models.vit_l_32(weights='DEFAULT')
                case 'vit_h_14':
                    model = models.vit_h_14(weights='DEFAULT')
                
                # ConvNeXt family
                case 'convnext_tiny':
                    model = models.convnext_tiny(weights='DEFAULT')
                case 'convnext_small':
                    model = models.convnext_small(weights='DEFAULT')
                case 'convnext_base':
                    model = models.convnext_base(weights='DEFAULT')
                case 'convnext_large':
                    model = models.convnext_large(weights='DEFAULT')
                
                # RegNet family
                case 'regnet_y_400mf':
                    model = models.regnet_y_400mf(weights='DEFAULT')
                case 'regnet_y_800mf':
                    model = models.regnet_y_800mf(weights='DEFAULT')
                case 'regnet_y_1_6gf':
                    model = models.regnet_y_1_6gf(weights='DEFAULT')
                case 'regnet_y_3_2gf':
                    model = models.regnet_y_3_2gf(weights='DEFAULT')
                case 'regnet_y_8gf':
                    model = models.regnet_y_8gf(weights='DEFAULT')
                case 'regnet_y_16gf':
                    model = models.regnet_y_16gf(weights='DEFAULT')
                case 'regnet_y_32gf':
                    model = models.regnet_y_32gf(weights='DEFAULT')
                
                # Swin Transformer family
                case 'swin_t':
                    model = models.swin_t(weights='DEFAULT')
                case 'swin_s':
                    model = models.swin_s(weights='DEFAULT')
                case 'swin_b':
                    model = models.swin_b(weights='DEFAULT')
                case 'swin_v2_t':
                    model = models.swin_v2_t(weights='DEFAULT')
                case 'swin_v2_s':
                    model = models.swin_v2_s(weights='DEFAULT')
                case 'swin_v2_b':
                    model = models.swin_v2_b(weights='DEFAULT')
                
                # Other architectures
                case 'shufflenet_v2_x0_5':
                    model = models.shufflenet_v2_x0_5(weights='DEFAULT')
                case 'shufflenet_v2_x1_0':
                    model = models.shufflenet_v2_x1_0(weights='DEFAULT')
                case 'shufflenet_v2_x1_5':
                    model = models.shufflenet_v2_x1_5(weights='DEFAULT')
                case 'shufflenet_v2_x2_0':
                    model = models.shufflenet_v2_x2_0(weights='DEFAULT')
                case 'inception_v3':
                    model = models.inception_v3(weights='DEFAULT')
                case 'googlenet':
                    model = models.googlenet(weights='DEFAULT')
                case 'squeezenet1_0':
                    model = models.squeezenet1_0(weights='DEFAULT')
                case 'squeezenet1_1':
                    model = models.squeezenet1_1(weights='DEFAULT')
                case 'alexnet':
                    model = models.alexnet(weights='DEFAULT')
                case _:
                    raise ValueError(f"Architecture {self.architecture} not supported")
        else:
            match self.arquitecture:
                # EfficientNet family (without pretrained weights)
                case 'efficientnet_b0':
                    model = models.efficientnet_b0()
                case 'efficientnet_b1':
                    model = models.efficientnet_b1()
                case 'efficientnet_b2':
                    model = models.efficientnet_b2()
                case 'efficientnet_b3':
                    model = models.efficientnet_b3()
                case 'efficientnet_b4':
                    model = models.efficientnet_b4()
                case 'efficientnet_b5':
                    model = models.efficientnet_b5()
                case 'efficientnet_b6':
                    model = models.efficientnet_b6()
                case 'efficientnet_b7':
                    model = models.efficientnet_b7()
                case 'efficientnet_v2_s':
                    model = models.efficientnet_v2_s()
                case 'efficientnet_v2_m':
                    model = models.efficientnet_v2_m()
                case 'efficientnet_v2_l':
                    model = models.efficientnet_v2_l()
                
                # ResNet family
                case 'resnet18':
                    model = models.resnet18()
                case 'resnet34':
                    model = models.resnet34()
                case 'resnet50':
                    model = models.resnet50()
                case 'resnet101':
                    model = models.resnet101()
                case 'resnet152':
                    model = models.resnet152()
                case 'wide_resnet50_2':
                    model = models.wide_resnet50_2()
                case 'wide_resnet101_2':
                    model = models.wide_resnet101_2()
                
                # MobileNet family
                case 'mobilenet_v2':
                    model = models.mobilenet_v2()
                case 'mobilenet_v3_small':
                    model = models.mobilenet_v3_small()
                case 'mobilenet_v3_large':
                    model = models.mobilenet_v3_large()
                
                # DenseNet family
                case 'densenet121':
                    model = models.densenet121()
                case 'densenet161':
                    model = models.densenet161()
                case 'densenet169':
                    model = models.densenet169()
                case 'densenet201':
                    model = models.densenet201()
                
                # VGG family
                case 'vgg11':
                    model = models.vgg11()
                case 'vgg13':
                    model = models.vgg13()
                case 'vgg16':
                    model = models.vgg16()
                case 'vgg19':
                    model = models.vgg19()
                case 'vgg11_bn':
                    model = models.vgg11_bn()
                case 'vgg13_bn':
                    model = models.vgg13_bn()
                case 'vgg16_bn':
                    model = models.vgg16_bn()
                case 'vgg19_bn':
                    model = models.vgg19_bn()
                
                # Vision Transformer family
                case 'vit_b_16':
                    model = models.vit_b_16()
                case 'vit_b_32':
                    model = models.vit_b_32()
                case 'vit_l_16':
                    model = models.vit_l_16()
                case 'vit_l_32':
                    model = models.vit_l_32()
                case 'vit_h_14':
                    model = models.vit_h_14()
                
                # ConvNeXt family
                case 'convnext_tiny':
                    model = models.convnext_tiny()
                case 'convnext_small':
                    model = models.convnext_small()
                case 'convnext_base':
                    model = models.convnext_base()
                case 'convnext_large':
                    model = models.convnext_large()
                
                # RegNet family
                case 'regnet_y_400mf':
                    model = models.regnet_y_400mf()
                case 'regnet_y_800mf':
                    model = models.regnet_y_800mf()
                case 'regnet_y_1_6gf':
                    model = models.regnet_y_1_6gf()
                case 'regnet_y_3_2gf':
                    model = models.regnet_y_3_2gf()
                case 'regnet_y_8gf':
                    model = models.regnet_y_8gf()
                case 'regnet_y_16gf':
                    model = models.regnet_y_16gf()
                case 'regnet_y_32gf':
                    model = models.regnet_y_32gf()
                
                # Swin Transformer family
                case 'swin_t':
                    model = models.swin_t()
                case 'swin_s':
                    model = models.swin_s()
                case 'swin_b':
                    model = models.swin_b()
                case 'swin_v2_t':
                    model = models.swin_v2_t()
                case 'swin_v2_s':
                    model = models.swin_v2_s()
                case 'swin_v2_b':
                    model = models.swin_v2_b()
                
                # Other architectures
                case 'shufflenet_v2_x0_5':
                    model = models.shufflenet_v2_x0_5()
                case 'shufflenet_v2_x1_0':
                    model = models.shufflenet_v2_x1_0()
                case 'shufflenet_v2_x1_5':
                    model = models.shufflenet_v2_x1_5()
                case 'shufflenet_v2_x2_0':
                    model = models.shufflenet_v2_x2_0()
                case 'inception_v3':
                    model = models.inception_v3()
                case 'googlenet':
                    model = models.googlenet()
                case 'squeezenet1_0':
                    model = models.squeezenet1_0()
                case 'squeezenet1_1':
                    model = models.squeezenet1_1()
                case 'alexnet':
                    model = models.alexnet()
                case _:
                    raise ValueError(f"Architecture {self.arquitecture} not supported")


        if docked:
             n_outputs = 3
        else:
             n_outputs = 2
        
      
        # this configures the classifier head, appending layers or not as specified by the mlp_head argument  
        
        self.model = self.update_classifier(model, n_outputs)

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
            try:
                self.model.load_state_dict(torch.load(path,map_location=self.device))
                print(f"Model loaded from {path}")
    
    
            except:
                # cargamos lo que podamos del clasificador
    
                self.partial_load_model(path)
                print(f"Model partially loaded from {path}")
            
        except:
            # no va a cargar NADA del clasificador
            new_state_dict = self.model.state_dict()
            
            for name, param in torch.load(path,map_location=self.device).items():
                if 'classifier.1' not in name:  # Skip the classifier layer
                    new_state_dict[name] = param
            self.model.load_state_dict(new_state_dict)
            print(f"Model partially loaded from {path}, no classifier has been loaded")


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
            subfolder_path = os.path.join(current_directory_path, self.figure_path)
            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            file_name = f'LOSS__A_{dataAugmentation}_P_{self.pretrained}_D_{self.docked}_MLP_{self.mlp_head}_{self.arquitecture}'
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
            subfolder_path = os.path.join(current_directory_path, self.figure_path)
            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            file_name = f'CM__A_{dataAugmentation}_P_{self.pretrained}_D_{self.docked}_MLP_{self.mlp_head}_{self.arquitecture}'
            file_path = os.path.join(subfolder_path, file_name)
            
            plt.savefig(file_path)

    def plotgrid(self, trainset, dataAugmentation, cols=8, rows=4,  argmax=True, show=True):
    
        figure = plt.figure(figsize=(cols*2, rows*2))
        view = np.random.permutation(cols * rows)
       
        preds = []
        indices_aleatorios = np.random.choice(np.arange(len(trainset)),cols * rows)
        for i, j in zip(range(1, cols * rows + 1), indices_aleatorios):
            sample, label = trainset[j]
            im = torch.permute(torch.tensor(np.expand_dims(sample,0),dtype=torch.float32),(0,1,2,3)).to(self.device)
    
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
            subfolder_path = os.path.join(current_directory_path, self.figure_path)
            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            file_name = f'GRID__A_{dataAugmentation}_P_{self.pretrained}_D_{self.docked}_MLP_{self.mlp_head}_{self.arquitecture}'
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
            subfolder_path = os.path.join(current_directory_path, self.figure_path)
            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            file_name = f'3D__A_{dataAugmentation}_P_{self.pretrained}_D_{self.docked}_MLP_{self.mlp_head}_{self.arquitecture}'
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

    def update_classifier(self, model, out_features=64):
        """
        Update the model's classifier layer, optionally replacing it with an MLP.
        
        Args:
            model: The PyTorch model to modify
            out_features: The number of output features/classes
            mlp_head: If True, add intermediate FC layers (64, 32) before the final output layer
            
        Returns:
            model: The modified model
        """
        # For EfficientNet models
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[1].in_features
            
            if self.mlp_head:
                model.classifier[1] = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=64, out_features=32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=32, out_features=out_features),
                    nn.Softmax(dim=1)
                )
            else:
                model.classifier[1] = nn.Linear(in_features=in_features, out_features=out_features)
        
        # For ResNet, VGG, DenseNet models with fc layer
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            
            if self.mlp_head:
                model.fc = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=64, out_features=32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=32, out_features=out_features),
                    nn.Softmax(dim=1)
                )
            else:
                model.fc = nn.Linear(in_features=in_features, out_features=out_features)
        
        # For models with classifier as single Linear layer
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            
            if self.mlp_head:
                model.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=64, out_features=32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=32, out_features=out_features),
                    nn.Softmax(dim=1)
                )
            else:
                model.classifier = nn.Linear(in_features=in_features, out_features=out_features)
        
        # For Vision Transformers
        elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
            in_features = model.heads.head.in_features
            
            if self.mlp_head:
                model.heads.head = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=64, out_features=32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=32, out_features=out_features),
                    nn.Softmax(dim=1)
                )
            else:
                model.heads.head = nn.Linear(in_features=in_features, out_features=out_features)
        
        # For MobileNetV3, ShuffleNet, etc.
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module) and hasattr(model.classifier, 'in_features'):
            in_features = model.classifier.in_features
            
            if self.mlp_head:
                model.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=64, out_features=32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=32, out_features=out_features),
                    nn.Softmax(dim=1)
                )
            else:
                model.classifier = nn.Linear(in_features=in_features, out_features=out_features)
        
        # For ConvNeXt models
        elif hasattr(model, 'classifier') and hasattr(model.classifier, 'flatten') and hasattr(model.classifier, 'linear'):
            in_features = model.classifier.linear.in_features
            
            if self.mlp_head:
                model.classifier.linear = nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=64, out_features=32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(in_features=32, out_features=out_features),
                    nn.Softmax(dim=1)
                )
            else:
                model.classifier.linear = nn.Linear(in_features=in_features, out_features=out_features)
        
        else:
            raise ValueError(f"Unable to identify the classifier in the model {type(model).__name__}")
        
        return model
