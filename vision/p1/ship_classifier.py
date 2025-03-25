import torch
import torchvision
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import Counter

class ShipClassifier:
    def __init__(self, pretrained=True, docked=True):
        self.model = None
        self.create_model(pretrained, docked)
        
    def create_model(self, pretrained=True, docked=True):
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
        
        # SEGUN EL PREENTRENAMIENTO QUE HAGAMOS TENEMOS QUE CAMBIAR LA CAPA ORIGINAL
        # original_conv = model.features[0][0]
        # model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # En esta implementacion congelamos los pesos para hacer transfer learning
        for param in model.features.parameters():
            param.requires_grad = False
            
        # Modificamos para usar dos clases (barco/no barco)
        model.classifier[1] = nn.Linear(in_features=1536, out_features=n_outputs)
        
        #model.classifier.append(nn.Linear(20, 16))
        #model.classifier.append(nn.ReLU())
        #model.classifier.append(nn.Linear(16, n_outputs))
        model.classifier.append(nn.Softmax())
        
        self.model = model

        return model
        
    def forward(self, x):
        x = self.model(x)
        return x

    def train_model(self, train_loader, optimizer=None, criterion=None, num_epochs=3, patience=2):
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
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
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
                _, preds = torch.max(outputs, 1)
                #preds = torch.argmax(outputs)
                
                batch_acc = torch.sum(preds == labels.data) / inputs.size(0)
                history['train_acc'].append(batch_acc.item())
                
                # Acumular estadisitcas
                current_loss += loss.item() * inputs.size(0)  
                current_corrects += torch.sum(preds == labels.data)  
                total_samples += inputs.size(0)
                
                if i % 100 == 0:
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
            return "Please create a model (ShipClassifier.create_model()) before training"

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

        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        
    def plot_metrics(self, history, test_acc, dataAugmentation, pretrained, cm):
        '''
        Grafica los resultados del entrenamiento y test
        
        Args:
            history: diccionario con las metricas a graficar
            test_acc: ultimo valor de accuracy en el test
        '''
        plt.figure(figsize=(12, 5))
        plt.title(f'Aumento de datos {dataAugmentation}, Preentrenado {pretrained}')

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

        # Visualizar la matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=list(range(len(cm))), yticklabels=list(range(len(cm))))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

class RandomLargestSquareCrop(object):
    def __init__(self):
        pass
    def __call__(self, img):

        width, height = img.size
        min_dim = min(width, height)
        
        if width > height:
            left = random.randint(0, width - min_dim)
            top = 0
        elif height > width:
            left = 0
            top = random.randint(0, height - min_dim)
        else:
            left = 0
            top = 0
        
        img = img.crop((left, top, left + min_dim, top + min_dim))
        
        return img


class ShipDataset(Dataset):
    def __init__(self, root_dir, train=True, dataAugmentation=False, docked=False, transform=None, train_ratio=0.8):
        """
        Args:
            root_dir (string): Directory containing the image folders
            dataAugmentation (bool): Whether to apply data augmentation and include cropped ship images
            docked (bool): Whether to include docked status in labels
            transform (callable, optional): Optional additional transform to be applied
        """
        self.root_dir = root_dir
        self.train = train
        self.dataAugmentation = dataAugmentation
        self.docked = docked
        
        self.base_transform = transforms.Compose([
            RandomLargestSquareCrop(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),    
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=2),
            transforms.RandomGrayscale(p=0.15), 
        ])

        self.crop_transform = transforms.Compose([
            RandomLargestSquareCrop(),
            transforms.Resize((350,350)),
            transforms.RandomCrop(224),  # Adjust size as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.additional_transform = transform

        # Collect image paths and labels
        self.images = []
        self.labels = []

        # codigo de etiquetas
        # 0 - no barco
        # 1 - barco (undocked)
        # 2 - barco (docked)

        
        # 1. No-ship images (images/ns-xxx.jpg)
        no_ship_dir = os.path.join(root_dir, "images")
        for filename in os.listdir(no_ship_dir):
            if filename.startswith("ns-") and filename.endswith(".jpg"):
                img_path = os.path.join(no_ship_dir, filename)
                if self.docked:
                    self.images.append((img_path, "no_ship"))
                    self.labels.append(0)
                else:
                    self.images.append((img_path, "no_ship"))
                    self.labels.append(0)

        # 2. Cropped-ship images (only if dataAugmentation is True)
        if self.dataAugmentation:
            cropped_dir = os.path.join(root_dir, "cropedImages")
            for filename in os.listdir(cropped_dir):
                if filename.startswith("s-") and filename.endswith(".jpg"):
                    img_path = os.path.join(cropped_dir, filename)
                    if self.docked:
                        is_docked = 1 if filename.endswith("undocked.jpg") else 2
                        self.images.append((img_path, "cropped_ship"))
                        self.labels.append(is_docked)
                    else:
                        self.images.append((img_path, "cropped_ship"))
                        self.labels.append(1)

        # 3. Regular ship images (images/s-xxx-docked.jpg)
        for filename in os.listdir(no_ship_dir):
            if filename.startswith("s-") and filename.endswith(".jpg"):
                img_path = os.path.join(no_ship_dir, filename)
                if self.docked:
                    is_docked = 1 if filename.endswith("undocked.jpg") else 2
                    self.images.append((img_path, "regular_ship"))
                    self.labels.append(is_docked)
                else:
                    self.images.append((img_path, "regular_ship"))
                    self.labels.append(1)

        no_ship_count = sum(1 for img, type in self.images if type == "no_ship")
        cropped_ship_count = sum(1 for img, type in self.images if type == "cropped_ship")
        regular_ship_count = sum(1 for img, type in self.images if type == "regular_ship")
        
        no_ship_split = int(no_ship_count * train_ratio)
        cropped_ship_split = int(cropped_ship_count * train_ratio)
        regular_ship_split = int(regular_ship_count * train_ratio)
        
        # Create indices for each category
        no_ship_indices = [i for i, (_, type) in enumerate(self.images) if type == "no_ship"]
        cropped_ship_indices = [i for i, (_, type) in enumerate(self.images) if type == "cropped_ship"]
        regular_ship_indices = [i for i, (_, type) in enumerate(self.images) if type == "regular_ship"]
        
        # Select validation indices
        if self.train:
            no_ship_indices = no_ship_indices[:no_ship_split]
            cropped_ship_indices = cropped_ship_indices[:cropped_ship_split]
            regular_ship_indices = regular_ship_indices[:regular_ship_split]
        else:
            no_ship_indices = no_ship_indices[no_ship_split:]
            cropped_ship_indices = cropped_ship_indices[cropped_ship_split:]
            regular_ship_indices = regular_ship_indices[regular_ship_split:]
        
        # Combine all indices
        valid_indices = no_ship_indices + cropped_ship_indices + regular_ship_indices
        
        # Filter images and labels
        self.images = [self.images[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, img_type = self.images[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Apply appropriate transforms
        if self.dataAugmentation:
            if img_type == "no_ship" or img_type == "regular_ship":
                # Apply crop transform for no-ship and regular-ship images
                image = self.crop_transform(self.augmentation_transform(image))
            else:
                # Cropped ship images already processed
                image = self.base_transform(self.augmentation_transform(image))
        else:
            # No data augmentation, just apply base transform
            image = self.base_transform(image)

        # Apply any additional transforms if provided
        if self.additional_transform:
            image = self.additional_transform(image)

        return image, label

def plotgrid(classifier, trainset, cols=8, rows=4):

    figure = plt.figure(figsize=(cols*2, rows*2))
    view = np.random.permutation(cols * rows)
    
    for i, j in zip(range(1, cols * rows + 1), np.random.choice(np.arange(len(trainset)),cols * rows)):
        sample, label = trainset[j]
        im = torch.permute(torch.tensor(np.expand_dims(sample,0),dtype=torch.float32),(0,1,2,3)).to('mps')

        sample = torch.Tensor.permute(sample,(1,2,0)).numpy()
        sample -= np.min(sample)
        sample /= np.max(sample)
        figure.add_subplot(rows, cols, i)
        x = torch.Tensor(sample).reshape((1,224,224,3))
        #x = x.to('mps')
        
        #pred = np.round(scipy.special.softmax(classifier.model(im)[0].cpu().detach().numpy()),2)
        pred = np.argmax(classifier.model(im)[0].cpu().detach().numpy())
        
        plt.title(f'y {label} - ŷ {pred}')
        #plt.title(label,pred)
        plt.axis("off")
        plt.imshow(sample, cmap="gray")
    plt.show();

def test_single_images(classifier, image_paths, device='mps', docked=True):
    """
    Test the classifier on individual images
    
    Args:
        classifier: Trained ShipClassifier
        image_paths: List of paths to images
        device: Computation device
    """
    classifier.model.to(device)
    classifier.model.eval()
    
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
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = classifier.model(image_tensor)
            _, pred = torch.max(output, 1)
            
        print(f"Image: {path}")
        print(f"Prediction: {labels[pred.item()]}")
        print(f"Confidence: {torch.nn.functional.softmax(output, dim=1)[0]}")
        print("-" * 30)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ship Classification Training Script')
    
    # Dataset parameters
    parser.add_argument('--root_dir', type=str, default='/Users/pepe/carrera/3/2/vca/practicas/p2',
                        help='Root directory containing the image folders')
    parser.add_argument('--data_augmentation', action='store_true', 
                        help='Apply data augmentation and include cropped ship images')
    parser.add_argument('--docked', action='store_true',
                        help='Include docked status in labels')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data to total data')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights for the model')
    parser.add_argument('--model_path', type=str, default='modelParams',
                        help='Path to save or load the model')
    parser.add_argument('--load_model', action='store_true',
                        help='Load a pretrained model instead of training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    
    # Testing parameters
    parser.add_argument('--test_images', nargs='+', default=[],
                        help='List of image paths to test individually')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    '''
    # Basic run with data augmentation and pretrained model
    python ship_classifier.py --data_augmentation --pretrained
    
    # Change learning rate and batch size
    python ship_classifier.py --learning_rate 0.0005 --batch_size 256
    
    # Run with docked classification and more epochs
    python ship_classifier.py --docked --num_epochs 20 --patience 5
    
    # Load a previously trained model and test it
    python ship_classifier.py --load_model --model_path my_saved_model
    
    # Test specific images with a trained model
    python ship_classifier.py --load_model --test_images imagen2.jpg imagen3.jpg
    '''
    
    # Print the configuration
    print("\nRunning with the following configuration:")
    print(f"Data Augmentation: {args.data_augmentation}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Docked classification: {args.docked}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping patience: {args.patience}\n")
    
    # Set up datasets
    trainset = ShipDataset(
        root_dir=args.root_dir, 
        train=True, 
        dataAugmentation=args.data_augmentation,
        docked=args.docked,
        train_ratio=args.train_ratio
    )
    
    testset = ShipDataset(
        root_dir=args.root_dir, 
        train=False, 
        dataAugmentation=False,  # No augmentation for test set
        docked=args.docked,
        train_ratio=args.train_ratio
    )

   # class_counts = Counter(trainset.labels)
   # weights = [1.0 / class_counts[label] for label in trainset.labels]
   # sampler = WeightedRandomSampler(weights, len(weights))
    
    # Set up data loaders
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
       # sampler=sampler
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Create classifier
    classifier = ShipClassifier(pretrained=args.pretrained, 
                                docked=args.docked)
    
    if args.load_model:
        # Load a pre-trained model
        classifier.load_model(args.model_path)
        test_acc, test_accuracies, f1, cm = classifier.test_model(testloader)
    else:
        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            [param for param in classifier.model.parameters() if param.requires_grad], 
            lr=args.learning_rate
        )
        
        model, history = classifier.train_model(
            train_loader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=args.num_epochs,
            patience=args.patience
        )
        
        # Test and plot results
        test_acc, test_accuracies, f1, cm = classifier.test_model(testloader)
        classifier.plot_metrics(
            history, 
            test_acc, 
            dataAugmentation=args.data_augmentation, 
            pretrained=args.pretrained,
            cm=cm)
        
        plotgrid(classifier,testset)
        
        # Save the model
        classifier.save_model(args.model_path)
    
    # Test individual images if provided
    if args.test_images:
        classifier.load_model(args.model_path)
        print("\nTesting individual images:")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        test_single_images(classifier, args.test_images, device=device,docked=args.docked)