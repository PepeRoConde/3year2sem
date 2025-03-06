import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),  # Output: (8, 7, 7)
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),   # Output: (4, 4, 4)
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (8, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (1, 28, 28)
            nn.Sigmoid(),  # Use Sigmoid to scale pixel values between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def trainModel(self, train_loader):
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        num_epochs = 5
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                # Forward pass
                output = self(data)
                loss = criterion(output, data)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    
'''
# DataLoader for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
'''





'''
#train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = Autoencoder()



# Test the autoencoder
model.eval()
with torch.no_grad():
    sample_image, _ = next(iter(train_loader))
    output_image = model(sample_image)

    # Displaying images can be done here if needed using libraries like Matplotlib
'''