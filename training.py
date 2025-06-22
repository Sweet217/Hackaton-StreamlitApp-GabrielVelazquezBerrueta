import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 50
latent_dim = 20
input_dim = 28 * 28

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, y):
        # Concatenate input with one-hot encoded label
        inputs = torch.cat([x, y], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        # Concatenate latent vector with one-hot encoded label
        inputs = torch.cat([z, y], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    return F.one_hot(labels, num_classes=num_classes).float()

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function combining reconstruction loss and KL divergence"""
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

def train_vae():
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(device)
            labels = labels.to(device)
            
            # One-hot encode labels
            labels_onehot = one_hot_encode(labels)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(data, labels_onehot)
            
            # Calculate loss
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item()/len(data):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
    
    return model, train_losses

def generate_digit_samples(model, digit, num_samples=5, latent_dim=20):
    """Generate samples for a specific digit"""
    model.eval()
    with torch.no_grad():
        # Create one-hot encoded label for the digit
        label = torch.tensor([digit] * num_samples).to(device)
        label_onehot = one_hot_encode(label)
        
        # Sample from latent space
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # Generate samples
        samples = model.decode(z, label_onehot)
        
        # Reshape to 28x28 images
        samples = samples.view(num_samples, 28, 28)
        
    return samples.cpu().numpy()

def save_model(model, filepath='mnist_vae_model.pth'):
    """Save the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'hidden_dim': 400,
            'num_classes': 10
        }
    }, filepath)
    print(f"Model saved to {filepath}")

def visualize_generated_digits(model):
    """Generate and visualize samples for all digits"""
    fig, axes = plt.subplots(10, 5, figsize=(15, 30))
    
    for digit in range(10):
        samples = generate_digit_samples(model, digit, num_samples=5)
        
        for i in range(5):
            axes[digit, i].imshow(samples[i], cmap='gray')
            axes[digit, i].axis('off')
            if i == 0:
                axes[digit, i].set_ylabel(f'Digit {digit}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('generated_digits_sample.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train the model
    print("Starting VAE training...")
    model, losses = train_vae()
    
    # Save the model
    save_model(model, 'mnist_vae_model.pth')
    
    # Generate sample images
    print("Generating sample images...")
    visualize_generated_digits(model)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    print("Training completed!")