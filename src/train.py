import os
import sys
from tqdm import tqdm
import argparse

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from src.model import MnistCNN
from src.utils import count_parameters, save_model

class TransformDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label
    
    def __len__(self):
        return len(self.dataset)

def train_model(epochs=1, batch_size=8):
    # Use CUDA only if available and not in CI environment
    device = torch.device('cuda' if torch.cuda.is_available() and not os.getenv('CI') else 'cpu')
    print(f"Using device: {device}")
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load dataset
    full_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True,
                                            transform=None,  # No transform here
                                            download=True)
    
    # Split into 50K training and 10K validation
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms after splitting
    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)

    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=0)  # Set to 0 for GitHub Actions
    
    val_loader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=0)

    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)

    # Training loop with epochs
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0
        
        use_progress_bar = not os.getenv('CI')
        train_iterator = train_loader
        if use_progress_bar:
            train_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')

        for batch_idx, (data, target) in enumerate(train_iterator):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
            
            # Update progress bar
            if use_progress_bar:
                train_iterator.set_postfix({
                    'loss': f'{train_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })

        # Testing phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch_idx, (data, target) in enumerate(val_pbar):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{val_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })

        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {100.*train_correct/train_total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100.*val_correct/val_total:.2f}%\n')

    # Save model using test accuracy
    val_accuracy = 100. * val_correct / val_total
    os.makedirs("models", exist_ok=True)
    model_path = save_model(model, val_accuracy)
    
    return model, val_accuracy, model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    args = parser.parse_args()
    
    model, accuracy, model_path = train_model(epochs=args.epochs)
    print(f"Training completed with test accuracy: {accuracy:.2f}%")
    print(f"Model saved as: {model_path}") 