import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
from feature_extract import ProcessedAudioDataset
import logging
import torch.nn.utils.rnn as rnn_utils
import numpy as np
# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class ImprovedVoiceAssistantRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(ImprovedVoiceAssistantRNN, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Calculate lengths by finding the last non-zero index for each sequence
        # Sum across feature dimension to get a 1D mask for each timestep
        mask = (x.sum(dim=-1) != 0).long()
        lengths = mask.sum(dim=1).cpu()
        
        # Ensure lengths are valid (greater than 0)
        lengths = torch.clamp(lengths, min=1)
        
        # Sort sequences by length in descending order
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        
        # Pack the sequences
        packed_x = rnn_utils.pack_padded_sequence(x, lengths.long(), batch_first=True)
        
        # Process through RNN
        rnn_out, _ = self.rnn(packed_x)
        
        # Unpack the sequences
        rnn_out, _ = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
        
        # Restore original batch order
        _, unsort_idx = sort_idx.sort(0)
        rnn_out = rnn_out[unsort_idx]
        
        # Take the last valid output for each sequence
        batch_size = rnn_out.size(0)
        last_output = rnn_out[torch.arange(batch_size), lengths[unsort_idx] - 1]
        
        # Continue with the rest of your forward logic
        x = self.batch_norm(last_output)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        output = self.fc2(x)
        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
    logger.info("Starting training process...")
    
    checkpoint_dir = Path('checkpoints') / datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Add L2 regularization
            l2_lambda = 0.01
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            
            # Update progress bar with training metrics
            progress_bar.set_postfix({
                'train_loss': f'{train_loss/(train_total/labels.size(0)):.4f}',
                'train_acc': f'{100 * train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        val_accuracy = 100 * val_correct / val_total
        
        # Learning rate adjustment
        scheduler.step(val_loss)
        
        # Logging
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                   f"Train Loss: {train_loss/len(train_loader):.4f} - "
                   f"Train Acc: {100 * train_correct/train_total:.2f}% - "
                   f"Val Loss: {val_loss/len(val_loader):.4f} - "
                   f"Val Acc: {val_accuracy:.2f}%")
        
        # Save checkpoint if validation accuracy improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy
            }, checkpoint_path)
            logger.info(f"New best model saved! Validation Accuracy: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break


def collate_fn(batch):
    """
    Custom collate function for padding sequences in a batch.
    
    Args:
        batch: List of tuples (features, label)
    Returns:
        Padded features tensor and labels tensor
    """
    # Separate features and labels
    features, labels = zip(*batch)
    
    # Convert features to numpy arrays if they aren't already
    features = [np.array(f) if not isinstance(f, np.ndarray) else f for f in features]
    
    # Get the length of each sequence
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    
    # Get the feature dimension from the first sequence
    feature_dim = features[0].shape[1]
    
    # Create a padded array filled with zeros
    features_padded = np.zeros((len(features), max_len, feature_dim))
    
    # Fill in the actual sequences
    for i, feat in enumerate(features):
        features_padded[i, :feat.shape[0], :] = feat
    
    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features_padded)
    labels_tensor = torch.LongTensor(labels)
    
    return features_tensor, labels_tensor

def main():
    # Hyperparameters
    input_size = 13
    hidden_size = 256  # Increased from 128
    output_size = 2
    num_epochs = 30  # Increased to allow for early stopping
    learning_rate = 0.001
    batch_size = 32  # Increased from 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and split dataset
    dataset = ProcessedAudioDataset('processed_features/processed_features.h5')
    
    # Split ratios: 70% train, 15% validation, 15% test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4 if torch.cuda.is_available() else 0,
    collate_fn=collate_fn  # Use dynamic padding
)

    val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4 if torch.cuda.is_available() else 0,
    collate_fn=collate_fn  # Use dynamic padding
    )
    
    # Initialize improved model
    model = ImprovedVoiceAssistantRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    class_weights = torch.tensor([1.0, 200000 / 80000], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3, 
        verbose=True
    )
    
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()