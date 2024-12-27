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
import json
from model_RNN import RNN
from model_GRU import GRU 
from model_LSTM import LSTM
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, 
                patience=5, min_batch_size=2, min_epochs=10, max_attempts=3, run_number=1):
    logger.info(f"Starting training run #{run_number}...")
    
    # Create run-specific directory
    base_dir = Path('training_runs')
    run_dir = base_dir / f'run_{run_number}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    attempt = 0
    best_run_accuracy = 0.0
    best_run_model = None
    
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"Starting training attempt {attempt}/{max_attempts} of run #{run_number}")
        
        # Reset model weights for new attempt
        if attempt > 1:
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            optimizer = optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=3, verbose=True
            )
        
        checkpoint_dir = run_dir / f'attempt_{attempt}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_accuracy = 0.0
        patience_counter = 0
        completed_epochs = 0
        
        # Save training configuration
        config = {
            'run_number': run_number,
            'attempt': attempt,
            'num_epochs': num_epochs,
            'min_epochs': min_epochs,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'batch_size': train_loader.batch_size if hasattr(train_loader, 'batch_size') else None,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            valid_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Run {run_number}, Attempt {attempt}, Epoch {epoch + 1}/{num_epochs}")
            
            for inputs, labels in progress_bar:
                if inputs.size(0) < min_batch_size:
                    continue
                    
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    l2_lambda = 0.01
                    l2_reg = torch.tensor(0., device=device)
                    for param in model.parameters():
                        l2_reg += torch.norm(param)
                    loss += l2_lambda * l2_reg
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    train_loss += loss.item()
                    valid_batches += 1
                    
                    if valid_batches > 0:
                        progress_bar.set_postfix({
                            'train_loss': f'{train_loss/valid_batches:.4f}',
                            'train_acc': f'{100 * train_correct/train_total:.2f}%'
                        })
                except RuntimeError as e:
                    logger.warning(f"Error processing batch: {str(e)}")
                    continue
            
            if valid_batches == 0:
                continue
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_valid_batches = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if inputs.size(0) < min_batch_size:
                        continue
                        
                    try:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        val_loss += loss.item()
                        val_valid_batches += 1
                    except RuntimeError as e:
                        logger.warning(f"Error processing validation batch: {str(e)}")
                        continue
            
            if val_valid_batches == 0:
                continue
                
            val_accuracy = 100 * val_correct / val_total
            completed_epochs = epoch + 1
            
            scheduler.step(val_loss / val_valid_batches)
            
            # Save metrics for this epoch
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss/valid_batches,
                'train_accuracy': 100 * train_correct/train_total,
                'val_loss': val_loss/val_valid_batches,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # Save metrics to CSV
            metrics_file = checkpoint_dir / 'metrics.csv'
            metrics_exists = metrics_file.exists()
            with open(metrics_file, 'a') as f:
                if not metrics_exists:
                    f.write(','.join(metrics.keys()) + '\n')
                f.write(','.join(map(str, metrics.values())) + '\n')
            
            logger.info(f"Run {run_number}, Attempt {attempt}, Epoch {epoch + 1}/{num_epochs} - "
                       f"Train Loss: {train_loss/valid_batches:.4f} - "
                       f"Train Acc: {100 * train_correct/train_total:.2f}% - "
                       f"Val Loss: {val_loss/val_valid_batches:.4f} - "
                       f"Val Acc: {val_accuracy:.2f}%")
            
            # Save checkpoint if validation accuracy improved
            if val_accuracy > best_val_accuracy or epoch%5==0:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                checkpoint_path = checkpoint_dir / f'model_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'metrics': metrics
                }, checkpoint_path)
                
                # Update best run model if this is the best we've seen
                if val_accuracy > best_run_accuracy:
                    best_run_accuracy = val_accuracy
                    best_run_model = checkpoint_path
            else:
                patience_counter += 1
            
            # Only allow early stopping after minimum epochs
            # if epoch + 1 >= min_epochs and patience_counter >= patience:
            #     logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            #     break
        
        # If we completed enough epochs, we can stop attempts
        if completed_epochs >= min_epochs:
            logger.info(f"Successfully completed {completed_epochs} epochs (>= {min_epochs})")
            break
        else:
            logger.info(f"Only completed {completed_epochs} epochs (< {min_epochs}), starting new attempt")
    
    return best_run_accuracy, best_run_model

def run_multiple_trainings(
    dataset,  # Changed to accept dataset instead of loaders
    input_size,
    hidden_size,
    output_size,
    learning_rate=0.001,
    num_epochs=30,
    num_runs=3,
    batch_size=32
):
    """
    Perform multiple complete training runs and save results.
    Each run uses a different random seed for dataset splitting.
    """
    # Create base directory for all runs
    base_dir = Path('training_runs')
    base_dir.mkdir(exist_ok=True)
    
    # Save overall results
    results = []
    best_overall_accuracy = 0.0
    best_overall_model_path = None
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save hyperparameters
    hyperparams = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'train_val_split': '80:20',
        'device': str(device)
    }
    
    with open(base_dir / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    for run in range(1, num_runs + 1):
        logger.info(f"\nStarting Training Run {run}/{num_runs}")
        
        # Generate a unique seed for this run
        run_seed = 42 + run  # Different seed for each run
        torch.manual_seed(run_seed)
        logger.info(f"Using random seed: {run_seed} for run {run}")
        
        # Split dataset with new random seed for this run
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(run_seed)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            collate_fn=collate_fn,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        # Initialize fresh model and optimizer for each run
        model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        ).to(device)
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # Initialize criterion with class weights if needed
        class_weights = torch.tensor([1.0, 50/50], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        
        # Train the model
        accuracy, best_model_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            run_number=run
        )
        
        results.append({
            'run': run,
            'seed': run_seed,
            'accuracy': accuracy,
            'model_path': str(best_model_path)
        })
        
        # Update best overall model
        if accuracy > best_overall_accuracy:
            best_overall_accuracy = accuracy
            best_overall_model_path = best_model_path
        
        # Save results summary
        with open(base_dir / 'overall_results.json', 'w') as f:
            json.dump({
                'results': results,
                'best_accuracy': best_overall_accuracy,
                'best_model_path': str(best_overall_model_path),
                'hyperparameters': hyperparams
            }, f, indent=4)
        
        logger.info(f"Completed Run {run}/{num_runs}")
        logger.info(f"Best accuracy this run: {accuracy:.2f}%")
        logger.info(f"Best overall accuracy so far: {best_overall_accuracy:.2f}%")
    
    return results, best_overall_accuracy, best_overall_model_path



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

import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a model on processed audio features.")
    parser.add_argument(
        '--input_path', type=str, required=True,
        help="Path to the processed features file (e.g., 'processed_features/processed_features.h5')."
    )
    parser.add_argument(
        '--output_dir', type=str, default='training_runs',
        help="Directory to save training results (default: 'training_runs')."
    )
    parser.add_argument(
        '--input_size', type=int, default=13,
        help="Input size for the model (default: 13)."
    )
    parser.add_argument(
        '--hidden_size', type=int, default=256,
        help="Hidden size for the RNN (default: 256)."
    )
    parser.add_argument(
        '--output_size', type=int, default=2,
        help="Output size for the model (default: 2)."
    )
    parser.add_argument(
        '--num_epochs', type=int, default=30,
        help="Number of epochs to train (default: 30)."
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help="Learning rate for the optimizer (default: 0.001)."
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help="Batch size for training and validation (default: 32)."
    )
    parser.add_argument(
        '--num_runs', type=int, default=3,
        help="Number of independent training runs to perform (default: 3)."
    )
    args = parser.parse_args()

    # Log received parameters
    logger.info(f"Received arguments: {args}")

    # Load dataset
    dataset = ProcessedAudioDataset(args.input_path)

    # Run multiple training sessions
    results, best_accuracy, best_model_path = run_multiple_trainings(
        dataset=dataset,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_runs=args.num_runs,
        batch_size=args.batch_size
    )

    logger.info("\nTraining Summary:")
    logger.info(f"Best overall accuracy: {best_accuracy:.2f}%")
    logger.info(f"Best model saved at: {best_model_path}")

    # Load best model for use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = RNN(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size
    ).to(device)

    checkpoint = torch.load(best_model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    return best_model

if __name__ == "__main__":
    main()
