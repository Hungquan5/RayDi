import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from tqdm import tqdm
from model_RNN_scratch import RNN
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from tqdm import tqdm

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    m = targets.shape[0]
    log_likelihood = -np.log(predictions[range(m), targets])
    return np.mean(log_likelihood)

def collate_fn(batch):
    features, labels = zip(*batch)
    features = [np.array(f) for f in features]
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    feature_dim = features[0].shape[1]
    
    features_padded = np.zeros((len(features), max_len, feature_dim))
    for i, feat in enumerate(features):
        features_padded[i, :feat.shape[0], :] = feat
    
    return features_padded, np.array(labels)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate=0.001, 
                patience=5, min_batch_size=2, run_number=1):
    logger = logging.getLogger(__name__)
    run_dir = Path(f'training_runs/run_{run_number}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train = True
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader)
        for inputs, labels in progress_bar:
            if inputs.shape[0] < min_batch_size:
                continue
                
            # Forward pass
            outputs = model.forward(inputs)
            probs = softmax(outputs)
            predicted = np.argmax(probs, axis=1)
            
            # Compute loss
            loss = cross_entropy_loss(probs, labels)
            
            # Backward pass (you'll need to implement this in the RNN class)
            model.backward(loss)
            
            # Update weights using simple gradient descent
            for param in model.weights:
                param -= learning_rate * param.grad
            
            train_total += labels.shape[0]
            train_correct += np.sum(predicted == labels)
            train_loss += loss
            
            progress_bar.set_description(
                f"Epoch {epoch+1} - Loss: {loss:.4f}, Acc: {100*train_correct/train_total:.2f}%"
            )
        
        # Validation phase
        model.train = False
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        for inputs, labels in val_loader:
            if inputs.shape[0] < min_batch_size:
                continue
            
            outputs = model.forward(inputs)
            probs = softmax(outputs)
            predicted = np.argmax(probs, axis=1)
            
            loss = cross_entropy_loss(probs, labels)
            
            val_total += labels.shape[0]
            val_correct += np.sum(predicted == labels)
            val_loss += loss
        
        val_accuracy = 100 * val_correct / val_total
        
        # Save metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_accuracy': float(100 * train_correct/train_total),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy)
        }
        
        with open(run_dir / f'metrics_epoch_{epoch+1}.json', 'w') as f:
            json.dump(metrics, f)
        
        logger.info(f"Epoch {epoch+1} - Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            # Save model weights (you'll need to implement save/load in RNN class)
            model.save(run_dir / f'model_epoch_{epoch+1}.npy')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return best_val_accuracy

def run_multiple_trainings(dataset, input_size, hidden_size, output_size, 
                         num_epochs=30, num_runs=3, batch_size=32):
    best_overall_accuracy = 0.0
    
    for run in range(1, num_runs + 1):
        # Split dataset
        indices = np.random.permutation(len(dataset))
        train_size = int(0.8 * len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_loader = DataLoader(dataset, batch_size, indices=train_indices)
        val_loader = DataLoader(dataset, batch_size, indices=val_indices)
        
        model = RNN(input_size, hidden_size, output_size)
        
        accuracy = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            run_number=run
        )
        
        best_overall_accuracy = max(best_overall_accuracy, accuracy)
        
    return best_overall_accuracy

class DataLoader:
    def __init__(self, dataset, batch_size, indices=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = indices if indices is not None else np.arange(len(dataset))
        
    def __iter__(self):
        self.current = 0
        return self
        
    def __next__(self):
        if self.current >= len(self.indices):
            raise StopIteration
            
        batch_indices = self.indices[self.current:self.current + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.current += self.batch_size
        
        return collate_fn(batch)
class ProcessedAudioDataset:
    def __init__(self, data_path):
        
        self.data = np.load(data_path, allow_pickle=True)
        self.features = self.data['features']
        self.labels = self.data['labels']
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, num_epochs, learning_rate=0.001, 
                patience=5, min_batch_size=2, run_number=1):
    logger = logging.getLogger(__name__)
    run_dir = Path(f'training_runs/run_{run_number}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_accuracy = 0.0
    patience_counter = 0
    optimizer = Adam(model.parameters(), learning_rate)
    
    for epoch in range(num_epochs):
        model.train = True
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in progress_bar:
            if inputs.shape[0] < min_batch_size:
                continue
            
            outputs = model.forward(inputs)
            probs = softmax(outputs)
            loss = cross_entropy_loss(probs, labels)
            
            model.backward(loss)
            optimizer.step()
            
            predicted = np.argmax(probs, axis=1)
            train_correct += np.sum(predicted == labels)
            train_total += labels.shape[0]
            train_loss += loss
            
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.train = False
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        for inputs, labels in val_loader:
            if inputs.shape[0] < min_batch_size:
                continue
            
            outputs = model.forward(inputs)
            probs = softmax(outputs)
            loss = cross_entropy_loss(probs, labels)
            
            predicted = np.argmax(probs, axis=1)
            val_correct += np.sum(predicted == labels)
            val_total += labels.shape[0]
            val_loss += loss
        
        val_accuracy = 100 * val_correct / val_total
        logger.info(f"Epoch {epoch+1} - Val Acc: {val_accuracy:.2f}%")
        
        # Save checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            model.save(run_dir / f'model_epoch_{epoch+1}.npy')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_accuracy

class Adam:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in parameters.items()}
        self.v = {k: np.zeros_like(v) for k, v in parameters.items()}
        self.t = 0
    
    def step(self):
        self.t += 1
        for param_name, param in self.parameters.items():
            grad = param.grad
            
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * grad**2
            
            m_hat = self.m[param_name] / (1 - self.beta1**self.t)
            v_hat = self.v[param_name] / (1 - self.beta2**self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    dataset = ProcessedAudioDataset('processed_features.npy')
    
    # Create model
    model = RNN(
        input_size=13,
        hidden_size=256,
        output_size=2
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_loader = DataLoader(dataset, 32, indices=train_indices)
    val_loader = DataLoader(dataset, 32, indices=val_indices)
    
    # Train
    best_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30
    )
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()