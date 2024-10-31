import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from datetime import datetime
from tqdm import tqdm

# Import your model class
from train import ImprovedVoiceAssistantRNN, ProcessedAudioDataset

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, device, checkpoint_path):
        """
        Initialize the evaluator with a model and checkpoint path
        """
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.load_model()
        
    def load_model(self):
        """
        Load the model weights from checkpoint
        """
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded successfully from {self.checkpoint_path}")
            logger.info(f"Checkpoint validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}%")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def evaluate(self, test_loader):
        """
        Evaluate the model on the test set
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        # For storing prediction probabilities
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'accuracy': 100 * correct / total
        }
    
    def plot_confusion_matrix(self, labels, predictions, class_names=None):
        """
        Plot confusion matrix using seaborn
        """
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        results_dir = Path('evaluation_results')
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'confusion_matrix.png')
        plt.close()
    
    def plot_roc_curve(self, labels, probabilities):
        """
        Plot ROC curve
        """
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        
        for i in range(probabilities.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(labels == i, probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the plot
        results_dir = Path('evaluation_results')
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'roc_curve.png')
        plt.close()
        
        return roc_auc
    
    def save_metrics(self, eval_results, class_names=None):
        """
        Save all evaluation metrics to a JSON file
        """
        results_dir = Path('evaluation_results')
        results_dir.mkdir(exist_ok=True)
        
        # Calculate detailed metrics
        report = classification_report(
            eval_results['labels'],
            eval_results['predictions'],
            target_names=class_names if class_names else None,
            output_dict=True
        )
        
        # Calculate ROC AUC
        roc_auc = self.plot_roc_curve(eval_results['labels'], eval_results['probabilities'])
        
        # Combine all metrics/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/scripts/checkpoints/20241024_153238/best_model.pth
        metrics = {
            'accuracy': eval_results['accuracy'],
            'classification_report': report,
            'roc_auc': {f'class_{i}': auc for i, auc in roc_auc.items()},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_checkpoint': str(self.checkpoint_path)
        }
        
        # Save metrics to JSON
        with open(results_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation metrics saved to {results_dir / 'evaluation_metrics.json'}")
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
    # Configuration
    input_size = 13
    hidden_size = 256
    output_size = 2
    batch_size = 32
    checkpoint_path = 'checkpoints/20241031_172229/best_model.pth'  # Update this path
    class_names = ['class_0', 'class_1']  # Update with your class names
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load test dataset
    dataset = ProcessedAudioDataset('processed_features/processed_features.h5')
    
    
    # Use the test split from your training script
    test_size = int(0.15 * len(dataset))
    _, _, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_size - test_size, test_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        collate_fn = collate_fn
    )
    
    # Initialize model
    model = ImprovedVoiceAssistantRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device, checkpoint_path)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    eval_results = evaluator.evaluate(test_loader)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        eval_results['labels'],
        eval_results['predictions'],
        class_names=class_names
    )
    
    # Save all metrics
    evaluator.save_metrics(eval_results, class_names=class_names)
    
    logger.info(f"Final Accuracy: {eval_results['accuracy']:.2f}%")
    logger.info("Evaluation completed! Check evaluation_results directory for detailed metrics and plots.")

if __name__ == "__main__":
    main()