"""Training pipeline for quantization-aware training."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.core import EarlyStopping, format_time
from src.models.quantization import QuantizationAwareTrainer, QATConfig

logger = logging.getLogger("qat")


class QATTrainer:
    """Quantization-aware training pipeline."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: QATConfig,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine"
    ):
        """Initialize QAT trainer.
        
        Args:
            model: Model to train
            device: Training device
            config: QAT configuration
            learning_rate: Learning rate
            weight_decay: Weight decay
            scheduler_type: Learning rate scheduler type
        """
        self.model = model
        self.device = device
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        
        # Initialize quantization trainer
        self.qat_trainer = QuantizationAwareTrainer(config)
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
    def prepare_for_training(self) -> nn.Module:
        """Prepare model for quantization-aware training.
        
        Returns:
            Prepared QAT model
        """
        logger.info("Preparing model for QAT...")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Prepare for QAT
        self.qat_model = self.qat_trainer.prepare_model(self.model)
        self.qat_model = self.qat_model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("Model prepared for QAT")
        return self.qat_model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer.
        
        Returns:
            Optimizer instance
        """
        return optim.AdamW(
            self.qat_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler.
        
        Returns:
            Learning rate scheduler
        """
        if self.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        elif self.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.qat_model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
            leave=False
        )
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.qat_model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.qat_model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.qat_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.qat_model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 5,
        save_best: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            save_path: Path to save model
            
        Returns:
            Training history
        """
        logger.info(f"Starting QAT training for {epochs} epochs...")
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            if self.scheduler_type == "plateau":
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)
            
            # Log progress
            elapsed_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f} - "
                f"Time: {format_time(elapsed_time)}"
            )
            
            # Save best model
            if save_best and val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                if save_path:
                    self.save_model(save_path)
            
            # Early stopping
            if early_stopping(val_acc, self.qat_model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {format_time(total_time)}")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        
        return self.training_history
    
    def evaluate(
        self,
        test_loader: DataLoader,
        calibration_data: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate the trained model.
        
        Args:
            test_loader: Test data loader
            calibration_data: Calibration data for quantization
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating QAT model...")
        
        # Calibrate if data provided
        if calibration_data is not None:
            self.qat_trainer.calibrate(calibration_data)
        
        # Evaluate QAT model
        self.qat_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.qat_model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        logger.info(f"Evaluation completed - Accuracy: {accuracy:.2f}%, F1: {f1:.4f}")
        
        return results
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.qat_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.qat_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {path}")


def create_trainer(
    model: nn.Module,
    device: torch.device,
    config: QATConfig,
    **kwargs
) -> QATTrainer:
    """Create QAT trainer.
    
    Args:
        model: Model to train
        device: Training device
        config: QAT configuration
        **kwargs: Additional trainer parameters
        
    Returns:
        QAT trainer instance
    """
    return QATTrainer(model, device, config, **kwargs)
