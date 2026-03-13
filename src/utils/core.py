"""Core utilities for quantization-aware training."""

import logging
import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf


def setup_logging(log_level: str = "INFO", log_dir: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("qat")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "qat.log")
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device_type: str = "auto", fallback_to_cpu: bool = True) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device_type: Device type ('cpu', 'cuda', 'mps', 'auto')
        fallback_to_cpu: Whether to fallback to CPU if preferred device unavailable
        
    Returns:
        PyTorch device object
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    
    try:
        device = torch.device(device_type)
        # Test device availability
        if device_type == "cuda":
            torch.zeros(1).to(device)
        elif device_type == "mps":
            torch.zeros(1).to(device)
        return device
    except Exception as e:
        if fallback_to_cpu:
            logging.warning(f"Failed to use {device_type}, falling back to CPU: {e}")
            return torch.device("cpu")
        else:
            raise RuntimeError(f"Device {device_type} not available: {e}")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(config, save_path)


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model size metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing size metrics in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        "param_size_mb": param_size / (1024 * 1024),
        "buffer_size_mb": buffer_size / (1024 * 1024),
        "total_size_mb": total_size / (1024 * 1024),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


def count_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    """Count FLOPs for a model with given input shape.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (excluding batch dimension)
        
    Returns:
        Number of FLOPs
    """
    from thop import profile
    
    dummy_input = torch.randn(1, *input_shape)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_score: Current validation score
            model: Model to potentially restore weights for
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def create_directory_structure(base_path: str, directories: list) -> None:
    """Create directory structure if it doesn't exist.
    
    Args:
        base_path: Base directory path
        directories: List of directory names to create
    """
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
