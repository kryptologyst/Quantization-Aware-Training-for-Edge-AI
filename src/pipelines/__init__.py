"""Data pipeline for quantization-aware training."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

logger = logging.getLogger("qat")


class MNISTDataset(Dataset):
    """Custom MNIST dataset with preprocessing for QAT."""
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        transform: Optional[transforms.Compose] = None,
        quantize: bool = False,
        quant_bits: int = 8
    ):
        """Initialize MNIST dataset.
        
        Args:
            data: Input data array
            targets: Target labels
            transform: Optional transforms to apply
            quantize: Whether to quantize input data
            quant_bits: Number of bits for quantization
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.quantize = quantize
        self.quant_bits = quant_bits
        
        if quantize:
            self.quant_min = 0
            self.quant_max = 2 ** quant_bits - 1
            
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, label)
        """
        image = self.data[idx]
        label = self.targets[idx]
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Quantize if requested
        if self.quantize:
            image = self._quantize_tensor(image)
            
        return image, torch.tensor(label, dtype=torch.long)
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified bit width.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Quantized tensor
        """
        # Normalize to [0, 1]
        tensor = tensor.clamp(0, 1)
        
        # Scale to quantization range
        tensor = tensor * (self.quant_max - self.quant_min) + self.quant_min
        
        # Round to nearest integer
        tensor = torch.round(tensor)
        
        # Scale back to [0, 1]
        tensor = tensor / (self.quant_max - self.quant_min)
        
        return tensor


class DataPipeline:
    """Data pipeline for loading and preprocessing datasets."""
    
    def __init__(
        self,
        dataset_name: str = "mnist",
        data_dir: str = "data/raw",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1
    ):
        """Initialize data pipeline.
        
        Args:
            dataset_name: Name of the dataset to load
            data_dir: Directory to store/load data
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            val_split: Fraction of data to use for validation
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load the specified dataset.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.dataset_name.lower() == "mnist":
            return self._load_mnist()
        elif self.dataset_name.lower() == "cifar10":
            return self._load_cifar10()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_mnist(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load MNIST dataset.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading MNIST dataset...")
        
        # Load raw data
        train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=None
        )
        test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=None
        )
        
        # Convert to numpy arrays for custom processing
        train_data = train_dataset.data.numpy()
        train_targets = train_dataset.targets.numpy()
        test_data = test_dataset.data.numpy()
        test_targets = test_dataset.targets.numpy()
        
        # Normalize and reshape
        train_data = train_data.astype(np.float32) / 255.0
        test_data = test_data.astype(np.float32) / 255.0
        
        train_data = train_data[..., np.newaxis]  # Add channel dimension
        test_data = test_data[..., np.newaxis]
        
        # Split training data into train/val
        val_size = int(len(train_data) * self.val_split)
        train_size = len(train_data) - val_size
        
        train_data_split, val_data_split = random_split(
            range(len(train_data)), [train_size, val_size]
        )
        
        train_indices = train_data_split.indices
        val_indices = val_data_split.indices
        
        # Create custom datasets
        train_dataset = MNISTDataset(
            train_data[train_indices],
            train_targets[train_indices],
            transform=self.train_transform
        )
        
        val_dataset = MNISTDataset(
            train_data[val_indices],
            train_targets[val_indices],
            transform=self.val_transform
        )
        
        test_dataset = MNISTDataset(
            test_data,
            test_targets,
            transform=self.val_transform
        )
        
        logger.info(f"Loaded MNIST: {len(train_dataset)} train, "
                   f"{len(val_dataset)} val, {len(test_dataset)} test samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def _load_cifar10(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load CIFAR-10 dataset.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading CIFAR-10 dataset...")
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            self.data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            self.data_dir, train=False, download=True, transform=val_transform
        )
        
        # Split training data
        val_size = int(len(train_dataset) * self.val_split)
        train_size = len(train_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create validation dataset with proper transform
        val_dataset.dataset = datasets.CIFAR10(
            self.data_dir, train=True, download=False, transform=val_transform
        )
        
        logger.info(f"Loaded CIFAR-10: {len(train_dataset)} train, "
                   f"{len(val_dataset)} val, {len(test_dataset)} test samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training, validation, and testing.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader
    
    def get_calibration_data(
        self,
        dataset: Dataset,
        num_samples: int = 1000
    ) -> torch.Tensor:
        """Get calibration data for quantization.
        
        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to use for calibration
            
        Returns:
            Calibration data tensor
        """
        indices = torch.randperm(len(dataset))[:num_samples]
        calibration_data = []
        
        for idx in indices:
            data, _ = dataset[idx]
            calibration_data.append(data)
        
        return torch.stack(calibration_data)


def create_synthetic_data(
    num_samples: int = 1000,
    input_shape: Tuple[int, ...] = (28, 28, 1),
    num_classes: int = 10,
    noise_level: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data for testing purposes.
    
    Args:
        num_samples: Number of samples to generate
        input_shape: Shape of input data
        num_classes: Number of classes
        noise_level: Amount of noise to add
        
    Returns:
        Tuple of (data, labels)
    """
    # Generate random data
    data = torch.randn(num_samples, *input_shape)
    
    # Add some structure to make it more realistic
    for i in range(num_samples):
        # Create simple patterns
        pattern_type = i % 4
        if pattern_type == 0:
            # Vertical lines
            data[i, :, :, 0] = torch.randn(input_shape[0], input_shape[1]) * noise_level
            data[i, :, 10:18, 0] = 1.0
        elif pattern_type == 1:
            # Horizontal lines
            data[i, :, :, 0] = torch.randn(input_shape[0], input_shape[1]) * noise_level
            data[i, 10:18, :, 0] = 1.0
        elif pattern_type == 2:
            # Diagonal lines
            data[i, :, :, 0] = torch.randn(input_shape[0], input_shape[1]) * noise_level
            for j in range(min(input_shape[0], input_shape[1])):
                data[i, j, j, 0] = 1.0
        else:
            # Random noise
            data[i, :, :, 0] = torch.randn(input_shape[0], input_shape[1]) * noise_level
    
    # Generate labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    return data, labels
