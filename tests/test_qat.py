"""Tests for quantization-aware training."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.utils.core import set_seed, get_device, get_model_size
from src.models.architectures import SimpleCNN, create_model
from src.models.quantization import QATConfig, QuantizationAwareTrainer
from src.pipelines.data import DataPipeline, create_synthetic_data


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are reproducible
        torch.manual_seed(42)
        rand1 = torch.randn(10)
        torch.manual_seed(42)
        rand2 = torch.randn(10)
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_get_model_size(self):
        """Test model size calculation."""
        model = nn.Linear(10, 5)
        size_info = get_model_size(model)
        
        assert "total_size_mb" in size_info
        assert "num_parameters" in size_info
        assert size_info["num_parameters"] == 55  # 10*5 + 5 bias


class TestModelArchitectures:
    """Test model architectures."""
    
    def test_simple_cnn_creation(self):
        """Test SimpleCNN model creation."""
        model = SimpleCNN(input_shape=(1, 28, 28), num_classes=10)
        
        assert model.num_classes == 10
        assert model.input_shape == (1, 28, 28)
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_create_model(self):
        """Test model creation function."""
        model = create_model("simple_cnn", (1, 28, 28), 10)
        assert isinstance(model, SimpleCNN)
        
        with pytest.raises(ValueError):
            create_model("invalid_arch", (1, 28, 28), 10)


class TestQuantization:
    """Test quantization functionality."""
    
    def test_qat_config(self):
        """Test QAT configuration."""
        config = QATConfig(
            method="qat",
            precision="int8",
            weight_bits=8,
            activation_bits=8
        )
        
        assert config.method == "qat"
        assert config.precision == "int8"
        assert config.weight_bits == 8
        assert config.activation_bits == 8
        assert config.quant_min == -128
        assert config.quant_max == 127
    
    def test_quantization_trainer(self):
        """Test quantization trainer."""
        config = QATConfig()
        trainer = QuantizationAwareTrainer(config)
        
        model = SimpleCNN(input_shape=(1, 28, 28), num_classes=10)
        qat_model = trainer.prepare_model(model)
        
        assert qat_model is not None
        assert hasattr(qat_model, 'quant')
        assert hasattr(qat_model, 'dequant')


class TestDataPipeline:
    """Test data pipeline functionality."""
    
    def test_synthetic_data_creation(self):
        """Test synthetic data creation."""
        data, labels = create_synthetic_data(
            num_samples=100,
            input_shape=(1, 28, 28),
            num_classes=10
        )
        
        assert data.shape == (100, 1, 28, 28)
        assert labels.shape == (100,)
        assert labels.min() >= 0
        assert labels.max() < 10
    
    @patch('src.pipelines.data.datasets.MNIST')
    def test_data_pipeline_mnist(self, mock_mnist):
        """Test MNIST data pipeline."""
        # Mock MNIST dataset
        mock_train = Mock()
        mock_train.data = torch.randint(0, 255, (1000, 28, 28))
        mock_train.targets = torch.randint(0, 10, (1000,))
        
        mock_test = Mock()
        mock_test.data = torch.randint(0, 255, (200, 28, 28))
        mock_test.targets = torch.randint(0, 10, (200,))
        
        mock_mnist.side_effect = [(mock_train, mock_test), (mock_train, mock_test)]
        
        pipeline = DataPipeline(dataset_name="mnist", data_dir="test_data")
        
        # This would normally download MNIST, but we're mocking it
        with patch.object(pipeline, '_load_mnist') as mock_load:
            mock_load.return_value = (mock_train, mock_test, mock_test)
            train_ds, val_ds, test_ds = pipeline.load_dataset()
            
            assert train_ds is not None
            assert val_ds is not None
            assert test_ds is not None


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow."""
        # Set seed for reproducibility
        set_seed(42)
        
        # Create model
        model = create_model("simple_cnn", (1, 28, 28), 10)
        
        # Create QAT config
        config = QATConfig()
        
        # Create trainer
        trainer = QuantizationAwareTrainer(config)
        qat_model = trainer.prepare_model(model)
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = qat_model(x)
        
        assert output.shape == (2, 10)
        
        # Test calibration
        calibration_data = torch.randn(10, 1, 28, 28)
        trainer.calibrate(calibration_data)
        
        # Test quantization conversion
        quantized_model = trainer.convert_to_quantized()
        assert quantized_model is not None


if __name__ == "__main__":
    pytest.main([__file__])
