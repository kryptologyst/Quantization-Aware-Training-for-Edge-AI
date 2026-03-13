"""Quantization-aware training implementation."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

logger = logging.getLogger("qat")


class QATConfig:
    """Configuration for quantization-aware training."""
    
    def __init__(
        self,
        method: str = "qat",
        precision: str = "int8",
        weight_bits: int = 8,
        activation_bits: int = 8,
        per_channel: bool = True,
        symmetric: bool = True,
        calibration_samples: int = 1000,
        observer: str = "minmax",
        backend: str = "fbgemm"
    ):
        """Initialize QAT configuration.
        
        Args:
            method: Quantization method ('qat', 'ptq', 'hybrid')
            precision: Precision type ('int8', 'int4', 'fp16')
            weight_bits: Number of bits for weights
            activation_bits: Number of bits for activations
            per_channel: Whether to use per-channel quantization
            symmetric: Whether to use symmetric quantization
            calibration_samples: Number of samples for calibration
            observer: Observer type ('minmax', 'kl_divergence', 'percentile')
            backend: Quantization backend ('fbgemm', 'qnnpack', 'onednn')
        """
        self.method = method
        self.precision = precision
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.calibration_samples = calibration_samples
        self.observer = observer
        self.backend = backend
        
        # Set quantization parameters
        self.quant_min = -(2 ** (weight_bits - 1)) if symmetric else 0
        self.quant_max = (2 ** (weight_bits - 1)) - 1 if symmetric else (2 ** weight_bits) - 1


class QuantizationAwareTrainer:
    """Quantization-aware training implementation."""
    
    def __init__(self, config: QATConfig):
        """Initialize QAT trainer.
        
        Args:
            config: QAT configuration
        """
        self.config = config
        self.original_model = None
        self.qat_model = None
        self.calibration_data = None
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization-aware training.
        
        Args:
            model: Original model
            
        Returns:
            Model prepared for QAT
        """
        logger.info("Preparing model for quantization-aware training...")
        
        # Store original model
        self.original_model = model
        
        # Create a copy for QAT
        self.qat_model = self._create_qat_model(model)
        
        # Set quantization configuration
        self._configure_quantization()
        
        logger.info("Model prepared for QAT")
        return self.qat_model
    
    def _create_qat_model(self, model: nn.Module) -> nn.Module:
        """Create QAT model from original model.
        
        Args:
            model: Original model
            
        Returns:
            QAT model
        """
        # Create a deep copy
        qat_model = self._copy_model(model)
        
        # Add quantization stubs if not present
        if not hasattr(qat_model, 'quant'):
            qat_model.quant = QuantStub()
        if not hasattr(qat_model, 'dequant'):
            qat_model.dequant = DeQuantStub()
        
        # Replace layers with quantized versions
        qat_model = self._replace_with_quantized_layers(qat_model)
        
        return qat_model
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model.
        
        Args:
            model: Original model
            
        Returns:
            Copied model
        """
        import copy
        return copy.deepcopy(model)
    
    def _replace_with_quantized_layers(self, model: nn.Module) -> nn.Module:
        """Replace layers with quantized versions.
        
        Args:
            model: Model to modify
            
        Returns:
            Modified model with quantized layers
        """
        # Replace Conv2d layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Create quantized convolution
                quant_conv = quant_nn.Conv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None,
                    module.padding_mode
                )
                
                # Copy weights
                quant_conv.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    quant_conv.bias.data = module.bias.data.clone()
                
                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, name.split('.')[-1], quant_conv)
                else:
                    setattr(model, name, quant_conv)
        
        # Replace Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Create quantized linear
                quant_linear = quant_nn.Linear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None
                )
                
                # Copy weights
                quant_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    quant_linear.bias.data = module.bias.data.clone()
                
                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, name.split('.')[-1], quant_linear)
                else:
                    setattr(model, name, quant_linear)
        
        return model
    
    def _configure_quantization(self) -> None:
        """Configure quantization parameters."""
        # Set quantization descriptors
        quant_desc_input = QuantDescriptor(
            num_bits=self.config.activation_bits,
            calib_method=self.config.observer,
            axis=None
        )
        
        quant_desc_weight = QuantDescriptor(
            num_bits=self.config.weight_bits,
            calib_method=self.config.observer,
            axis=0 if self.config.per_channel else None
        )
        
        # Apply to all quantized layers
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        
        for name, module in self.qat_model.named_modules():
            if isinstance(module, (quant_nn.Conv2d, quant_nn.Linear)):
                module.input_quantizer = quant_desc_input
                module.weight_quantizer = quant_desc_weight
    
    def calibrate(self, calibration_data: torch.Tensor) -> None:
        """Calibrate quantization parameters.
        
        Args:
            calibration_data: Calibration dataset
        """
        logger.info("Calibrating quantization parameters...")
        
        self.calibration_data = calibration_data
        
        # Enable calibration mode
        self.qat_model.eval()
        
        with torch.no_grad():
            for i in range(0, len(calibration_data), 32):  # Process in batches
                batch = calibration_data[i:i+32]
                if batch.size(0) > 0:
                    _ = self.qat_model(batch)
        
        logger.info("Calibration completed")
    
    def convert_to_quantized(self) -> nn.Module:
        """Convert QAT model to quantized model.
        
        Returns:
            Quantized model
        """
        logger.info("Converting QAT model to quantized model...")
        
        # Set to evaluation mode
        self.qat_model.eval()
        
        # Convert to quantized model
        quantized_model = quant.convert(self.qat_model)
        
        logger.info("Model converted to quantized format")
        return quantized_model
    
    def evaluate_quantization_error(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate quantization error.
        
        Args:
            test_data: Test data
            test_labels: Test labels
            
        Returns:
            Dictionary containing error metrics
        """
        logger.info("Evaluating quantization error...")
        
        # Original model evaluation
        self.original_model.eval()
        with torch.no_grad():
            original_outputs = self.original_model(test_data)
            original_predictions = torch.argmax(original_outputs, dim=1)
            original_accuracy = (original_predictions == test_labels).float().mean()
        
        # Quantized model evaluation
        quantized_model = self.convert_to_quantized()
        quantized_model.eval()
        with torch.no_grad():
            quantized_outputs = quantized_model(test_data)
            quantized_predictions = torch.argmax(quantized_outputs, dim=1)
            quantized_accuracy = (quantized_predictions == test_labels).float().mean()
        
        # Calculate metrics
        accuracy_drop = original_accuracy - quantized_accuracy
        mse_error = torch.mean((original_outputs - quantized_outputs) ** 2)
        
        # Model size comparison
        from src.utils.core import get_model_size
        original_size = get_model_size(self.original_model)
        quantized_size = get_model_size(quantized_model)
        size_reduction = (original_size['total_size_mb'] - quantized_size['total_size_mb']) / original_size['total_size_mb']
        
        results = {
            'original_accuracy': float(original_accuracy),
            'quantized_accuracy': float(quantized_accuracy),
            'accuracy_drop': float(accuracy_drop),
            'mse_error': float(mse_error),
            'size_reduction_ratio': float(size_reduction),
            'original_size_mb': original_size['total_size_mb'],
            'quantized_size_mb': quantized_size['total_size_mb']
        }
        
        logger.info(f"Quantization evaluation completed: {results}")
        return results


class PostTrainingQuantization:
    """Post-training quantization implementation."""
    
    def __init__(self, config: QATConfig):
        """Initialize PTQ.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor
    ) -> nn.Module:
        """Quantize model using post-training quantization.
        
        Args:
            model: Model to quantize
            calibration_data: Calibration data
            
        Returns:
            Quantized model
        """
        logger.info("Applying post-training quantization...")
        
        # Set model to evaluation mode
        model.eval()
        
        # Prepare model for quantization
        model_prepared = quant.prepare(model)
        
        # Calibrate with data
        with torch.no_grad():
            for i in range(0, len(calibration_data), 32):
                batch = calibration_data[i:i+32]
                if batch.size(0) > 0:
                    _ = model_prepared(batch)
        
        # Convert to quantized model
        quantized_model = quant.convert(model_prepared)
        
        logger.info("Post-training quantization completed")
        return quantized_model


def create_quantization_config(
    method: str = "qat",
    precision: str = "int8",
    **kwargs
) -> QATConfig:
    """Create quantization configuration.
    
    Args:
        method: Quantization method
        precision: Precision type
        **kwargs: Additional configuration parameters
        
    Returns:
        QAT configuration object
    """
    return QATConfig(method=method, precision=precision, **kwargs)


def benchmark_quantization_methods(
    model: nn.Module,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    calibration_data: torch.Tensor
) -> Dict[str, Dict[str, float]]:
    """Benchmark different quantization methods.
    
    Args:
        model: Original model
        test_data: Test data
        test_labels: Test labels
        calibration_data: Calibration data
        
    Returns:
        Dictionary containing benchmark results
    """
    logger.info("Benchmarking quantization methods...")
    
    results = {}
    
    # QAT
    qat_config = QATConfig(method="qat", precision="int8")
    qat_trainer = QuantizationAwareTrainer(qat_config)
    qat_model = qat_trainer.prepare_model(model)
    qat_trainer.calibrate(calibration_data)
    results['qat'] = qat_trainer.evaluate_quantization_error(test_data, test_labels)
    
    # PTQ
    ptq_config = QATConfig(method="ptq", precision="int8")
    ptq_trainer = PostTrainingQuantization(ptq_config)
    ptq_model = ptq_trainer.quantize_model(model, calibration_data)
    
    # Evaluate PTQ
    ptq_model.eval()
    with torch.no_grad():
        ptq_outputs = ptq_model(test_data)
        ptq_predictions = torch.argmax(ptq_outputs, dim=1)
        ptq_accuracy = (ptq_predictions == test_labels).float().mean()
    
    from src.utils.core import get_model_size
    ptq_size = get_model_size(ptq_model)
    original_size = get_model_size(model)
    
    results['ptq'] = {
        'quantized_accuracy': float(ptq_accuracy),
        'size_reduction_ratio': float((original_size['total_size_mb'] - ptq_size['total_size_mb']) / original_size['total_size_mb']),
        'quantized_size_mb': ptq_size['total_size_mb']
    }
    
    logger.info("Quantization benchmarking completed")
    return results
