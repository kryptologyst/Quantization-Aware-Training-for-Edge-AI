"""Model architectures for quantization-aware training."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

logger = logging.getLogger("qat")


class SimpleCNN(nn.Module):
    """Simple CNN architecture for MNIST classification with QAT support."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        num_classes: int = 10,
        dropout_rate: float = 0.2
    ):
        """Initialize SimpleCNN model.
        
        Args:
            input_shape: Input tensor shape (C, H, W)
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self._forward_features(dummy_input)
            self.flattened_size = dummy_output.numel()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        # Quantize input
        x = self.quant(x)
        
        # Feature extraction
        x = self._forward_features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Dequantize output
        x = self.dequant(x)
        
        return x


class MobileNetV2Block(nn.Module):
    """MobileNetV2 inverted residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_factor: int = 6
    ):
        """Initialize MobileNetV2 block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            expansion_factor: Expansion factor for depthwise convolution
        """
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion layer
        expanded_channels = in_channels * expansion_factor
        self.expand_conv = nn.Conv2d(
            in_channels, expanded_channels, 1, bias=False
        )
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, 3, stride, 1,
            groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Projection layer
        self.project_conv = nn.Conv2d(
            expanded_channels, out_channels, 1, bias=False
        )
        self.project_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        residual = x
        
        # Expansion
        x = F.relu6(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise convolution
        x = F.relu6(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Projection
        x = self.project_bn(self.project_conv(x))
        
        # Residual connection
        if self.use_residual:
            x = x + residual
            
        return x


class MobileNetV2(nn.Module):
    """MobileNetV2 architecture optimized for edge deployment."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        dropout_rate: float = 0.2
    ):
        """Initialize MobileNetV2 model.
        
        Args:
            input_shape: Input tensor shape (C, H, W)
            num_classes: Number of output classes
            width_multiplier: Width multiplier for model scaling
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Initial convolution
        first_channels = int(32 * width_multiplier)
        self.conv1 = nn.Conv2d(input_shape[0], first_channels, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(first_channels)
        
        # MobileNetV2 blocks
        self.blocks = nn.ModuleList([
            MobileNetV2Block(first_channels, 16, 1, 1),
            MobileNetV2Block(16, 24, 2, 6),
            MobileNetV2Block(24, 24, 1, 6),
            MobileNetV2Block(24, 32, 2, 6),
            MobileNetV2Block(32, 32, 1, 6),
            MobileNetV2Block(32, 32, 1, 6),
            MobileNetV2Block(32, 64, 2, 6),
            MobileNetV2Block(64, 64, 1, 6),
            MobileNetV2Block(64, 64, 1, 6),
            MobileNetV2Block(64, 64, 1, 6),
            MobileNetV2Block(64, 96, 1, 6),
            MobileNetV2Block(96, 96, 1, 6),
            MobileNetV2Block(96, 96, 1, 6),
            MobileNetV2Block(96, 160, 2, 6),
            MobileNetV2Block(160, 160, 1, 6),
            MobileNetV2Block(160, 160, 1, 6),
            MobileNetV2Block(160, 320, 1, 6),
        ])
        
        # Final layers
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        # Quantize input
        x = self.quant(x)
        
        # Initial convolution
        x = F.relu6(self.bn1(self.conv1(x)))
        
        # MobileNetV2 blocks
        for block in self.blocks:
            x = block(x)
        
        # Final convolution
        x = F.relu6(self.bn2(self.conv2(x)))
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.classifier(x)
        
        # Dequantize output
        x = self.dequant(x)
        
        return x


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 architecture for edge deployment."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        num_classes: int = 1000,
        dropout_rate: float = 0.2
    ):
        """Initialize EfficientNet-B0 model.
        
        Args:
            input_shape: Input tensor shape (C, H, W)
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Stem
        self.stem = nn.Conv2d(input_shape[0], 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks (simplified EfficientNet-B0)
        self.blocks = nn.ModuleList([
            # Stage 1
            self._make_mbconv(32, 16, 1, 1, 1),
            # Stage 2
            self._make_mbconv(16, 24, 6, 2, 2),
            self._make_mbconv(24, 24, 6, 1, 2),
            # Stage 3
            self._make_mbconv(24, 40, 6, 2, 2),
            self._make_mbconv(40, 40, 6, 1, 2),
            # Stage 4
            self._make_mbconv(40, 80, 6, 2, 3),
            self._make_mbconv(80, 80, 6, 1, 3),
            self._make_mbconv(80, 80, 6, 1, 3),
            # Stage 5
            self._make_mbconv(80, 112, 6, 1, 3),
            self._make_mbconv(112, 112, 6, 1, 3),
            self._make_mbconv(112, 112, 6, 1, 3),
            # Stage 6
            self._make_mbconv(112, 192, 6, 2, 4),
            self._make_mbconv(192, 192, 6, 1, 4),
            self._make_mbconv(192, 192, 6, 1, 4),
            self._make_mbconv(192, 192, 6, 1, 4),
            # Stage 7
            self._make_mbconv(192, 320, 6, 1, 1),
        ])
        
        # Head
        self.conv_head = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn_head = nn.BatchNorm2d(1280)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_mbconv(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int,
        stride: int,
        num_blocks: int
    ) -> nn.Module:
        """Create MBConv blocks.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            expansion_factor: Expansion factor
            stride: Convolution stride
            num_blocks: Number of blocks
            
        Returns:
            MBConv module
        """
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MobileNetV2Block(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expansion_factor
                )
            )
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        # Quantize input
        x = self.quant(x)
        
        # Stem
        x = F.relu6(self.bn1(self.stem(x)))
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = F.relu6(self.bn_head(self.conv_head(x)))
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.classifier(x)
        
        # Dequantize output
        x = self.dequant(x)
        
        return x


def create_model(
    architecture: str,
    input_shape: Tuple[int, ...],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """Create model based on architecture name.
    
    Args:
        architecture: Model architecture name
        input_shape: Input tensor shape
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    if architecture.lower() == "simple_cnn":
        return SimpleCNN(input_shape, num_classes, **kwargs)
    elif architecture.lower() == "mobilenetv2":
        return MobileNetV2(input_shape, num_classes, **kwargs)
    elif architecture.lower() == "efficientnet_b0":
        return EfficientNetB0(input_shape, num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def get_model_info(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Union[int, float]]:
    """Get model information and statistics.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        
    Returns:
        Dictionary containing model statistics
    """
    from src.utils.core import get_model_size, count_flops
    
    # Model size
    size_info = get_model_size(model)
    
    # FLOPs count
    try:
        flops = count_flops(model, input_shape)
    except ImportError:
        flops = 0
    
    # Layer count
    layer_count = len(list(model.modules()))
    
    return {
        **size_info,
        "flops": flops,
        "layer_count": layer_count,
        "input_shape": input_shape,
        "output_classes": getattr(model, 'num_classes', 'unknown')
    }
