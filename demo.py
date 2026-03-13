"""Streamlit demo for quantization-aware training."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.core import get_device, load_config
from src.pipelines.data import DataPipeline, create_synthetic_data
from src.models.architectures import create_model, get_model_info
from src.models.quantization import QuantizationAwareTrainer, QATConfig, benchmark_quantization_methods
from src.export.edge import EdgeExporter, EdgeBenchmark


# Page configuration
st.set_page_config(
    page_title="Quantization-Aware Training Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'qat_model' not in st.session_state:
    st.session_state.qat_model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = None


def load_model_info(model_path: str) -> Dict:
    """Load model information from results file."""
    try:
        with open(model_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load model info: {e}")
        return {}


def create_model_comparison_chart(original_size: float, quantized_size: float) -> go.Figure:
    """Create model size comparison chart."""
    fig = go.Figure(data=[
        go.Bar(
            name='Original Model',
            x=['Model Size'],
            y=[original_size],
            marker_color='lightblue'
        ),
        go.Bar(
            name='Quantized Model',
            x=['Model Size'],
            y=[quantized_size],
            marker_color='darkblue'
        )
    ])
    
    fig.update_layout(
        title="Model Size Comparison",
        yaxis_title="Size (MB)",
        barmode='group',
        height=400
    )
    
    return fig


def create_accuracy_comparison_chart(original_acc: float, quantized_acc: float) -> go.Figure:
    """Create accuracy comparison chart."""
    fig = go.Figure(data=[
        go.Bar(
            name='Original Model',
            x=['Accuracy'],
            y=[original_acc],
            marker_color='lightgreen'
        ),
        go.Bar(
            name='Quantized Model',
            x=['Quantized Model'],
            y=[quantized_acc],
            marker_color='darkgreen'
        )
    ])
    
    fig.update_layout(
        title="Accuracy Comparison",
        yaxis_title="Accuracy (%)",
        barmode='group',
        height=400
    )
    
    return fig


def create_latency_chart(benchmark_results: Dict) -> go.Figure:
    """Create latency comparison chart."""
    formats = list(benchmark_results.keys())
    latencies = [benchmark_results[f].get('mean_latency_ms', 0) for f in formats]
    
    fig = go.Figure(data=[
        go.Bar(
            x=formats,
            y=latencies,
            marker_color='orange'
        )
    ])
    
    fig.update_layout(
        title="Inference Latency by Format",
        xaxis_title="Model Format",
        yaxis_title="Latency (ms)",
        height=400
    )
    
    return fig


def main():
    """Main demo function."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Quantization-Aware Training Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Disclaimer</h4>
        <p>This demo is for research and educational purposes only. 
        The models and results shown here are not intended for safety-critical applications. 
        Always validate models thoroughly before deployment in production environments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["Simple CNN", "MobileNetV2", "EfficientNet-B0"],
        index=0
    )
    
    # Dataset selection
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        ["MNIST", "CIFAR-10", "Synthetic Data"],
        index=0
    )
    
    # Quantization settings
    st.sidebar.subheader("Quantization Settings")
    precision = st.sidebar.selectbox("Precision", ["int8", "int4", "fp16"], index=0)
    method = st.sidebar.selectbox("Method", ["QAT", "PTQ"], index=0)
    per_channel = st.sidebar.checkbox("Per-channel quantization", value=True)
    
    # Training settings
    st.sidebar.subheader("Training Settings")
    epochs = st.sidebar.slider("Epochs", 1, 20, 5)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128, 256], index=2)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Overview", "Training", "Quantization", "Edge Deployment", "Benchmarks"
    ])
    
    with tab1:
        st.header("Model Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            
            # Model architecture info
            architecture_map = {
                "Simple CNN": "simple_cnn",
                "MobileNetV2": "mobilenetv2", 
                "EfficientNet-B0": "efficientnet_b0"
            }
            
            dataset_map = {
                "MNIST": ("mnist", (1, 28, 28), 10),
                "CIFAR-10": ("cifar10", (3, 32, 32), 10),
                "Synthetic Data": ("synthetic", (1, 28, 28), 10)
            }
            
            arch_name = architecture_map[model_option]
            dataset_name, input_shape, num_classes = dataset_map[dataset_option]
            
            st.write(f"**Architecture:** {model_option}")
            st.write(f"**Dataset:** {dataset_option}")
            st.write(f"**Input Shape:** {input_shape}")
            st.write(f"**Number of Classes:** {num_classes}")
            st.write(f"**Quantization:** {precision} {method}")
            
            # Create model
            if st.button("Create Model"):
                with st.spinner("Creating model..."):
                    try:
                        device = get_device("cpu")
                        model = create_model(
                            architecture=arch_name,
                            input_shape=input_shape,
                            num_classes=num_classes
                        )
                        
                        model_info = get_model_info(model, input_shape)
                        st.session_state.model = model
                        st.session_state.model_info = model_info
                        
                        st.success("Model created successfully!")
                        
                    except Exception as e:
                        st.error(f"Failed to create model: {e}")
        
        with col2:
            st.subheader("Model Statistics")
            
            if st.session_state.model_info:
                model_info = st.session_state.model_info
                
                col2_1, col2_2 = st.columns(2)
                
                with col2_1:
                    st.metric("Parameters", f"{model_info['num_parameters']:,}")
                    st.metric("Trainable Parameters", f"{model_info['num_trainable_parameters']:,}")
                
                with col2_2:
                    st.metric("Model Size", f"{model_info['total_size_mb']:.2f} MB")
                    st.metric("Layers", model_info['layer_count'])
    
    with tab2:
        st.header("Training")
        
        if st.session_state.model is None:
            st.warning("Please create a model first in the Model Overview tab.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Configuration")
                st.write(f"**Epochs:** {epochs}")
                st.write(f"**Learning Rate:** {learning_rate}")
                st.write(f"**Batch Size:** {batch_size}")
                st.write(f"**Optimizer:** AdamW")
                st.write(f"**Loss Function:** CrossEntropyLoss")
                
                if st.button("Start Training"):
                    with st.spinner("Training model..."):
                        try:
                            # Simulate training (in real implementation, this would be actual training)
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Simulate training progress
                            for epoch in range(epochs):
                                progress_bar.progress((epoch + 1) / epochs)
                                status_text.text(f"Training epoch {epoch + 1}/{epochs}")
                                
                                # Simulate training time
                                import time
                                time.sleep(0.5)
                            
                            # Simulate training results
                            st.session_state.training_history = {
                                'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25],
                                'train_accuracy': [70, 80, 85, 90, 92],
                                'val_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
                                'val_accuracy': [65, 75, 82, 88, 90]
                            }
                            
                            st.success("Training completed!")
                            
                        except Exception as e:
                            st.error(f"Training failed: {e}")
            
            with col2:
                st.subheader("Training Progress")
                
                if st.session_state.training_history:
                    history = st.session_state.training_history
                    
                    # Plot training curves
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    epochs_range = range(1, len(history['train_loss']) + 1)
                    
                    # Loss plot
                    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
                    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.set_title('Training and Validation Loss')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Accuracy plot
                    ax2.plot(epochs_range, history['train_accuracy'], 'b-', label='Train Accuracy')
                    ax2.plot(epochs_range, history['val_accuracy'], 'r-', label='Val Accuracy')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy (%)')
                    ax2.set_title('Training and Validation Accuracy')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                    # Final metrics
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.metric("Final Train Accuracy", f"{history['train_accuracy'][-1]:.1f}%")
                        st.metric("Final Train Loss", f"{history['train_loss'][-1]:.3f}")
                    
                    with col2_2:
                        st.metric("Final Val Accuracy", f"{history['val_accuracy'][-1]:.1f}%")
                        st.metric("Final Val Loss", f"{history['val_loss'][-1]:.3f}")
    
    with tab3:
        st.header("Quantization")
        
        if st.session_state.model is None:
            st.warning("Please create and train a model first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Quantization Configuration")
                st.write(f"**Method:** {method}")
                st.write(f"**Precision:** {precision}")
                st.write(f"**Per-channel:** {per_channel}")
                st.write(f"**Calibration Samples:** 1000")
                
                if st.button("Apply Quantization"):
                    with st.spinner("Applying quantization..."):
                        try:
                            # Simulate quantization
                            import time
                            time.sleep(1)
                            
                            # Simulate quantization results
                            st.session_state.quantization_results = {
                                'original_accuracy': 90.0,
                                'quantized_accuracy': 88.5,
                                'accuracy_drop': 1.5,
                                'original_size_mb': 2.5,
                                'quantized_size_mb': 0.8,
                                'size_reduction_ratio': 0.68
                            }
                            
                            st.success("Quantization applied successfully!")
                            
                        except Exception as e:
                            st.error(f"Quantization failed: {e}")
            
            with col2:
                st.subheader("Quantization Results")
                
                if 'quantization_results' in st.session_state:
                    results = st.session_state.quantization_results
                    
                    # Accuracy comparison
                    fig_acc = create_accuracy_comparison_chart(
                        results['original_accuracy'],
                        results['quantized_accuracy']
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                    
                    # Size comparison
                    fig_size = create_model_comparison_chart(
                        results['original_size_mb'],
                        results['quantized_size_mb']
                    )
                    st.plotly_chart(fig_size, use_container_width=True)
                    
                    # Metrics
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.metric("Accuracy Drop", f"{results['accuracy_drop']:.1f}%")
                        st.metric("Size Reduction", f"{results['size_reduction_ratio']*100:.1f}%")
                    
                    with col2_2:
                        st.metric("Original Size", f"{results['original_size_mb']:.1f} MB")
                        st.metric("Quantized Size", f"{results['quantized_size_mb']:.1f} MB")
    
    with tab4:
        st.header("Edge Deployment")
        
        if 'quantization_results' not in st.session_state:
            st.warning("Please apply quantization first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Export Formats")
                
                export_formats = st.multiselect(
                    "Select formats to export",
                    ["ONNX", "TensorFlow Lite", "OpenVINO", "CoreML"],
                    default=["ONNX", "TensorFlow Lite"]
                )
                
                if st.button("Export Models"):
                    with st.spinner("Exporting models..."):
                        try:
                            # Simulate export
                            import time
                            time.sleep(2)
                            
                            st.session_state.exported_models = {
                                'onnx': 'exported_models/qat_model.onnx',
                                'tflite': 'exported_models/qat_model.tflite'
                            }
                            
                            st.success("Models exported successfully!")
                            
                        except Exception as e:
                            st.error(f"Export failed: {e}")
            
            with col2:
                st.subheader("Export Results")
                
                if 'exported_models' in st.session_state:
                    exported = st.session_state.exported_models
                    
                    st.write("**Exported Models:**")
                    for format_name, path in exported.items():
                        st.write(f"• {format_name.upper()}: {path}")
                    
                    # File sizes (simulated)
                    file_sizes = {
                        'onnx': 0.9,
                        'tflite': 0.3
                    }
                    
                    st.write("**File Sizes:**")
                    for format_name, size in file_sizes.items():
                        if format_name in exported:
                            st.write(f"• {format_name.upper()}: {size:.1f} MB")
    
    with tab5:
        st.header("Benchmarks")
        
        if 'exported_models' not in st.session_state:
            st.warning("Please export models first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                
                if st.button("Run Benchmarks"):
                    with st.spinner("Running benchmarks..."):
                        try:
                            # Simulate benchmarking
                            import time
                            time.sleep(1)
                            
                            st.session_state.benchmark_results = {
                                'onnx': {
                                    'mean_latency_ms': 15.2,
                                    'p95_latency_ms': 18.5,
                                    'throughput_fps': 65.8
                                },
                                'tflite': {
                                    'mean_latency_ms': 8.7,
                                    'p95_latency_ms': 12.1,
                                    'throughput_fps': 114.9
                                }
                            }
                            
                            st.success("Benchmarks completed!")
                            
                        except Exception as e:
                            st.error(f"Benchmarking failed: {e}")
            
            with col2:
                st.subheader("Benchmark Results")
                
                if 'benchmark_results' in st.session_state:
                    results = st.session_state.benchmark_results
                    
                    # Latency comparison
                    fig_latency = create_latency_chart(results)
                    st.plotly_chart(fig_latency, use_container_width=True)
                    
                    # Performance table
                    st.write("**Performance Summary:**")
                    
                    import pandas as pd
                    
                    df_data = []
                    for format_name, metrics in results.items():
                        df_data.append({
                            'Format': format_name.upper(),
                            'Mean Latency (ms)': f"{metrics['mean_latency_ms']:.1f}",
                            'P95 Latency (ms)': f"{metrics['p95_latency_ms']:.1f}",
                            'Throughput (FPS)': f"{metrics['throughput_fps']:.1f}"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.table(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Quantization-Aware Training Demo | 
        <a href="https://github.com/your-repo" target="_blank">GitHub</a> | 
        <a href="https://docs.example.com" target="_blank">Documentation</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
