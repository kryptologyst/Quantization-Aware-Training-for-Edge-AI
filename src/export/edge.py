"""Model export and deployment for edge devices."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.onnx
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger("qat")


class EdgeExporter:
    """Export models for edge deployment."""
    
    def __init__(self, output_dir: str = "exported_models"):
        """Initialize edge exporter.
        
        Args:
            output_dir: Directory to save exported models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        filename: str = "model.onnx",
        opset_version: int = 11,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            filename: Output filename
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration
            
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting model to ONNX...")
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export path
        export_path = os.path.join(self.output_dir, filename)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info(f"ONNX model exported to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def export_to_tflite(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        filename: str = "model.tflite",
        representative_data: Optional[torch.Tensor] = None
    ) -> str:
        """Export model to TensorFlow Lite format.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            filename: Output filename
            representative_data: Representative data for quantization
            
        Returns:
            Path to exported TFLite model
        """
        logger.info("Exporting model to TensorFlow Lite...")
        
        try:
            import tensorflow as tf
            import tf2onnx
            
            # First export to ONNX
            onnx_path = self.export_to_onnx(
                model, input_shape, filename.replace('.tflite', '.onnx')
            )
            
            # Convert ONNX to TensorFlow
            tf_path = onnx_path.replace('.onnx', '.pb')
            
            # Convert ONNX to TensorFlow
            from onnx_tf.backend import prepare
            import onnx
            
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(tf_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
            
            if representative_data is not None:
                converter.representative_dataset = self._create_representative_dataset(
                    representative_data
                )
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = os.path.join(self.output_dir, filename)
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TensorFlow Lite model exported to {tflite_path}")
            return tflite_path
            
        except ImportError:
            logger.warning("TensorFlow not available, skipping TFLite export")
            return None
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
            return None
    
    def _create_representative_dataset(
        self,
        data: torch.Tensor,
        batch_size: int = 32
    ) -> callable:
        """Create representative dataset for TFLite quantization.
        
        Args:
            data: Representative data
            batch_size: Batch size for processing
            
        Returns:
            Representative dataset function
        """
        def representative_data_gen():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                yield [batch.numpy()]
        
        return representative_data_gen
    
    def export_to_openvino(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        filename: str = "model.xml"
    ) -> str:
        """Export model to OpenVINO format.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            filename: Output filename
            
        Returns:
            Path to exported OpenVINO model
        """
        logger.info("Exporting model to OpenVINO...")
        
        try:
            from openvino.tools import mo
            from openvino.runtime import Core
            
            # First export to ONNX
            onnx_path = self.export_to_onnx(
                model, input_shape, filename.replace('.xml', '.onnx')
            )
            
            # Convert ONNX to OpenVINO
            ov_model = mo.convert_model(onnx_path)
            
            # Save OpenVINO model
            ov_path = os.path.join(self.output_dir, filename)
            ov_model.serialize(ov_path)
            
            logger.info(f"OpenVINO model exported to {ov_path}")
            return ov_path
            
        except ImportError:
            logger.warning("OpenVINO not available, skipping OpenVINO export")
            return None
        except Exception as e:
            logger.error(f"OpenVINO export failed: {e}")
            return None
    
    def export_to_coreml(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        filename: str = "model.mlmodel"
    ) -> str:
        """Export model to CoreML format.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            filename: Output filename
            
        Returns:
            Path to exported CoreML model
        """
        logger.info("Exporting model to CoreML...")
        
        try:
            import coremltools as ct
            
            # First export to ONNX
            onnx_path = self.export_to_onnx(
                model, input_shape, filename.replace('.mlmodel', '.onnx')
            )
            
            # Convert ONNX to CoreML
            coreml_model = ct.convert(onnx_path)
            
            # Save CoreML model
            coreml_path = os.path.join(self.output_dir, filename)
            coreml_model.save(coreml_path)
            
            logger.info(f"CoreML model exported to {coreml_path}")
            return coreml_path
            
        except ImportError:
            logger.warning("CoreML Tools not available, skipping CoreML export")
            return None
        except Exception as e:
            logger.error(f"CoreML export failed: {e}")
            return None
    
    def export_all_formats(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        model_name: str = "qat_model",
        representative_data: Optional[torch.Tensor] = None
    ) -> Dict[str, str]:
        """Export model to all supported formats.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            model_name: Base name for exported models
            representative_data: Representative data for quantization
            
        Returns:
            Dictionary mapping format names to file paths
        """
        logger.info(f"Exporting {model_name} to all formats...")
        
        exported_models = {}
        
        # ONNX
        try:
            onnx_path = self.export_to_onnx(
                model, input_shape, f"{model_name}.onnx"
            )
            exported_models['onnx'] = onnx_path
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
        
        # TensorFlow Lite
        try:
            tflite_path = self.export_to_tflite(
                model, input_shape, f"{model_name}.tflite", representative_data
            )
            if tflite_path:
                exported_models['tflite'] = tflite_path
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
        
        # OpenVINO
        try:
            ov_path = self.export_to_openvino(
                model, input_shape, f"{model_name}.xml"
            )
            if ov_path:
                exported_models['openvino'] = ov_path
        except Exception as e:
            logger.error(f"OpenVINO export failed: {e}")
        
        # CoreML
        try:
            coreml_path = self.export_to_coreml(
                model, input_shape, f"{model_name}.mlmodel"
            )
            if coreml_path:
                exported_models['coreml'] = coreml_path
        except Exception as e:
            logger.error(f"CoreML export failed: {e}")
        
        logger.info(f"Export completed. Exported formats: {list(exported_models.keys())}")
        return exported_models


class EdgeBenchmark:
    """Benchmark exported models for edge deployment."""
    
    def __init__(self):
        """Initialize edge benchmark."""
        pass
    
    def benchmark_onnx(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark ONNX model.
        
        Args:
            model_path: Path to ONNX model
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        try:
            import onnxruntime as ort
            
            # Create inference session
            session = ort.InferenceSession(model_path)
            
            # Prepare input
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})
            
            # Benchmark
            import time
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                _ = session.run(None, {input_name: dummy_input})
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            
            return {
                'mean_latency_ms': np.mean(times) * 1000,
                'std_latency_ms': np.std(times) * 1000,
                'p50_latency_ms': np.percentile(times, 50) * 1000,
                'p95_latency_ms': np.percentile(times, 95) * 1000,
                'p99_latency_ms': np.percentile(times, 99) * 1000,
                'throughput_fps': 1.0 / np.mean(times)
            }
            
        except ImportError:
            logger.warning("ONNX Runtime not available")
            return {}
        except Exception as e:
            logger.error(f"ONNX benchmarking failed: {e}")
            return {}
    
    def benchmark_tflite(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark TFLite model.
        
        Args:
            model_path: Path to TFLite model
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        try:
            import tensorflow as tf
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare input
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
            
            # Benchmark
            import time
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            
            return {
                'mean_latency_ms': np.mean(times) * 1000,
                'std_latency_ms': np.std(times) * 1000,
                'p50_latency_ms': np.percentile(times, 50) * 1000,
                'p95_latency_ms': np.percentile(times, 95) * 1000,
                'p99_latency_ms': np.percentile(times, 99) * 1000,
                'throughput_fps': 1.0 / np.mean(times)
            }
            
        except ImportError:
            logger.warning("TensorFlow Lite not available")
            return {}
        except Exception as e:
            logger.error(f"TFLite benchmarking failed: {e}")
            return {}
    
    def benchmark_all_formats(
        self,
        exported_models: Dict[str, str],
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark all exported models.
        
        Args:
            exported_models: Dictionary of exported model paths
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results for all formats
        """
        logger.info("Benchmarking all exported models...")
        
        results = {}
        
        for format_name, model_path in exported_models.items():
            logger.info(f"Benchmarking {format_name} model...")
            
            if format_name == 'onnx':
                results[format_name] = self.benchmark_onnx(
                    model_path, input_shape, num_runs
                )
            elif format_name == 'tflite':
                results[format_name] = self.benchmark_tflite(
                    model_path, input_shape, num_runs
                )
            else:
                logger.warning(f"Benchmarking not supported for {format_name}")
        
        return results


def create_edge_exporter(output_dir: str = "exported_models") -> EdgeExporter:
    """Create edge exporter instance.
    
    Args:
        output_dir: Output directory for exported models
        
    Returns:
        Edge exporter instance
    """
    return EdgeExporter(output_dir)


def create_edge_benchmark() -> EdgeBenchmark:
    """Create edge benchmark instance.
    
    Returns:
        Edge benchmark instance
    """
    return EdgeBenchmark()
