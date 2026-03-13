#!/usr/bin/env python3
"""Main training script for quantization-aware training."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.core import (
    setup_logging, set_seed, get_device, load_config, save_config,
    create_directory_structure
)
from src.pipelines.data import DataPipeline
from src.models.architectures import create_model, get_model_info
from src.models.quantization import QuantizationAwareTrainer, QATConfig
from src.pipelines.training import QATTrainer
from src.export.edge import EdgeExporter, EdgeBenchmark


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Quantization-Aware Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training (cpu, cuda, mps, auto)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export models without training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device != "auto":
        config.device.device_type = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(
        log_level=config.logging.level,
        log_dir=os.path.join(args.output_dir, "logs")
    )
    
    logger.info("Starting Quantization-Aware Training")
    logger.info(f"Configuration: {config}")
    
    # Set random seeds
    set_seed(config.seed, config.deterministic)
    
    # Get device
    device = get_device(config.device.device_type, config.device.fallback_to_cpu)
    logger.info(f"Using device: {device}")
    
    # Create directory structure
    create_directory_structure(args.output_dir, [
        "models", "logs", "exports", "assets", "configs"
    ])
    
    # Save configuration
    save_config(config, os.path.join(args.output_dir, "configs", "config.yaml"))
    
    try:
        # Load data
        logger.info("Loading dataset...")
        data_pipeline = DataPipeline(
            dataset_name=config.data.dataset,
            data_dir=config.data.data_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            val_split=0.1
        )
        
        train_dataset, val_dataset, test_dataset = data_pipeline.load_dataset()
        train_loader, val_loader, test_loader = data_pipeline.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Get calibration data
        calibration_data = data_pipeline.get_calibration_data(
            train_dataset, config.quantization.calibration_samples
        )
        
        # Create model
        logger.info("Creating model...")
        model = create_model(
            architecture=config.model.architecture,
            input_shape=tuple(config.model.input_shape),
            num_classes=config.model.num_classes
        )
        
        # Get model info
        model_info = get_model_info(model, tuple(config.model.input_shape))
        logger.info(f"Model info: {model_info}")
        
        # Create QAT configuration
        qat_config = QATConfig(
            method=config.quantization.method,
            precision=config.quantization.precision,
            weight_bits=config.quantization.weight_bits,
            activation_bits=config.quantization.activation_bits,
            per_channel=config.quantization.per_channel,
            symmetric=config.quantization.symmetric,
            calibration_samples=config.quantization.calibration_samples,
            observer=config.quantization.advanced.observer,
            backend=config.quantization.advanced.backend
        )
        
        if not args.export_only:
            # Create trainer
            trainer = QATTrainer(
                model=model,
                device=device,
                config=qat_config,
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                scheduler_type=config.training.scheduler
            )
            
            # Prepare for training
            qat_model = trainer.prepare_for_training()
            
            # Resume from checkpoint if specified
            if args.resume:
                logger.info(f"Resuming from checkpoint: {args.resume}")
                trainer.load_model(args.resume)
            
            # Train model
            logger.info("Starting training...")
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.training.epochs,
                early_stopping_patience=config.training.early_stopping_patience,
                save_best=True,
                save_path=os.path.join(args.output_dir, "models", "best_model.pth")
            )
            
            # Evaluate model
            logger.info("Evaluating model...")
            eval_results = trainer.evaluate(
                test_loader=test_loader,
                calibration_data=calibration_data
            )
            
            logger.info(f"Evaluation results: {eval_results}")
            
            # Save final model
            trainer.save_model(os.path.join(args.output_dir, "models", "final_model.pth"))
            
        else:
            # Load pre-trained model for export only
            if args.resume:
                logger.info(f"Loading model from: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                qat_model = model
            else:
                logger.error("--export-only requires --resume to specify model path")
                return
        
        # Export models
        logger.info("Exporting models for edge deployment...")
        exporter = EdgeExporter(os.path.join(args.output_dir, "exports"))
        
        exported_models = exporter.export_all_formats(
            model=qat_model,
            input_shape=tuple(config.model.input_shape),
            model_name="qat_model",
            representative_data=calibration_data
        )
        
        # Benchmark exported models
        logger.info("Benchmarking exported models...")
        benchmark = EdgeBenchmark()
        
        benchmark_results = benchmark.benchmark_all_formats(
            exported_models=exported_models,
            input_shape=tuple(config.model.input_shape),
            num_runs=100
        )
        
        logger.info(f"Benchmark results: {benchmark_results}")
        
        # Save results
        import json
        results = {
            'model_info': model_info,
            'exported_models': exported_models,
            'benchmark_results': benchmark_results
        }
        
        if not args.export_only:
            results['evaluation_results'] = eval_results
            results['training_history'] = history
        
        with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Training and export completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
