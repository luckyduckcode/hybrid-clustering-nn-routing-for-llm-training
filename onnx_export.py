"""
ONNX Export for Quantized Models

Export quantized PyTorch and TensorFlow models to ONNX format
for cross-platform deployment and optimization.

Features:
- PyTorch to ONNX quantized export
- TensorFlow to ONNX quantized export
- Quantization metadata preservation
- Inference validation
- Model compression statistics
"""

import torch
import torch.onnx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pytorch_integration import (
    QuantConfig,
    HybridTransformerWrapper,
    get_model_bit_compression_ratio
)

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Note: Install ONNX with: pip install onnx onnxruntime")


class PyTorchONNXExporter:
    """Export PyTorch quantized models to ONNX."""
    
    def __init__(self, model: torch.nn.Module, model_name: str = 'model'):
        """Initialize ONNX exporter."""
        self.model = model
        self.model_name = model_name
        self.device = next(model.parameters()).device
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 512),
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 14,
        do_constant_folding: bool = True,
        use_external_data_format: bool = False
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Shape of dummy input (batch_size, seq_length)
            input_names: Names of input tensors
            output_names: Names of output tensors
            opset_version: ONNX opset version
            do_constant_folding: Enable constant folding optimization
            use_external_data_format: Use external format for large models
        
        Returns:
            Path to exported model
        """
        if input_names is None:
            input_names = ['input_ids']
        
        if output_names is None:
            output_names = ['logits']
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, dtype=torch.long).to(self.device)
        
        # Export model
        print(f"Exporting {self.model_name} to ONNX...")
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                name: {0: 'batch_size'} for name in input_names + output_names
            },
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            use_external_data_format=use_external_data_format,
            verbose=False
        )
        
        print(f"✓ Model exported to {output_path}")
        
        # Verify export
        if ONNX_AVAILABLE:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model validation passed")
        
        return output_path
    
    def export_with_quantization_metadata(
        self,
        output_path: str,
        bit_widths: Dict[str, float],
        input_shape: Tuple[int, ...] = (1, 512),
        **export_kwargs
    ) -> str:
        """
        Export model with quantization metadata.
        
        Args:
            output_path: Path to save ONNX model
            bit_widths: Dictionary of parameter bit-widths
            input_shape: Shape of dummy input
            **export_kwargs: Additional export arguments
        
        Returns:
            Path to exported model
        """
        # Export main model
        onnx_path = self.export_to_onnx(
            output_path,
            input_shape=input_shape,
            **export_kwargs
        )
        
        # Save quantization metadata
        metadata_path = output_path.replace('.onnx', '_quantization.json')
        import json
        
        metadata = {
            'model_name': self.model_name,
            'target_bits': 1.58,
            'bit_widths': bit_widths,
            'num_quantized_params': len(bit_widths),
            'average_bits': float(np.mean(list(bit_widths.values()))),
            'compression_ratio': 32.0 / (np.mean(list(bit_widths.values())) + 1e-8)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Quantization metadata saved to {metadata_path}")
        
        return onnx_path
    
    def validate_onnx_inference(
        self,
        onnx_path: str,
        test_input_shape: Tuple[int, ...] = (1, 512),
        num_tests: int = 5
    ) -> Dict[str, Any]:
        """
        Validate ONNX inference against PyTorch.
        
        Args:
            onnx_path: Path to ONNX model
            test_input_shape: Shape of test input
            num_tests: Number of test cases
        
        Returns:
            Validation metrics
        """
        if not ONNX_AVAILABLE:
            print("ONNX not available for validation")
            return {}
        
        print(f"\nValidating ONNX inference...")
        
        # Create ONNX session
        sess = ort.InferenceSession(onnx_path)
        
        max_difference = 0.0
        avg_difference = 0.0
        all_differences = []
        
        with torch.no_grad():
            for i in range(num_tests):
                # Create random input
                test_input = torch.randint(0, 1000, test_input_shape)
                test_input_np = test_input.numpy()
                
                # PyTorch inference
                pytorch_output = self.model(test_input.to(self.device))
                if isinstance(pytorch_output, tuple):
                    pytorch_output = pytorch_output[0]
                pytorch_output_np = pytorch_output.cpu().numpy()
                
                # ONNX inference
                input_name = sess.get_inputs()[0].name
                onnx_output = sess.run(
                    None,
                    {input_name: test_input_np}
                )
                onnx_output_np = onnx_output[0]
                
                # Compare
                difference = np.max(np.abs(pytorch_output_np - onnx_output_np))
                all_differences.append(float(difference))
                max_difference = max(max_difference, difference)
                avg_difference += difference
        
        avg_difference /= num_tests
        
        validation_result = {
            'onnx_path': onnx_path,
            'num_tests': num_tests,
            'max_difference': float(max_difference),
            'avg_difference': float(avg_difference),
            'std_difference': float(np.std(all_differences)),
            'validation_passed': avg_difference < 1e-3
        }
        
        print(f"✓ ONNX Validation Results:")
        print(f"  Max Difference: {validation_result['max_difference']:.6f}")
        print(f"  Avg Difference: {validation_result['avg_difference']:.6f}")
        print(f"  Validation Passed: {validation_result['validation_passed']}")
        
        return validation_result


class TensorFlowONNXExporter:
    """Export TensorFlow quantized models to ONNX."""
    
    def __init__(self, model, model_name: str = 'model'):
        """Initialize TensorFlow ONNX exporter."""
        self.model = model
        self.model_name = model_name
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 512),
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None
    ) -> str:
        """
        Export TensorFlow model to ONNX.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Shape of input
            input_names: Names of input tensors
            output_names: Names of output tensors
        
        Returns:
            Path to exported model
        """
        try:
            import tf2onnx
            import tensorflow as tf
        except ImportError:
            print("Install tf2onnx with: pip install tf2onnx")
            return None
        
        if input_names is None:
            input_names = ['input_ids']
        
        if output_names is None:
            output_names = ['logits']
        
        print(f"Exporting TensorFlow model to ONNX...")
        
        # Convert model
        spec = (tf.TensorSpec(shape=input_shape, dtype=tf.int32, name='input_ids'),)
        
        output_path, _ = tf2onnx.convert.from_keras(
            self.model,
            input_signature=spec,
            output_path=output_path
        )
        
        print(f"✓ Model exported to {output_path}")
        
        return output_path


class QuantizationExportAnalyzer:
    """Analyze and report quantization export statistics."""
    
    def __init__(self, model: torch.nn.Module, bit_widths: Dict[str, float]):
        """Initialize analyzer."""
        self.model = model
        self.bit_widths = bit_widths
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get comprehensive export statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        bit_width_values = list(self.bit_widths.values())
        
        return {
            'total_parameters': total_params,
            'num_quantized_parameters': len(self.bit_widths),
            'average_bits': float(np.mean(bit_width_values)),
            'min_bits': float(np.min(bit_width_values)),
            'max_bits': float(np.max(bit_width_values)),
            'std_bits': float(np.std(bit_width_values)),
            'original_size_mb': (total_params * 32) / (8 * 1024 * 1024),
            'quantized_size_mb': (total_params * np.mean(bit_width_values)) / (8 * 1024 * 1024),
            'compression_ratio': 32.0 / np.mean(bit_width_values),
            'bit_distribution': self._get_bit_distribution()
        }
    
    def _get_bit_distribution(self) -> Dict[str, int]:
        """Get distribution of bit allocations."""
        distribution = {}
        for bits in self.bit_widths.values():
            bits_rounded = round(bits, 1)
            distribution[f'{bits_rounded}'] = distribution.get(f'{bits_rounded}', 0) + 1
        return distribution
    
    def print_export_report(self):
        """Print detailed export report."""
        stats = self.get_export_statistics()
        
        print("\n" + "="*60)
        print("QUANTIZATION EXPORT REPORT")
        print("="*60)
        print(f"\nModel Statistics:")
        print(f"  Total Parameters: {stats['total_parameters']:,}")
        print(f"  Quantized Parameters: {stats['num_quantized_parameters']:,}")
        
        print(f"\nBit Allocation:")
        print(f"  Average Bits: {stats['average_bits']:.2f}")
        print(f"  Min Bits: {stats['min_bits']:.1f}")
        print(f"  Max Bits: {stats['max_bits']:.1f}")
        print(f"  Std Dev: {stats['std_bits']:.2f}")
        
        print(f"\nModel Size:")
        print(f"  Original: {stats['original_size_mb']:.2f} MB")
        print(f"  Quantized: {stats['quantized_size_mb']:.2f} MB")
        print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
        
        print(f"\nBit Distribution:")
        for bits, count in sorted(stats['bit_distribution'].items()):
            percentage = (count / stats['num_quantized_parameters']) * 100
            print(f"  {bits} bits: {count} params ({percentage:.1f}%)")
        
        print("="*60 + "\n")


def export_quantized_model_complete(
    model: torch.nn.Module,
    config: QuantConfig,
    bit_widths: Dict[str, float],
    output_dir: str = './quantized_models',
    model_name: str = 'model'
) -> Dict[str, str]:
    """
    Complete export pipeline for quantized model.
    
    Args:
        model: PyTorch model to export
        config: Quantization configuration
        bit_widths: Dictionary of parameter bit-widths
        output_dir: Directory to save exports
        model_name: Name for the model
    
    Returns:
        Dictionary with paths to all exported files
    """
    import os
    import json
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nStarting export pipeline for {model_name}...")
    print(f"Output directory: {output_dir}\n")
    
    # 1. Export to ONNX
    exporter = PyTorchONNXExporter(model, model_name)
    onnx_path = os.path.join(output_dir, f'{model_name}.onnx')
    exporter.export_with_quantization_metadata(
        onnx_path,
        bit_widths
    )
    
    # 2. Validate ONNX
    if ONNX_AVAILABLE:
        validation = exporter.validate_onnx_inference(onnx_path)
    else:
        validation = {}
    
    # 3. Generate report
    analyzer = QuantizationExportAnalyzer(model, bit_widths)
    analyzer.print_export_report()
    
    stats = analyzer.get_export_statistics()
    stats_path = os.path.join(output_dir, f'{model_name}_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 4. Save config
    config_path = os.path.join(output_dir, f'{model_name}_config.json')
    config_dict = {
        'target_bits': config.target_bits,
        'adaptive_bits': config.adaptive_bits,
        'min_bits': config.min_bits,
        'max_bits': config.max_bits,
        'enable_mixed_precision': config.enable_mixed_precision
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 5. Save PyTorch model
    pytorch_path = os.path.join(output_dir, f'{model_name}_pytorch.pt')
    torch.save(model.state_dict(), pytorch_path)
    
    print(f"\nExport pipeline complete!")
    print(f"Files saved to: {output_dir}")
    
    return {
        'onnx': onnx_path,
        'statistics': stats_path,
        'config': config_path,
        'pytorch': pytorch_path,
        'metadata': onnx_path.replace('.onnx', '_quantization.json')
    }


if __name__ == '__main__':
    # Example usage
    print("ONNX Export Module")
    print("This module provides utilities for exporting quantized models to ONNX format")
    print("\nUsage:")
    print("1. Use PyTorchONNXExporter for PyTorch models")
    print("2. Use TensorFlowONNXExporter for TensorFlow models")
    print("3. Use export_quantized_model_complete for full pipeline")
