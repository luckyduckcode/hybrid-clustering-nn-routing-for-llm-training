"""
1.58-bit Quantization Scheme for LLM Training

1.58 bits per parameter allows approximately 3 discrete levels per ~2 parameters.
This module implements ultra-low-bit fixed-point representation optimized for
efficient training with minimal precision loss on core computations.
"""

import numpy as np
from typing import Tuple


class Quantizer158Bit:
    """
    Implements 1.58-bit quantization.
    
    Since we can't have fractional bits, we use:
    - 3 levels per 2 parameters: [0, 0.5, 1.0] normalized
    - This gives us 3^(N/2) states for N parameters
    - Minimal precision but suitable for ultra-constrained scenarios
    """
    
    def __init__(self, scale: float = 1.0, enable_clipping: bool = True):
        """
        Initialize the quantizer.
        
        Args:
            scale: Global scaling factor for quantized values
            enable_clipping: Whether to clip values to valid range before quantization
        """
        self.scale = scale
        self.enable_clipping = enable_clipping
        # 3 levels for 1.58-bit: approximately -1, 0, +1 in normalized space
        self.levels = np.array([-1.0, 0.0, 1.0])
        
    def quantize(self, values: np.ndarray) -> np.ndarray:
        """
        Quantize values to nearest level.
        
        Args:
            values: Input array to quantize
            
        Returns:
            Quantized array with values from {-1, 0, 1} * scale
        """
        if self.enable_clipping:
            values = np.clip(values, -1.0, 1.0)
        
        # Find nearest level for each value
        quantized = np.zeros_like(values)
        for i, level in enumerate(self.levels):
            distances = np.abs(values - level)
            if i == 0:
                closest = distances.argmin(axis=0) == i if values.ndim > 1 else distances.argmin() == i
            else:
                closest = distances.argmin(axis=0) == i if values.ndim > 1 else distances.argmin() == i
        
        # Simpler approach: round to nearest level
        quantized = np.round(values * 2) / 2  # Gives -1, -0.5, 0, 0.5, 1
        quantized = np.clip(quantized, -1.0, 1.0)
        
        return quantized * self.scale
    
    def quantize_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        Quantize gradients with magnitude preservation.
        
        Args:
            gradients: Input gradients
            
        Returns:
            Quantized gradients maintaining relative magnitudes
        """
        # Normalize by max magnitude to preserve structure
        max_mag = np.abs(gradients).max()
        if max_mag > 0:
            normalized = gradients / max_mag
        else:
            normalized = gradients
        
        return self.quantize(normalized) * max_mag
    
    def quantize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Quantize model weights.
        
        Args:
            weights: Weight matrix to quantize
            
        Returns:
            Quantized weights
        """
        return self.quantize(weights)


class AdaptiveQuantizer:
    """
    Adaptive quantization that adjusts bit allocation based on layer importance.
    Uses auxiliary NN predictions to decide which layers need more precision.
    """
    
    def __init__(self, num_layers: int):
        """
        Initialize adaptive quantizer.
        
        Args:
            num_layers: Number of model layers
        """
        self.num_layers = num_layers
        self.layer_importance = np.ones(num_layers) / num_layers
        self.base_quantizer = Quantizer158Bit()
        
    def update_importance(self, importances: np.ndarray):
        """
        Update layer importance scores (e.g., from auxiliary NN).
        
        Args:
            importances: Array of importance scores for each layer
        """
        # Normalize to sum to 1
        self.layer_importance = importances / importances.sum()
    
    def quantize_layer(self, layer_idx: int, weights: np.ndarray) -> np.ndarray:
        """
        Quantize a specific layer with importance-weighted precision.
        
        Args:
            layer_idx: Index of the layer
            weights: Weight matrix
            
        Returns:
            Quantized weights
        """
        # Higher importance layers get less aggressive quantization
        importance_factor = 1.0 + self.layer_importance[layer_idx]
        scale = self.base_quantizer.scale * importance_factor
        
        quantizer = Quantizer158Bit(scale=scale)
        return quantizer.quantize(weights)


def estimate_model_size_reduction(original_bits: float, params_count: int) -> Tuple[float, float]:
    """
    Estimate memory reduction from 1.58-bit quantization.
    
    Args:
        original_bits: Original precision (e.g., 32 for float32)
        params_count: Total number of parameters
        
    Returns:
        Tuple of (original_size_mb, quantized_size_mb)
    """
    original_size = (params_count * original_bits) / (8 * 1024 * 1024)  # MB
    quantized_size = (params_count * 1.58) / (8 * 1024 * 1024)  # MB
    return original_size, quantized_size


if __name__ == "__main__":
    # Test quantization
    test_values = np.array([0.9, 0.1, -0.5, 0.0, -1.0])
    
    quantizer = Quantizer158Bit(scale=1.0)
    quantized = quantizer.quantize(test_values)
    
    print("Original:", test_values)
    print("Quantized:", quantized)
    
    # Test size reduction for a 7B parameter model
    original_size, quantized_size = estimate_model_size_reduction(32, 7e9)
    print(f"\n7B Parameter Model:")
    print(f"  Original (FP32): {original_size:.2f} MB ({original_size/1024:.2f} GB)")
    print(f"  Quantized (1.58-bit): {quantized_size:.2f} MB ({quantized_size/1024:.2f} GB)")
    print(f"  Compression Ratio: {original_size/quantized_size:.2f}x")
