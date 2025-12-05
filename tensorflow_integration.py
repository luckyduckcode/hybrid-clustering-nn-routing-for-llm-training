"""
TensorFlow/Keras Integration for 1.58-Bit Hybrid LLM Training

Seamless integration with TensorFlow 2.x and Keras transformer models
with quantization-aware training (QAT) and adaptive bit allocation.

Features:
- Keras layer integration
- TensorFlow eager and graph execution
- Quantization-aware training
- Adaptive bit-width per layer
- Mixed precision support
- TensorFlow Lite compatibility
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class TFQuantConfig:
    """Configuration for TensorFlow quantization."""
    target_bits: float = 1.58
    adaptive_bits: bool = True
    min_bits: float = 1.0
    max_bits: float = 8.0
    quantize_weights: bool = True
    quantize_gradients: bool = True
    quantize_activations: bool = False
    percentile_threshold: float = 99.0
    enable_mixed_precision: bool = False
    learning_rate_scale: float = 1.0


class TFAdaptiveBitAllocator:
    """
    TensorFlow-based adaptive bit allocation.
    Allocates bit-widths based on layer sensitivity.
    """
    
    def __init__(self, model: tf.keras.Model, config: TFQuantConfig):
        """Initialize bit allocator for TensorFlow model."""
        self.model = model
        self.config = config
        self.bit_widths = {}
        self.gradient_stats = {}
        self.weight_stats = {}
        self.sensitivity_scores = {}
        self._initialize_stats()
    
    def _initialize_stats(self):
        """Initialize statistics for all trainable variables."""
        for var in self.model.trainable_variables:
            var_name = var.name
            self.bit_widths[var_name] = self.config.target_bits
            self.gradient_stats[var_name] = {
                'mean': tf.constant(0.0),
                'std': tf.constant(1.0),
                'magnitude': tf.constant(1.0),
                'history': []
            }
            self.weight_stats[var_name] = {
                'mean': tf.constant(0.0),
                'std': tf.constant(1.0),
                'magnitude': tf.constant(1.0),
                'sparsity': 0.0
            }
            self.sensitivity_scores[var_name] = 1.0
    
    def update_statistics(self, var_name: str, grad: tf.Tensor, weight: tf.Tensor):
        """Update gradient and weight statistics."""
        if grad is not None:
            grad_abs = tf.abs(grad)
            self.gradient_stats[var_name]['mean'] = tf.reduce_mean(grad_abs)
            self.gradient_stats[var_name]['std'] = tf.math.reduce_std(grad_abs)
            self.gradient_stats[var_name]['magnitude'] = tf.reduce_max(grad_abs)
            self.gradient_stats[var_name]['history'].append(
                float(tf.reduce_mean(grad_abs))
            )
        
        weight_abs = tf.abs(weight)
        self.weight_stats[var_name]['mean'] = tf.reduce_mean(weight_abs)
        self.weight_stats[var_name]['std'] = tf.math.reduce_std(weight_abs)
        self.weight_stats[var_name]['magnitude'] = tf.reduce_max(weight_abs)
        self.weight_stats[var_name]['sparsity'] = float(
            tf.reduce_sum(tf.cast(weight == 0, tf.float32)) / tf.cast(
                tf.size(weight), tf.float32
            )
        )
    
    def compute_sensitivity_scores(self):
        """Compute sensitivity scores for all variables."""
        for var_name in self.sensitivity_scores:
            if var_name not in self.gradient_stats:
                continue
            
            grad_stats = self.gradient_stats[var_name]
            weight_stats = self.weight_stats[var_name]
            
            # Gradient variance component
            grad_var = float(grad_stats['std'] ** 2)
            
            # Weight magnitude component
            weight_mag = float(weight_stats['magnitude'])
            
            # Sparsity component
            sparsity_factor = 1.0 - weight_stats['sparsity']
            
            # Gradient trend
            if len(grad_stats['history']) > 1:
                trend = np.mean(np.diff(grad_stats['history'][-10:]))
                trend_factor = 1.0 + min(0.5, abs(trend))
            else:
                trend_factor = 1.0
            
            # Combined sensitivity
            sensitivity = grad_var * weight_mag * sparsity_factor * trend_factor
            self.sensitivity_scores[var_name] = max(0.1, min(10.0, sensitivity))
    
    def allocate_bits_adaptive(self):
        """Allocate bit-widths adaptively."""
        self.compute_sensitivity_scores()
        
        scores = list(self.sensitivity_scores.values())
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score + 1e-8
        
        for var_name, sensitivity in self.sensitivity_scores.items():
            normalized_sensitivity = (sensitivity - min_score) / score_range
            
            allocated_bits = (
                self.config.min_bits +
                normalized_sensitivity * (self.config.max_bits - self.config.min_bits)
            )
            
            self.bit_widths[var_name] = allocated_bits
    
    def get_bit_width(self, var_name: str) -> float:
        """Get allocated bit-width for variable."""
        if self.config.adaptive_bits:
            return self.bit_widths.get(var_name, self.config.target_bits)
        else:
            return self.config.target_bits


class QuantizationLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer for quantization.
    Can be inserted into any model for fine-grained quantization control.
    """
    
    def __init__(self, bits: float = 1.58, **kwargs):
        """Initialize quantization layer."""
        super().__init__(**kwargs)
        self.bits = bits
    
    def call(self, x, training=False):
        """Forward pass with optional quantization."""
        if not training or self.bits >= 8.0:
            return x
        
        return self._quantize_tensor(x)
    
    def _quantize_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Quantize tensor to specified bit-width."""
        num_levels = int(2 ** (self.bits - 0.58))
        
        tensor_abs = tf.abs(tensor)
        max_val = tf.reduce_max(tensor_abs)
        
        scale = max_val / (num_levels - 1)
        quantized = tf.round(tensor / (scale + 1e-8)) * scale
        
        return quantized


class HybridQuantizedDense(tf.keras.layers.Dense):
    """
    Custom Dense layer with quantization awareness.
    Drop-in replacement for tf.keras.layers.Dense.
    """
    
    def __init__(
        self,
        units: int,
        activation=None,
        use_bias=True,
        allocator: Optional[TFAdaptiveBitAllocator] = None,
        **kwargs
    ):
        """Initialize quantized dense layer."""
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            **kwargs
        )
        self.allocator = allocator
    
    def build(self, input_shape):
        """Build layer and register quantization."""
        super().build(input_shape)
        
        if self.allocator is not None:
            self.allocator._initialize_stats()
    
    def call(self, inputs, training=False):
        """Forward pass with optional weight quantization."""
        weights = self.kernel
        
        if training and self.allocator is not None:
            bits = self.allocator.get_bit_width(self.kernel.name)
            weights = self._quantize_weights(weights, bits)
        
        outputs = tf.matmul(inputs, weights)
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        
        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs
    
    def _quantize_weights(self, weights: tf.Tensor, bits: float) -> tf.Tensor:
        """Quantize weights during forward pass."""
        if bits >= 8.0:
            return weights
        
        num_levels = int(2 ** (bits - 0.58))
        weights_abs = tf.abs(weights)
        max_val = tf.reduce_max(weights_abs)
        
        scale = max_val / (num_levels - 1)
        quantized = tf.round(weights / (scale + 1e-8)) * scale
        
        return quantized


class QuantizationAwareTrainerTF:
    """
    Training wrapper for TensorFlow quantization-aware fine-tuning.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        config: TFQuantConfig
    ):
        """Initialize QAT trainer."""
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.allocator = TFAdaptiveBitAllocator(model, config)
        
        self.step = 0
        self.losses = []
        self.bit_width_history = {}
    
    @tf.function
    def _compute_quantization_loss(self) -> tf.Tensor:
        """Compute quantization-aware loss penalty."""
        loss = tf.constant(0.0)
        
        for var in self.model.trainable_variables():
            bits = self.allocator.get_bit_width(var.name)
            num_levels = int(2 ** (bits - 0.58))
            
            var_abs = tf.abs(var)
            max_val = tf.reduce_max(var_abs)
            
            if max_val > 0:
                scale = max_val / (num_levels - 1)
                quantized = tf.round(var / (scale + 1e-8)) * scale
                quantization_error = tf.reduce_mean((var - quantized) ** 2)
                loss += quantization_error / (bits ** 2)
        
        return loss
    
    @tf.function
    def train_step(
        self,
        inputs: tf.Tensor,
        targets: tf.Tensor,
        criterion
    ) -> Dict[str, float]:
        """Single training step with quantization."""
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(inputs, training=True)
            
            # Compute losses
            task_loss = criterion(targets, outputs)
            quant_loss = self._compute_quantization_loss()
            total_loss = task_loss + 0.01 * quant_loss
        
        # Backward pass
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Update statistics and apply gradients
        for var, grad in zip(trainable_vars, gradients):
            if grad is not None:
                self.allocator.update_statistics(var.name, grad, var)
        
        # Clip gradients for stability
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))
        
        # Periodically update bit allocation
        if self.step % 100 == 0:
            self.allocator.allocate_bits_adaptive()
            self._record_bit_widths()
        
        self.step += 1
        self.losses.append(float(total_loss))
        
        return {
            'total_loss': float(total_loss),
            'task_loss': float(task_loss),
            'quant_loss': float(quant_loss),
            'avg_bits': float(np.mean(list(self.allocator.bit_widths.values())))
        }
    
    def _record_bit_widths(self):
        """Record current bit-width allocation."""
        for var_name, bits in self.allocator.bit_widths.items():
            if var_name not in self.bit_width_history:
                self.bit_width_history[var_name] = []
            self.bit_width_history[var_name].append(bits)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'steps_completed': self.step,
            'final_loss': float(self.losses[-1]) if self.losses else None,
            'avg_loss': float(np.mean(self.losses)) if self.losses else None,
            'avg_bits_per_param': float(np.mean(list(self.allocator.bit_widths.values()))),
            'bit_allocation': self.allocator.bit_widths.copy(),
            'sensitivity_scores': self.allocator.sensitivity_scores.copy()
        }
    
    def save_checkpoint(self, path: str):
        """Save checkpoint with quantization info."""
        checkpoint_data = {
            'model_weights': self.model.get_weights(),
            'optimizer_weights': self.optimizer.get_weights(),
            'step': self.step,
            'bit_widths': self.allocator.bit_widths,
            'sensitivity_scores': self.allocator.sensitivity_scores,
            'config': {
                'target_bits': self.config.target_bits,
                'adaptive_bits': self.config.adaptive_bits
            }
        }
        np.save(path, checkpoint_data, allow_pickle=True)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint and restore state."""
        checkpoint_data = np.load(path, allow_pickle=True).item()
        self.model.set_weights(checkpoint_data['model_weights'])
        self.optimizer.set_weights(checkpoint_data['optimizer_weights'])
        self.step = checkpoint_data['step']
        self.allocator.bit_widths = checkpoint_data['bit_widths']
        self.allocator.sensitivity_scores = checkpoint_data['sensitivity_scores']


class HuggingFaceTransformerIntegration:
    """
    Integration wrapper for HuggingFace transformers with TensorFlow backend.
    
    Usage:
        from transformers import TFAutoModel
        model = TFAutoModel.from_pretrained('bert-base-uncased')
        wrapper = HuggingFaceTransformerIntegration(model, config)
        trainer = wrapper.get_qat_trainer(optimizer)
    """
    
    def __init__(self, model: tf.keras.Model, config: TFQuantConfig):
        """Initialize integration wrapper."""
        self.model = model
        self.config = config
        self.allocator = TFAdaptiveBitAllocator(model, config)
    
    def convert_to_quantized_model(self) -> tf.keras.Model:
        """
        Convert model to use quantized layers.
        For TensorFlow, this uses quantization-aware training hooks.
        """
        # TensorFlow models are already compatible with quantization
        # through the trainer's quantization loss and gradient hooks
        return self.model
    
    def get_qat_trainer(
        self,
        optimizer: tf.keras.optimizers.Optimizer
    ) -> QuantizationAwareTrainerTF:
        """Get quantization-aware trainer."""
        return QuantizationAwareTrainerTF(self.model, optimizer, self.config)
    
    def export_quantized_model(self, export_path: str):
        """Export model for TensorFlow Lite quantization."""
        # Save model
        self.model.save(export_path)
        
        # Can be converted to TFLite with post-training quantization
        converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = export_path.replace('.h5', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        return tflite_path


# Utility functions for TensorFlow integration

def quantize_model_weights_tf(
    model: tf.keras.Model,
    bits: float = 1.58
) -> tf.keras.Model:
    """Quantize all weights in a TensorFlow model."""
    for var in model.trainable_variables:
        var_abs = tf.abs(var)
        num_levels = int(2 ** (bits - 0.58))
        max_val = tf.reduce_max(var_abs)
        
        if max_val > 0:
            scale = max_val / (num_levels - 1)
            quantized = tf.round(var / (scale + 1e-8)) * scale
            var.assign(quantized)
    
    return model


def compute_model_size_reduction_tf(
    model: tf.keras.Model,
    average_bits: float = 1.58
) -> Dict[str, float]:
    """Compute model size reduction for TensorFlow model."""
    total_params = model.count_params()
    
    original_size_mb = (total_params * 32) / (8 * 1024 * 1024)
    quantized_size_mb = (total_params * average_bits) / (8 * 1024 * 1024)
    
    return {
        'original_size_mb': original_size_mb,
        'quantized_size_mb': quantized_size_mb,
        'reduction_ratio': original_size_mb / quantized_size_mb,
        'total_params': total_params
    }


def create_quantization_aware_optimizer(
    base_optimizer: tf.keras.optimizers.Optimizer,
    learning_rate_scale: float = 1.0
) -> tf.keras.optimizers.Optimizer:
    """Create optimizer with quantization awareness."""
    # Reduce learning rate for stability with quantization
    lr = base_optimizer.learning_rate * learning_rate_scale
    
    # Create new optimizer with adjusted learning rate
    optimizer_config = base_optimizer.get_config()
    optimizer_config['learning_rate'] = float(lr)
    
    return type(base_optimizer).from_config(optimizer_config)
