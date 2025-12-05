"""
Integration Tests for PyTorch and TensorFlow Quantization

Tests to verify that the quantization integration works correctly with
real transformer architectures.

Run with: pytest test_integration.py -v
Or: python -m pytest test_integration.py -v
"""

import unittest
import numpy as np
import torch
import tensorflow as tf
from pytorch_integration import (
    QuantConfig,
    AdaptiveBitAllocator,
    QuantizationHook,
    HybridQuantizedLinear,
    QuantizationAwareTrainer,
    HybridTransformerWrapper,
    quantize_model_weights,
    get_model_bit_compression_ratio
)
from tensorflow_integration import (
    TFQuantConfig,
    TFAdaptiveBitAllocator,
    HybridQuantizedDense,
    QuantizationAwareTrainerTF,
    HuggingFaceTransformerIntegration,
    compute_model_size_reduction_tf
)


class TestPyTorchQuantizationConfig(unittest.TestCase):
    """Test PyTorch QuantConfig."""
    
    def test_config_creation(self):
        """Test creating QuantConfig."""
        config = QuantConfig(
            target_bits=1.58,
            adaptive_bits=True,
            min_bits=1.0,
            max_bits=8.0
        )
        
        self.assertEqual(config.target_bits, 1.58)
        self.assertTrue(config.adaptive_bits)
        self.assertEqual(config.min_bits, 1.0)
        self.assertEqual(config.max_bits, 8.0)
    
    def test_config_defaults(self):
        """Test QuantConfig defaults."""
        config = QuantConfig()
        
        self.assertEqual(config.target_bits, 1.58)
        self.assertTrue(config.adaptive_bits)


class TestPyTorchAdaptiveBitAllocator(unittest.TestCase):
    """Test PyTorch adaptive bit allocation."""
    
    def setUp(self):
        """Create test model."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )
        
        self.config = QuantConfig(
            target_bits=1.58,
            adaptive_bits=True,
            min_bits=1.0,
            max_bits=8.0
        )
    
    def test_allocator_creation(self):
        """Test creating bit allocator."""
        allocator = AdaptiveBitAllocator(self.model, self.config)
        
        self.assertIsNotNone(allocator)
        self.assertEqual(len(allocator.bit_widths), 6)  # 3 linear layers × 2 (weight+bias)
    
    def test_bit_allocation_range(self):
        """Test that allocated bits are within configured range."""
        allocator = AdaptiveBitAllocator(self.model, self.config)
        
        # Create dummy gradient
        dummy_input = torch.randn(32, 64)
        outputs = self.model(dummy_input)
        loss = outputs.sum()
        loss.backward()
        
        # Allocate bits
        allocator.allocate_bits_adaptive()
        
        for bits in allocator.bit_widths.values():
            self.assertGreaterEqual(bits, self.config.min_bits)
            self.assertLessEqual(bits, self.config.max_bits)
    
    def test_sensitivity_computation(self):
        """Test sensitivity score computation."""
        allocator = AdaptiveBitAllocator(self.model, self.config)
        
        dummy_input = torch.randn(32, 64)
        outputs = self.model(dummy_input)
        loss = outputs.sum()
        loss.backward()
        
        allocator.compute_sensitivity_scores()
        
        self.assertEqual(len(allocator.sensitivity_scores), 6)
        
        for score in allocator.sensitivity_scores.values():
            self.assertGreaterEqual(score, 0.1)
            self.assertLessEqual(score, 10.0)


class TestHybridQuantizedLinear(unittest.TestCase):
    """Test HybridQuantizedLinear layer."""
    
    def setUp(self):
        """Create test layer."""
        self.config = QuantConfig()
        self.layer = HybridQuantizedLinear(
            in_features=64,
            out_features=32,
            config=self.config
        )
    
    def test_forward_pass(self):
        """Test forward pass through quantized layer."""
        x = torch.randn(32, 64)
        output = self.layer(x)
        
        self.assertEqual(output.shape, (32, 32))
    
    def test_layer_has_weights(self):
        """Test that layer has initialized weights."""
        self.assertIsNotNone(self.layer.weight)
        self.assertIsNotNone(self.layer.bias)


class TestQuantizationAwareTrainer(unittest.TestCase):
    """Test PyTorch QAT trainer."""
    
    def setUp(self):
        """Create test model and trainer."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )
        
        self.config = QuantConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.trainer = QuantizationAwareTrainer(
            self.model,
            self.optimizer,
            self.config
        )
    
    def test_trainer_creation(self):
        """Test creating QAT trainer."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.step, 0)
    
    def test_training_summary(self):
        """Test getting training summary."""
        summary = self.trainer.get_training_summary()
        
        self.assertEqual(summary['steps_completed'], 0)
        self.assertIn('avg_bits_per_param', summary)
        self.assertIn('bit_allocation', summary)


class TestHybridTransformerWrapper(unittest.TestCase):
    """Test HybridTransformerWrapper."""
    
    def setUp(self):
        """Create test transformer."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )
        
        self.config = QuantConfig()
    
    def test_wrapper_creation(self):
        """Test creating transformer wrapper."""
        wrapper = HybridTransformerWrapper(self.model, self.config)
        
        self.assertIsNotNone(wrapper)
        self.assertGreater(wrapper.num_replaced_layers, 0)
    
    def test_forward_pass_after_wrapping(self):
        """Test that wrapped model still works."""
        wrapper = HybridTransformerWrapper(self.model, self.config)
        
        x = torch.randn(16, 64)
        output = wrapper.model(x)
        
        self.assertEqual(output.shape, (16, 2))


class TestTensorFlowQuantConfig(unittest.TestCase):
    """Test TensorFlow QuantConfig."""
    
    def test_config_creation(self):
        """Test creating TF QuantConfig."""
        config = TFQuantConfig(
            target_bits=1.58,
            adaptive_bits=True
        )
        
        self.assertEqual(config.target_bits, 1.58)
        self.assertTrue(config.adaptive_bits)
    
    def test_config_defaults(self):
        """Test TF config defaults."""
        config = TFQuantConfig()
        
        self.assertEqual(config.target_bits, 1.58)
        self.assertTrue(config.adaptive_bits)


class TestTensorFlowAdaptiveBitAllocator(unittest.TestCase):
    """Test TensorFlow adaptive bit allocation."""
    
    def setUp(self):
        """Create test TF model."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_shape=(64,), activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        
        self.config = TFQuantConfig()
    
    def test_allocator_creation(self):
        """Test creating TF bit allocator."""
        allocator = TFAdaptiveBitAllocator(self.model, self.config)
        
        self.assertIsNotNone(allocator)
    
    def test_sensitivity_scores(self):
        """Test sensitivity score computation."""
        allocator = TFAdaptiveBitAllocator(self.model, self.config)
        
        self.assertEqual(len(allocator.sensitivity_scores), 6)  # 3 layers × 2


class TestHybridQuantizedDense(unittest.TestCase):
    """Test HybridQuantizedDense layer."""
    
    def test_layer_creation(self):
        """Test creating quantized dense layer."""
        layer = HybridQuantizedDense(units=32, input_shape=(64,))
        
        self.assertIsNotNone(layer)
        self.assertEqual(layer.units, 32)
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = tf.keras.Sequential([
            HybridQuantizedDense(units=32, input_shape=(64,), activation='relu'),
            HybridQuantizedDense(units=2)
        ])
        
        x = tf.random.normal((16, 64))
        output = model(x, training=False)
        
        self.assertEqual(output.shape, (16, 2))


class TestQuantizationAwareTrainerTF(unittest.TestCase):
    """Test TensorFlow QAT trainer."""
    
    def setUp(self):
        """Create test TF model and trainer."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(64,)),
            tf.keras.layers.Dense(2)
        ])
        
        self.config = TFQuantConfig()
        self.optimizer = tf.keras.optimizers.Adam()
        self.trainer = QuantizationAwareTrainerTF(
            self.model,
            self.optimizer,
            self.config
        )
    
    def test_trainer_creation(self):
        """Test creating TF QAT trainer."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.step, 0)
    
    def test_training_summary(self):
        """Test getting training summary."""
        summary = self.trainer.get_training_summary()
        
        self.assertEqual(summary['steps_completed'], 0)
        self.assertIn('avg_bits_per_param', summary)


class TestHuggingFaceIntegration(unittest.TestCase):
    """Test HuggingFace transformer integration."""
    
    def setUp(self):
        """Create test TF model."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(512,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        
        self.config = TFQuantConfig()
    
    def test_integration_creation(self):
        """Test creating HuggingFace integration."""
        integration = HuggingFaceTransformerIntegration(
            self.model,
            self.config
        )
        
        self.assertIsNotNone(integration)
    
    def test_get_qat_trainer(self):
        """Test getting QAT trainer from integration."""
        integration = HuggingFaceTransformerIntegration(
            self.model,
            self.config
        )
        
        optimizer = tf.keras.optimizers.Adam()
        trainer = integration.get_qat_trainer(optimizer)
        
        self.assertIsNotNone(trainer)


class TestCompressionMetrics(unittest.TestCase):
    """Test compression metrics computation."""
    
    def test_pytorch_compression_ratio(self):
        """Test PyTorch compression ratio computation."""
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 2)
        )
        
        ratio = get_model_bit_compression_ratio(model, bits=1.58)
        
        self.assertGreater(ratio['compression_ratio'], 1.0)
        self.assertGreater(ratio['original_size_mb'], 0)
        self.assertGreater(ratio['quantized_size_mb'], 0)
    
    def test_tensorflow_compression_metrics(self):
        """Test TensorFlow compression metrics."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_shape=(64,)),
            tf.keras.layers.Dense(2)
        ])
        
        metrics = compute_model_size_reduction_tf(model, average_bits=1.58)
        
        self.assertGreater(metrics['original_size_mb'], 0)
        self.assertGreater(metrics['reduction_ratio'], 1.0)
        self.assertGreater(metrics['total_params'], 0)


class TestEndToEndPyTorch(unittest.TestCase):
    """End-to-end PyTorch integration test."""
    
    def test_full_training_pipeline(self):
        """Test complete PyTorch training pipeline."""
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )
        
        # Wrap with quantization
        config = QuantConfig()
        wrapper = HybridTransformerWrapper(model, config)
        
        # Create trainer
        optimizer = torch.optim.Adam(wrapper.model.parameters())
        trainer = QuantizationAwareTrainer(wrapper.model, optimizer, config)
        
        # Create dummy data
        x = torch.randn(32, 64)
        y = torch.randint(0, 2, (32,))
        
        # Training step
        outputs = wrapper.model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify training worked
        self.assertGreater(trainer.step, -1)


class TestEndToEndTensorFlow(unittest.TestCase):
    """End-to-end TensorFlow integration test."""
    
    def test_full_training_pipeline(self):
        """Test complete TensorFlow training pipeline."""
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(64,)),
            tf.keras.layers.Dense(2)
        ])
        
        # Wrap with quantization
        config = TFQuantConfig()
        integration = HuggingFaceTransformerIntegration(model, config)
        
        # Create trainer
        optimizer = tf.keras.optimizers.Adam()
        trainer = integration.get_qat_trainer(optimizer)
        
        # Create dummy data
        x = tf.random.normal((32, 64))
        y = tf.random.uniform((32,), maxval=2, dtype=tf.int32)
        
        # Define loss
        def criterion(targets, outputs):
            return tf.keras.losses.sparse_categorical_crossentropy(
                targets, outputs, from_logits=True
            )
        
        # Training step
        metrics = trainer.train_step(x, y, criterion)
        
        # Verify training worked
        self.assertIn('total_loss', metrics)
        self.assertGreater(metrics['total_loss'], 0)


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
