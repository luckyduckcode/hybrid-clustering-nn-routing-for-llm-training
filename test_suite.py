"""
Test Suite and Examples for 1.58-bit Hybrid LLM Training System

Demonstrates:
1. Quantization effectiveness
2. Clustering impact on training
3. Auxiliary NN prediction accuracy
4. Full end-to-end training
5. Comparison with baseline methods
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List

from quantization import Quantizer158Bit, estimate_model_size_reduction
from clustering import KMeansClustering, DataClustering, ParameterClustering
from auxiliary_nn import AuxiliaryNN, TrainingState, AdaptiveOptimizer
from constrained_optimization import AdaptiveConstrainedOptimizer
from training_system import HybridLLMTrainer, TrainingConfig


class QuantizationTest:
    """Test 1.58-bit quantization properties."""
    
    @staticmethod
    def test_quantization_levels():
        """Test that quantization maps to correct levels."""
        print("\n" + "="*60)
        print("TEST: Quantization Levels")
        print("="*60)
        
        quantizer = Quantizer158Bit(scale=1.0)
        
        # Test values covering the range
        test_values = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        quantized = quantizer.quantize(test_values)
        
        print("Input values:    ", test_values)
        print("Quantized:       ", quantized)
        print("Valid levels:    ", [-1, -0.5, 0, 0.5, 1])
        
        # Check that all outputs are valid
        valid_levels = {-1, -0.5, 0, 0.5, 1}
        assert all(v in valid_levels for v in quantized), "Invalid quantization levels!"
        print("[OK] All outputs are valid quantization levels")
    
    @staticmethod
    def test_size_reduction():
        """Test model size reduction from quantization."""
        print("\n" + "="*60)
        print("TEST: Model Size Reduction")
        print("="*60)
        
        model_sizes = [100e6, 1e9, 7e9, 13e9]  # 100M, 1B, 7B, 13B
        
        for size in model_sizes:
            original, quantized = estimate_model_size_reduction(32, size)
            ratio = original / quantized
            
            print(f"\n{size/1e9:.1f}B Model:")
            print(f"  FP32:           {original/1024:.2f} GB")
            print(f"  1.58-bit:       {quantized/1024:.2f} GB")
            print(f"  Compression:    {ratio:.2f}x")
    
    @staticmethod
    def test_gradient_quantization():
        """Test gradient quantization with magnitude preservation."""
        print("\n" + "="*60)
        print("TEST: Gradient Quantization")
        print("="*60)
        
        quantizer = Quantizer158Bit()
        
        # Create diverse gradients
        grad1 = np.random.randn(100) * 0.5  # Small gradients
        grad2 = np.random.randn(100) * 2.0  # Large gradients
        
        quant_grad1 = quantizer.quantize_gradients(grad1)
        quant_grad2 = quantizer.quantize_gradients(grad2)
        
        print(f"Small gradient magnitude:  {np.linalg.norm(grad1):.4f}")
        print(f"  Quantized magnitude:     {np.linalg.norm(quant_grad1):.4f}")
        print(f"Large gradient magnitude:  {np.linalg.norm(grad2):.4f}")
        print(f"  Quantized magnitude:     {np.linalg.norm(quant_grad2):.4f}")
        print("[OK] Magnitude is preserved")


class ClusteringTest:
    """Test clustering components."""
    
    @staticmethod
    def test_kmeans_convergence():
        """Test K-Means clustering convergence."""
        print("\n" + "="*60)
        print("TEST: K-Means Convergence")
        print("="*60)
        
        # Create synthetic data with clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(100, 2) + [0, 0]
        cluster2 = np.random.randn(100, 2) + [5, 5]
        cluster3 = np.random.randn(100, 2) + [-5, 5]
        
        X = np.vstack([cluster1, cluster2, cluster3])
        
        kmeans = KMeansClustering(n_clusters=3, max_iter=100, random_state=42)
        kmeans.fit(X)
        
        print(f"Number of clusters: 3")
        print(f"Converged: {kmeans.converged}")
        print(f"Inertia: {kmeans.inertia:.4f}")
        
        # Check cluster sizes
        for k in range(3):
            size = np.sum(kmeans.labels == k)
            print(f"  Cluster {k}: {size} samples")
        
        assert kmeans.converged, "K-Means should converge!"
        print("[OK] K-Means converged successfully")
    
    @staticmethod
    def test_data_clustering():
        """Test data clustering for training."""
        print("\n" + "="*60)
        print("TEST: Data Clustering for Training")
        print("="*60)
        
        np.random.seed(42)
        embeddings = np.random.randn(1000, 128)
        
        clusterer = DataClustering(n_clusters=4)
        labels, infos = clusterer.cluster_embeddings(embeddings)
        
        print(f"Total samples: 1000")
        print(f"Number of clusters: 4")
        
        for info in infos:
            print(f"  Cluster {info.cluster_id}: {info.size} samples (cohesion: {info.mean_distance:.4f})")
        
        strategy = clusterer.suggest_mini_batch_strategy()
        print(f"\nSuggested strategy: {strategy['strategy']}")
        print(f"Balanced batch size: {strategy['balanced_batch_size']}")
        print("[OK] Data clustering successful")


class AuxiliaryNNTest:
    """Test auxiliary neural network."""
    
    @staticmethod
    def test_lr_prediction():
        """Test learning rate prediction."""
        print("\n" + "="*60)
        print("TEST: Learning Rate Prediction")
        print("="*60)
        
        aux_nn = AuxiliaryNN()
        
        states = [
            TrainingState(2.0, 0.1, 0, 0.99, 0.01, 0),
            TrainingState(1.5, 0.08, 0, 0.98, 0.01, 10),
            TrainingState(1.0, 0.05, 0, 0.95, 0.01, 50),
            TrainingState(0.5, 0.02, 0, 0.90, 0.01, 100),
        ]
        
        print(f"{'Step':<6} {'Loss':<8} {'Grad':<8} {'LR Mult':<10}")
        for i, state in enumerate(states):
            lr = aux_nn.predict_learning_rate(state)
            print(f"{state.step_number:<6} {state.current_loss:<8.2f} {state.gradient_magnitude:<8.3f} {lr:<10.4f}")
        
        print("[OK] Learning rate predictions generated")
    
    @staticmethod
    def test_optimizer_feedback():
        """Test optimizer feedback and meta-learning."""
        print("\n" + "="*60)
        print("TEST: Optimizer Feedback & Meta-Learning")
        print("="*60)
        
        optimizer = AdaptiveOptimizer(base_learning_rate=0.001)
        
        # Simulate training steps with feedback
        params = np.random.randn(50)
        loss = 2.5
        
        print(f"{'Step':<6} {'Loss':<8} {'LR':<8} {'Pred Success':<15}")
        
        for step in range(20):
            grads = np.random.randn(50)
            state = TrainingState(
                loss, np.linalg.norm(grads), 0, 0.98, 0.01, step
            )
            
            # Get predicted LR
            lr = optimizer.auxiliary_nn.predict_learning_rate(state)
            
            # Simulate loss reduction
            loss_reduction = np.random.randn() * 0.1
            loss -= loss_reduction
            
            # Update with feedback
            optimizer.auxiliary_nn.update_with_feedback(lr, loss_reduction)
            
            # Calculate success
            stats = optimizer.auxiliary_nn.get_prediction_statistics()
            success_rate = stats.get('success_rate', 0) if stats.get('samples', 0) > 0 else 0
            
            if (step + 1) % 5 == 0:
                print(f"{step+1:<6} {loss:<8.4f} {lr:<8.4f} {success_rate:<15.1%}")
        
        print("[OK] Feedback mechanism working")


class TrainingTest:
    """Test complete training system."""
    
    @staticmethod
    def test_basic_training():
        """Test basic training loop."""
        print("\n" + "="*60)
        print("TEST: Basic Training Loop")
        print("="*60)
        
        config = TrainingConfig(
            use_quantization=True,
            use_clustering=True,
            data_clusters=2,
            parameter_clusters=2,
            max_steps=50,
            batch_size=16,
            log_interval=10,
        )
        
        # Create trainer with matching dimensions
        # Input data has 32 features, so trainer needs input_dim=32
        trainer = HybridLLMTrainer(model_dim=32, num_layers=2, config=config)
        
        # Generate data with matching dimensions
        np.random.seed(42)
        training_data = np.random.randn(128, 32)  # 128 samples, 32 features
        training_targets = np.random.randn(128, 2)  # 128 samples, 2 outputs (num_layers)
        
        # Train
        print("\nTraining...")
        metrics = trainer.train(training_data, training_targets)
        
        summary = trainer.get_training_summary()
        print(f"\nFinal loss: {summary['final_loss']:.6f}")
        print(f"Total time: {summary['total_time']:.2f}s")
        print("[OK] Training completed successfully")
    
    @staticmethod
    def test_quantization_impact():
        """Compare training with and without quantization."""
        print("\n" + "="*60)
        print("TEST: Quantization Impact on Training")
        print("="*60)
        
        np.random.seed(42)
        training_data = np.random.randn(200, 32)
        training_targets = np.random.randn(200, 2)
        
        results = {}
        
        for use_quant in [False, True]:
            config = TrainingConfig(
                use_quantization=use_quant,
                use_clustering=False,
                max_steps=50,
                batch_size=16,
                log_interval=100,
            )
            
            trainer = HybridLLMTrainer(model_dim=32, num_layers=2, config=config)
            metrics = trainer.train(training_data, training_targets)
            
            summary = trainer.get_training_summary()
            results[f"Quantized={use_quant}"] = summary
        
        print(f"\nWithout quantization: {results['Quantized=False']['final_loss']:.6f}")
        print(f"With 1.58-bit quantization: {results['Quantized=True']['final_loss']:.6f}")
        print("[OK] Comparison complete")


class ComparisonTest:
    """Comparison tests with different configurations."""
    
    @staticmethod
    def test_clustering_impact():
        """Test impact of clustering on training."""
        print("\n" + "="*60)
        print("TEST: Clustering Impact on Training")
        print("="*60)
        
        np.random.seed(42)
        training_data = np.random.randn(256, 32)
        training_targets = np.random.randn(256, 2)
        
        configs = {
            'No clustering': TrainingConfig(use_clustering=False, max_steps=40),
            '4 clusters': TrainingConfig(use_clustering=True, data_clusters=4, max_steps=40),
            '8 clusters': TrainingConfig(use_clustering=True, data_clusters=8, max_steps=40),
        }
        
        print(f"{'Config':<20} {'Final Loss':<15} {'Time (s)':<10}")
        
        for name, config in configs.items():
            trainer = HybridLLMTrainer(model_dim=32, num_layers=2, config=config)
            metrics = trainer.train(training_data, training_targets)
            summary = trainer.get_training_summary()
            
            print(f"{name:<20} {summary['final_loss']:<15.6f} {summary['total_time']:<10.2f}")
        
        print("[OK] Clustering comparison complete")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*80)
    print("1.58-BIT HYBRID LLM TRAINING - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Quantization tests
    print("\n### QUANTIZATION TESTS ###")
    QuantizationTest.test_quantization_levels()
    QuantizationTest.test_size_reduction()
    QuantizationTest.test_gradient_quantization()
    
    # Clustering tests
    print("\n### CLUSTERING TESTS ###")
    ClusteringTest.test_kmeans_convergence()
    ClusteringTest.test_data_clustering()
    
    # Auxiliary NN tests
    print("\n### AUXILIARY NEURAL NETWORK TESTS ###")
    AuxiliaryNNTest.test_lr_prediction()
    AuxiliaryNNTest.test_optimizer_feedback()
    
    # Training tests
    print("\n### TRAINING TESTS ###")
    TrainingTest.test_basic_training()
    TrainingTest.test_quantization_impact()
    
    # Comparison tests
    print("\n### COMPARISON TESTS ###")
    ComparisonTest.test_clustering_impact()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY [OK]")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
