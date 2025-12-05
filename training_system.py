"""
1.58-Bit Hybrid LLM Training System

Complete training framework integrating:
- Clustering (data and parameter partitioning)
- Auxiliary NN (dynamic weight prediction)
- Constrained Optimization (trust region + quantization)
- Shortest Path/Routing (optimization trajectory)

This system implements the mathematical formulation:
  Θ* = arg min E[(x,y)∈D][L(M(x;Θ),y)]
  
  With hybrid optimization:
  1. Cluster D into {C_1, ..., C_K}
  2. Predict ŵ_t = N_aux(Features; Φ)
  3. Update: ΔΘ^t = arg min[L] s.t. ||ΔΘ||_2 ≤ ŵ_t^2
  4. Quantize: Θ_q = Quantize(Θ, 1.58-bit)
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
import time

from quantization import Quantizer158Bit, AdaptiveQuantizer
from clustering import KMeansClustering, DataClustering, ParameterClustering
from auxiliary_nn import AuxiliaryNN, TrainingState, AdaptiveOptimizer
from constrained_optimization import (
    ConstrainedOptimizationStep,
    AdaptiveConstrainedOptimizer,
    ConstrainedUpdateInfo
)


@dataclass
class TrainingConfig:
    """Configuration for 1.58-bit LLM training."""
    # Quantization
    use_quantization: bool = True
    quantizer_scale: float = 1.0
    
    # Clustering
    data_clusters: int = 8
    parameter_clusters: int = 4
    use_clustering: bool = True
    
    # Optimization
    base_learning_rate: float = 0.001
    max_steps: int = 1000
    batch_size: int = 32
    use_adaptive_lr: bool = True
    
    # Meta-learning (auxiliary NN)
    use_auxiliary_nn: bool = True
    auxiliary_nn_hidden_size: int = 16
    
    # Constraints
    use_trust_region: bool = True
    initial_trust_radius: float = 0.01
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 50


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    step: int = 0
    loss: float = 0.0
    loss_reduction: float = 0.0
    gradient_magnitude: float = 0.0
    update_magnitude: float = 0.0
    quantization_error: float = 0.0
    constraint_active: bool = False
    learning_rate: float = 0.0
    time_elapsed: float = 0.0
    
    metrics_history: Dict[str, List] = field(default_factory=dict)
    
    def record(self, key: str, value: float):
        """Record a metric value."""
        if key not in self.metrics_history:
            self.metrics_history[key] = []
        self.metrics_history[key].append(value)
    
    def to_dict(self) -> Dict:
        """Convert current metrics to dictionary."""
        return {
            'step': self.step,
            'loss': self.loss,
            'loss_reduction': self.loss_reduction,
            'gradient_magnitude': self.gradient_magnitude,
            'update_magnitude': self.update_magnitude,
            'quantization_error': self.quantization_error,
            'constraint_active': self.constraint_active,
            'learning_rate': self.learning_rate,
            'time_elapsed': self.time_elapsed,
        }


class HybridLLMTrainer:
    """
    Complete 1.58-bit LLM training system combining all hybrid components.
    """
    
    def __init__(self,
                 model_dim: int = 768,
                 num_layers: int = 12,
                 config: Optional[TrainingConfig] = None):
        """
        Initialize hybrid LLM trainer.
        
        Args:
            model_dim: Model dimension / hidden size
            num_layers: Number of model layers
            config: Training configuration
        """
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.config = config or TrainingConfig()
        
        # Initialize components
        self._init_quantization()
        self._init_clustering()
        self._init_optimization()
        
        # Training state - parameters match input dimension to output dimension
        # parameters shape: (input_dim, output_dim) 
        self.parameters = np.random.randn(model_dim, num_layers) * 0.01
        self.metrics = TrainingMetrics()
        self.training_start_time = None
        
    def _init_quantization(self):
        """Initialize quantization components."""
        self.quantizer = Quantizer158Bit(scale=self.config.quantizer_scale)
        self.adaptive_quantizer = AdaptiveQuantizer(self.num_layers)
        
    def _init_clustering(self):
        """Initialize clustering components."""
        if self.config.use_clustering:
            self.data_clustering = DataClustering(n_clusters=self.config.data_clusters)
            self.param_clustering = ParameterClustering(n_clusters=self.config.parameter_clusters)
        else:
            self.data_clustering = None
            self.param_clustering = None
    
    def _init_optimization(self):
        """Initialize optimization components."""
        self.adaptive_optimizer = AdaptiveConstrainedOptimizer(
            base_learning_rate=self.config.base_learning_rate,
            quantizer=self.quantizer,
            use_quantization=self.config.use_quantization,
            use_adaptive_optimizer=self.config.use_adaptive_lr
        )
        
    def prepare_training_data(self,
                            data_embeddings: np.ndarray) -> Dict:
        """
        Prepare training data by clustering.
        
        Args:
            data_embeddings: Input embeddings of shape (n_samples, embedding_dim)
            
        Returns:
            Dictionary with cluster information
        """
        if not self.config.use_clustering or self.data_clustering is None:
            return {'clusters': [np.arange(len(data_embeddings))]}
        
        cluster_labels, cluster_infos = self.data_clustering.cluster_embeddings(data_embeddings)
        clusters = self.data_clustering.get_balanced_clusters()
        
        return {
            'clusters': clusters,
            'cluster_infos': cluster_infos,
            'strategy': self.data_clustering.suggest_mini_batch_strategy()
        }
    
    def compute_batch_gradients(self,
                               batch_data: np.ndarray,
                               batch_targets: np.ndarray,
                               loss_fn: Callable) -> np.ndarray:
        """
        Compute gradients for a batch (simplified forward-backward pass).
        
        Args:
            batch_data: Batch input data
            batch_targets: Batch target data
            loss_fn: Loss function
            
        Returns:
            Gradients with respect to parameters
        """
        # Simplified forward pass: linear combination
        predictions = batch_data @ self.parameters
        
        # Compute loss
        batch_loss = loss_fn(predictions, batch_targets)
        
        # Simplified backward pass: analytical gradient
        # For MSE loss: ∇L = 2 * X^T * (Xw - y)
        residual = predictions - batch_targets
        batch_gradients = 2.0 * batch_data.T @ residual
        
        return batch_gradients, batch_loss
    
    def create_training_state(self,
                             loss: float,
                             gradients: np.ndarray,
                             cluster_id: int = 0,
                             previous_loss: Optional[float] = None) -> TrainingState:
        """
        Create training state for auxiliary NN.
        
        Args:
            loss: Current loss
            gradients: Current gradients
            cluster_id: Current cluster ID
            previous_loss: Previous loss value
            
        Returns:
            TrainingState object
        """
        grad_magnitude = np.linalg.norm(gradients)
        grad_variance = np.var(gradients)
        
        # Loss trend: ratio to previous loss
        if previous_loss is not None and previous_loss > 0:
            loss_trend = loss / previous_loss
        else:
            loss_trend = 1.0
        
        return TrainingState(
            current_loss=loss,
            gradient_magnitude=grad_magnitude,
            cluster_id=cluster_id,
            loss_trend=loss_trend,
            gradient_variance=grad_variance,
            step_number=self.metrics.step
        )
    
    def train_step(self,
                   batch_data: np.ndarray,
                   batch_targets: np.ndarray,
                   loss_fn: Callable,
                   cluster_id: int = 0) -> TrainingMetrics:
        """
        Perform one training step with hybrid optimization.
        
        Args:
            batch_data: Input batch
            batch_targets: Target batch
            loss_fn: Loss function
            cluster_id: Cluster ID for state tracking
            
        Returns:
            Updated metrics
        """
        # Compute gradients
        gradients, current_loss = self.compute_batch_gradients(
            batch_data, batch_targets, loss_fn
        )
        
        # Create training state
        previous_loss = self.metrics.loss if self.metrics.step > 0 else None
        state = self.create_training_state(
            current_loss, gradients, cluster_id, previous_loss
        )
        
        # Perform adaptive constrained optimization step
        update_info = self.adaptive_optimizer.step(
            self.parameters, gradients, state, current_loss
        )
        
        # Update parameters
        self.parameters = update_info.parameters_updated
        
        # Update metrics
        self.metrics.step += 1
        loss_reduction = (previous_loss - current_loss) if previous_loss else 0.0
        
        self.metrics.loss = current_loss
        self.metrics.loss_reduction = loss_reduction
        self.metrics.gradient_magnitude = state.gradient_magnitude
        self.metrics.update_magnitude = update_info.update_magnitude
        self.metrics.quantization_error = update_info.quantization_error
        self.metrics.constraint_active = update_info.constraint_active
        self.metrics.learning_rate = self.config.base_learning_rate
        
        # Calculate elapsed time
        if self.training_start_time:
            self.metrics.time_elapsed = time.time() - self.training_start_time
        
        # Record metrics
        for key in ['loss', 'loss_reduction', 'gradient_magnitude', 'update_magnitude', 'quantization_error']:
            self.metrics.record(key, getattr(self.metrics, key))
        
        return self.metrics
    
    def train(self,
              training_data: np.ndarray,
              training_targets: np.ndarray,
              loss_fn: Callable = None,
              eval_fn: Optional[Callable] = None) -> TrainingMetrics:
        """
        Full training loop with hybrid optimization.
        
        Args:
            training_data: Training input data
            training_targets: Training target data
            loss_fn: Loss function (default: MSE)
            eval_fn: Evaluation function (optional)
            
        Returns:
            Final metrics
        """
        # Default loss function: MSE
        if loss_fn is None:
            loss_fn = lambda y_pred, y_true: np.mean((y_pred - y_true) ** 2)
        
        self.training_start_time = time.time()
        
        # Prepare data with clustering
        data_prep = self.prepare_training_data(training_data)
        clusters = data_prep['clusters']
        
        # Training loop
        print(f"Starting hybrid LLM training with 1.58-bit quantization")
        print(f"  Model: {self.model_dim}D x {self.num_layers} layers")
        print(f"  Data clusters: {len(clusters)}")
        print(f"  Max steps: {self.config.max_steps}")
        print()
        
        for step in range(self.config.max_steps):
            # Select cluster for this step
            cluster_id = step % len(clusters)
            cluster_indices = clusters[cluster_id]
            
            # Create mini-batch from cluster
            if len(cluster_indices) > self.config.batch_size:
                batch_indices = np.random.choice(
                    cluster_indices, self.config.batch_size, replace=False
                )
            else:
                batch_indices = cluster_indices
            
            batch_data = training_data[batch_indices]
            batch_targets = training_targets[batch_indices]
            
            # Training step
            metrics = self.train_step(batch_data, batch_targets, loss_fn, cluster_id)
            
            # Logging
            if (step + 1) % self.config.log_interval == 0:
                print(f"Step {step+1}/{self.config.max_steps} | "
                      f"Loss: {metrics.loss:.6f} | "
                      f"dLoss: {metrics.loss_reduction:.6f} | "
                      f"|grad|: {metrics.gradient_magnitude:.4f} | "
                      f"|Delta|: {metrics.update_magnitude:.6f} | "
                      f"Constraint: {int(metrics.constraint_active)}")
            
            # Evaluation
            if eval_fn and (step + 1) % self.config.eval_interval == 0:
                eval_loss = eval_fn(self.parameters)
                print(f"  → Evaluation loss: {eval_loss:.6f}")
        
        print(f"\nTraining complete. Final loss: {self.metrics.loss:.6f}")
        print(f"Total time: {self.metrics.time_elapsed:.2f}s")
        
        return self.metrics
    
    def get_training_summary(self) -> Dict:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with training statistics
        """
        summary = {
            'total_steps': self.metrics.step,
            'final_loss': self.metrics.loss,
            'total_time': self.metrics.time_elapsed,
            'optimizer_summary': self.adaptive_optimizer.get_training_summary(),
            'metrics_history': self.metrics.metrics_history,
        }
        
        if self.metrics.metrics_history.get('loss'):
            losses = self.metrics.metrics_history['loss']
            summary['initial_loss'] = losses[0]
            summary['total_loss_reduction'] = losses[0] - losses[-1]
            summary['loss_reduction_rate'] = summary['total_loss_reduction'] / self.metrics.step
        
        return summary
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'parameters': self.parameters,
            'step': self.metrics.step,
            'loss': self.metrics.loss,
            'metrics_history': self.metrics.metrics_history,
        }
        np.save(filepath, checkpoint, allow_pickle=True)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = np.load(filepath, allow_pickle=True).item()
        self.parameters = checkpoint['parameters']
        self.metrics.step = checkpoint['step']
        self.metrics.loss = checkpoint['loss']
        self.metrics.metrics_history = checkpoint['metrics_history']
        print(f"Checkpoint loaded from {filepath}")


if __name__ == "__main__":
    print("=" * 80)
    print("1.58-BIT HYBRID LLM TRAINING SYSTEM")
    print("=" * 80)
    
    # Create configuration
    config = TrainingConfig(
        use_quantization=True,
        use_clustering=True,
        data_clusters=4,
        parameter_clusters=3,
        use_adaptive_lr=True,
        base_learning_rate=0.01,
        max_steps=200,
        batch_size=32,
        log_interval=20,
        eval_interval=50,
    )
    
    # Create trainer
    trainer = HybridLLMTrainer(
        model_dim=128,
        num_layers=4,
        config=config
    )
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 512
    input_dim = 64
    
    training_data = np.random.randn(n_samples, input_dim)
    # Generate targets as linear combination + noise
    true_weights = np.random.randn(input_dim, 4) * 0.1
    training_targets = training_data @ true_weights + np.random.randn(n_samples, 4) * 0.01
    
    # Define loss function
    def mse_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    # Train
    final_metrics = trainer.train(
        training_data,
        training_targets,
        loss_fn=mse_loss
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    summary = trainer.get_training_summary()
    print(f"Total steps: {summary['total_steps']}")
    print(f"Initial loss: {summary.get('initial_loss', 'N/A'):.6f}")
    print(f"Final loss: {summary['final_loss']:.6f}")
    print(f"Total loss reduction: {summary.get('total_loss_reduction', 0):.6f}")
    print(f"Total time: {summary['total_time']:.2f}s")
    
    opt_summary = summary['optimizer_summary']
    print(f"\nOptimizer Statistics:")
    print(f"  Constraints active: {opt_summary['constraints_active']}/{opt_summary['total_steps']}")
    print(f"  Activation rate: {opt_summary['constraint_activation_rate']:.2%}")
    print(f"  Mean update magnitude: {opt_summary['mean_update_magnitude']:.6f}")
    print(f"  Mean quantization error: {opt_summary['mean_quantization_error']:.6f}")
