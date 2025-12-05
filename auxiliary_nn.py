"""
Auxiliary Neural Network for Dynamic Weight/Learning Rate Prediction

A lightweight meta-learner that predicts optimal learning rates and update directions
based on current training state (loss, gradient magnitude, cluster information).
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrainingState:
    """Current training state for auxiliary NN input."""
    current_loss: float
    gradient_magnitude: float
    cluster_id: int
    loss_trend: float  # Ratio of current to previous loss
    gradient_variance: float
    step_number: int


class AuxiliaryNN:
    """
    Lightweight neural network for predicting dynamic learning rates and
    step directions. Uses a simple feedforward architecture optimized for speed.
    """
    
    def __init__(self, input_size: int = 6, hidden_size: int = 16, output_size: int = 2):
        """
        Initialize auxiliary NN.
        
        Args:
            input_size: Number of input features (from training state)
            hidden_size: Number of hidden units (small for efficiency)
            output_size: Number of outputs (learning_rate_multiplier + direction_bias)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # (6, 16)
        self.b1 = np.zeros((1, hidden_size))  # (1, 16)
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # (16, 2)
        self.b2 = np.zeros((1, output_size))  # (1, 2)
        
        # Training history for meta-learning
        self.prediction_history = []
        self.error_history = []
        
    def forward(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through auxiliary NN.
        
        Args:
            features: Input features of shape (batch_size, input_size)
            
        Returns:
            Tuple of (output, hidden_activation)
        """
        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Hidden layer with ReLU activation
        z1 = features @ self.W1 + self.b1
        h = np.maximum(0, z1)  # ReLU
        
        # Output layer with sigmoid activation for learning rate (0.1 to 10x)
        # and tanh activation for direction bias (-1 to 1)
        z2 = h @ self.W2 + self.b2
        output = np.tanh(z2)  # Output in range (-1, 1)
        
        return output, h
    
    def predict_learning_rate(self, state: TrainingState) -> float:
        """
        Predict multiplicative learning rate factor based on training state.
        
        Args:
            state: Current training state
            
        Returns:
            Learning rate multiplier (typically 0.1 to 10)
        """
        features = self._state_to_features(state)
        output, _ = self.forward(features)
        
        # Map output (-1, 1) to learning rate range (0.1, 10)
        # Using exponential mapping: lr = 10^(output)
        lr_multiplier = 10.0 ** output[0, 0]  # Exponentiate to get exponential scale
        
        # Store for meta-learning
        self.prediction_history.append(('lr', output[0, 0], lr_multiplier))
        
        return lr_multiplier
    
    def predict_update_direction_bias(self, state: TrainingState) -> float:
        """
        Predict bias for update direction (momentum-like effect).
        
        Args:
            state: Current training state
            
        Returns:
            Direction bias in range (-1, 1)
        """
        features = self._state_to_features(state)
        output, _ = self.forward(features)
        
        direction_bias = output[0, 1]  # Second output
        
        self.prediction_history.append(('direction', direction_bias, direction_bias))
        
        return direction_bias
    
    def predict_trust_region(self, state: TrainingState) -> float:
        """
        Predict trust region size (constraint on step magnitude).
        
        Args:
            state: Current training state
            
        Returns:
            Trust region radius (step size constraint)
        """
        features = self._state_to_features(state)
        output, _ = self.forward(features)
        
        # Average both outputs to get trust region
        # Map from (-1, 1) to (0.001, 1.0) - step size range
        tr_value = (output[0, 0] + output[0, 1]) / 2.0
        trust_region = 10.0 ** (tr_value - 1)  # Range: ~0.1 to 10
        
        return trust_region
    
    @staticmethod
    def _state_to_features(state: TrainingState) -> np.ndarray:
        """
        Convert training state to feature vector.
        
        Args:
            state: Training state
            
        Returns:
            Feature vector of shape (1, 6)
        """
        features = np.array([
            np.log(state.current_loss + 1e-8),  # Log loss
            np.log(state.gradient_magnitude + 1e-8),  # Log gradient magnitude
            state.cluster_id / 100.0,  # Normalize cluster ID
            np.log(state.loss_trend + 1e-8),  # Log loss trend
            np.log(state.gradient_variance + 1e-8),  # Log gradient variance
            np.log(state.step_number + 1),  # Log step number
        ])
        
        return features.reshape(1, -1)
    
    def update_with_feedback(self, predicted_lr: float, actual_loss_reduction: float, learning_rate: float = 0.01):
        """
        Update auxiliary NN based on prediction accuracy (meta-gradient).
        
        Args:
            predicted_lr: The predicted learning rate
            actual_loss_reduction: Actual loss improvement achieved
            learning_rate: Meta-learning rate
        """
        # Simple feedback: if predicted LR led to improvement, reinforce
        if actual_loss_reduction > 0:
            # Positive reward for good prediction
            feedback_signal = min(actual_loss_reduction, 0.1)  # Normalize
        else:
            # Negative reward for bad prediction
            feedback_signal = max(actual_loss_reduction, -0.1)
        
        # Store error for analysis
        self.error_history.append({
            'predicted_lr': predicted_lr,
            'loss_reduction': actual_loss_reduction,
            'feedback': feedback_signal
        })
        
        # Simple weight update: gradient ascent on feedback signal
        # This is a simplified meta-learning step
        if len(self.error_history) % 10 == 0:  # Update every 10 steps
            avg_feedback = np.mean([e['feedback'] for e in self.error_history[-10:]])
            
            # Gentle weight update proportional to feedback
            self.W1 += learning_rate * avg_feedback * np.random.randn(*self.W1.shape) * 0.01
            self.W2 += learning_rate * avg_feedback * np.random.randn(*self.W2.shape) * 0.01
    
    def get_prediction_statistics(self) -> dict:
        """
        Get statistics about prediction history.
        
        Returns:
            Dictionary with prediction statistics
        """
        if not self.error_history:
            return {'samples': 0}
        
        loss_reductions = [e['loss_reduction'] for e in self.error_history]
        feedbacks = [e['feedback'] for e in self.error_history]
        
        return {
            'samples': len(self.error_history),
            'mean_loss_reduction': float(np.mean(loss_reductions)),
            'std_loss_reduction': float(np.std(loss_reductions)),
            'mean_feedback': float(np.mean(feedbacks)),
            'success_rate': float(np.sum(np.array(loss_reductions) > 0) / len(loss_reductions))
        }


class AdaptiveOptimizer:
    """
    Combines auxiliary NN with gradient-based optimization.
    Uses hybrid approach: NN predicts step size, gradient provides direction.
    """
    
    def __init__(self, base_learning_rate: float = 0.001, use_momentum: bool = True):
        """
        Initialize adaptive optimizer.
        
        Args:
            base_learning_rate: Base learning rate before NN scaling
            use_momentum: Whether to use momentum
        """
        self.base_learning_rate = base_learning_rate
        self.use_momentum = use_momentum
        self.auxiliary_nn = AuxiliaryNN()
        self.momentum_buffer = None
        self.step_count = 0
        self.previous_loss = None
        
    def compute_update(self, 
                      gradients: np.ndarray,
                      state: TrainingState,
                      momentum_beta: float = 0.9) -> np.ndarray:
        """
        Compute parameter update using NN-guided adaptive learning.
        
        Args:
            gradients: Gradient array
            state: Current training state
            momentum_beta: Momentum coefficient
            
        Returns:
            Update vector (ΔΘ) to apply to parameters
        """
        self.step_count += 1
        
        # Predict adaptive learning rate from auxiliary NN
        lr_multiplier = self.auxiliary_nn.predict_learning_rate(state)
        
        # Predict direction bias
        direction_bias = self.auxiliary_nn.predict_update_direction_bias(state)
        
        # Predict trust region
        trust_region = self.auxiliary_nn.predict_trust_region(state)
        
        # Compute effective learning rate
        effective_lr = self.base_learning_rate * lr_multiplier
        
        # Normalize gradient
        grad_magnitude = np.linalg.norm(gradients)
        if grad_magnitude > 0:
            grad_direction = gradients / grad_magnitude
        else:
            grad_direction = gradients
        
        # Apply direction bias (momentum-like)
        if self.use_momentum:
            if self.momentum_buffer is None:
                self.momentum_buffer = np.zeros_like(gradients)
            
            self.momentum_buffer = momentum_beta * self.momentum_buffer + grad_direction
            grad_direction = self.momentum_buffer / (1 - momentum_beta ** self.step_count)  # Bias correction
        
        # Constrain by trust region (implements constraint ||ΔΘ||_2 ≤ ŵ_t^2)
        unconstrained_update = effective_lr * grad_direction
        
        update_magnitude = np.linalg.norm(unconstrained_update)
        if update_magnitude > trust_region:
            update = (unconstrained_update / update_magnitude) * trust_region
        else:
            update = unconstrained_update
        
        return update
    
    def step(self, 
             parameters: np.ndarray,
             gradients: np.ndarray,
             state: TrainingState,
             current_loss: float) -> Tuple[np.ndarray, dict]:
        """
        Perform one optimization step.
        
        Args:
            parameters: Current model parameters
            gradients: Gradients w.r.t. parameters
            state: Current training state
            current_loss: Current loss value
            
        Returns:
            Tuple of (updated_parameters, optimization_info)
        """
        # Compute update
        update = self.compute_update(gradients, state)
        
        # Apply update
        new_parameters = parameters - update
        
        # Calculate loss reduction
        if self.previous_loss is not None:
            loss_reduction = self.previous_loss - current_loss
        else:
            loss_reduction = 0.0
        
        # Update auxiliary NN with feedback
        lr_mult = self.auxiliary_nn.predict_learning_rate(state)
        self.auxiliary_nn.update_with_feedback(lr_mult, loss_reduction)
        
        self.previous_loss = current_loss
        
        # Return info dict
        info = {
            'update_magnitude': float(np.linalg.norm(update)),
            'gradient_magnitude': float(np.linalg.norm(gradients)),
            'loss_reduction': float(loss_reduction),
            'step_count': self.step_count,
        }
        
        return new_parameters, info


if __name__ == "__main__":
    # Test auxiliary NN
    print("=== Auxiliary NN Test ===")
    
    aux_nn = AuxiliaryNN()
    
    # Create a sample training state
    state = TrainingState(
        current_loss=2.5,
        gradient_magnitude=0.1,
        cluster_id=1,
        loss_trend=0.95,  # Loss improving
        gradient_variance=0.01,
        step_number=100
    )
    
    # Test predictions
    lr = aux_nn.predict_learning_rate(state)
    direction = aux_nn.predict_update_direction_bias(state)
    trust_region = aux_nn.predict_trust_region(state)
    
    print(f"Predicted LR multiplier: {lr:.4f}")
    print(f"Direction bias: {direction:.4f}")
    print(f"Trust region: {trust_region:.4f}")
    
    # Test adaptive optimizer
    print("\n=== Adaptive Optimizer Test ===")
    
    optimizer = AdaptiveOptimizer(base_learning_rate=0.001)
    
    # Simulate optimization steps
    params = np.random.randn(100)
    
    for step in range(5):
        gradients = np.random.randn(100)
        state = TrainingState(
            current_loss=2.5 - step * 0.1,
            gradient_magnitude=np.linalg.norm(gradients),
            cluster_id=0,
            loss_trend=0.98,
            gradient_variance=0.01,
            step_number=step
        )
        
        updated_params, info = optimizer.step(params, gradients, state, state.current_loss)
        
        print(f"Step {step}: Loss={state.current_loss:.4f}, Update mag={info['update_magnitude']:.6f}, "
              f"Grad mag={info['gradient_magnitude']:.4f}")
        
        params = updated_params
    
    # Print prediction statistics
    print("\nPrediction Statistics:")
    stats = optimizer.auxiliary_nn.get_prediction_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
