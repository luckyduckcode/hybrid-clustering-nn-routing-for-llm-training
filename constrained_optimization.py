"""
Constrained Optimization Update System

Implements the constrained SGD update mechanism that uses NN-predicted step sizes
and incorporates 1.58-bit quantization for ultra-low-bit LLM training.

Mathematical formulation:
  ΔΘ* = arg min[L(Θ + ΔΘ) - L(Θ)]
  subject to: ||ΔΘ||_2 ≤ ŵ_t^2
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from quantization import Quantizer158Bit, AdaptiveQuantizer
from auxiliary_nn import TrainingState, AdaptiveOptimizer


@dataclass
class ConstrainedUpdateInfo:
    """Information about a constrained update step."""
    parameters_updated: np.ndarray
    update_vector: np.ndarray
    constraint_active: bool  # Was the constraint binding?
    constraint_radius: float
    update_magnitude: float
    update_efficiency: float  # Ratio of actual improvement to update magnitude
    quantization_error: float


class ConstrainedOptimizationStep:
    """
    Implements constrained optimization step with quantization.
    Core component of the hybrid optimization framework.
    """
    
    def __init__(self, 
                 quantizer: Optional[Quantizer158Bit] = None,
                 use_quantization: bool = True):
        """
        Initialize constrained optimization step.
        
        Args:
            quantizer: 1.58-bit quantizer (creates default if None)
            use_quantization: Whether to apply quantization
        """
        self.quantizer = quantizer or Quantizer158Bit(scale=1.0)
        self.use_quantization = use_quantization
        self.step_history = []
        
    def apply_constraint(self,
                        unconstrained_update: np.ndarray,
                        constraint_radius: float) -> Tuple[np.ndarray, bool]:
        """
        Apply 2-norm constraint to update vector.
        Implements: ||ΔΘ||_2 ≤ ŵ_t
        
        Args:
            unconstrained_update: Unscaled gradient/update direction
            constraint_radius: Maximum allowed 2-norm (ŵ_t)
            
        Returns:
            Tuple of (constrained_update, constraint_active)
        """
        update_norm = np.linalg.norm(unconstrained_update)
        
        if update_norm <= constraint_radius:
            return unconstrained_update, False
        else:
            # Rescale update to satisfy constraint
            constrained = (unconstrained_update / update_norm) * constraint_radius
            return constrained, True
    
    def apply_quantization_post_update(self,
                                       parameters: np.ndarray,
                                       quantizer: Optional[Quantizer158Bit] = None) -> Tuple[np.ndarray, float]:
        """
        Apply 1.58-bit quantization to updated parameters.
        
        Args:
            parameters: Updated parameters
            quantizer: Specific quantizer (uses self.quantizer if None)
            
        Returns:
            Tuple of (quantized_parameters, quantization_error)
        """
        if not self.use_quantization:
            return parameters, 0.0
        
        q = quantizer or self.quantizer
        
        # Store original for error calculation
        original = parameters.copy()
        
        # Apply quantization
        quantized = q.quantize_weights(parameters)
        
        # Calculate quantization error
        error = np.linalg.norm(quantized - original)
        
        return quantized, error
    
    def apply_quantization_gradient(self,
                                    gradients: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply 1.58-bit quantization to gradients before update.
        Preserves gradient direction while reducing magnitude precision.
        
        Args:
            gradients: Full-precision gradients
            
        Returns:
            Tuple of (quantized_gradients, quantization_error)
        """
        if not self.use_quantization:
            return gradients, 0.0
        
        original = gradients.copy()
        quantized = self.quantizer.quantize_gradients(gradients)
        
        error = np.linalg.norm(quantized - original)
        
        return quantized, error
    
    def step(self,
             parameters: np.ndarray,
             gradients: np.ndarray,
             constraint_radius: float,
             learning_rate: float = 0.001,
             quantize_gradients: bool = True,
             quantize_parameters: bool = True) -> ConstrainedUpdateInfo:
        """
        Perform a complete constrained optimization step with quantization.
        
        Mathematical process:
        1. Optionally quantize gradients
        2. Compute scaled gradient direction: g_scaled = lr * ∇L
        3. Apply constraint: ΔΘ = min(||g_scaled||, ŵ_t) * (g_scaled / ||g_scaled||)
        4. Update parameters: Θ_new = Θ - ΔΘ
        5. Optionally quantize parameters
        
        Args:
            parameters: Current model parameters
            gradients: Full-precision gradients
            constraint_radius: Trust region radius (ŵ_t)
            learning_rate: Learning rate coefficient
            quantize_gradients: Apply quantization to gradients first
            quantize_parameters: Apply quantization to final parameters
            
        Returns:
            ConstrainedUpdateInfo with detailed step information
        """
        # Step 1: Optionally quantize gradients
        if quantize_gradients:
            quantized_grads, grad_quant_error = self.apply_quantization_gradient(gradients)
        else:
            quantized_grads = gradients
            grad_quant_error = 0.0
        
        # Step 2: Compute scaled gradient
        scaled_gradient = learning_rate * quantized_grads
        
        # Step 3: Apply constraint (trust region)
        constrained_update, constraint_active = self.apply_constraint(
            scaled_gradient,
            constraint_radius
        )
        
        # Step 4: Update parameters
        updated_parameters = parameters - constrained_update
        
        # Step 5: Optionally quantize updated parameters
        if quantize_parameters:
            final_parameters, param_quant_error = self.apply_quantization_post_update(
                updated_parameters
            )
        else:
            final_parameters = updated_parameters
            param_quant_error = 0.0
        
        # Calculate total quantization error
        total_quant_error = grad_quant_error + param_quant_error
        
        # Calculate update efficiency
        update_magnitude = np.linalg.norm(constrained_update)
        if update_magnitude > 0:
            # Efficiency: how much the constraint limited us (0 = no limit, 1 = fully constrained)
            efficiency = min(1.0, update_magnitude / np.linalg.norm(scaled_gradient)) if np.linalg.norm(scaled_gradient) > 0 else 1.0
        else:
            efficiency = 0.0
        
        info = ConstrainedUpdateInfo(
            parameters_updated=final_parameters,
            update_vector=constrained_update,
            constraint_active=constraint_active,
            constraint_radius=constraint_radius,
            update_magnitude=update_magnitude,
            update_efficiency=efficiency,
            quantization_error=total_quant_error
        )
        
        self.step_history.append(info)
        
        return info


class AdaptiveConstrainedOptimizer:
    """
    Combines constrained optimization with auxiliary NN for adaptive constraint radius.
    Integrates all components of the hybrid framework.
    """
    
    def __init__(self,
                 base_learning_rate: float = 0.001,
                 quantizer: Optional[Quantizer158Bit] = None,
                 use_quantization: bool = True,
                 use_adaptive_optimizer: bool = True):
        """
        Initialize adaptive constrained optimizer.
        
        Args:
            base_learning_rate: Base learning rate
            quantizer: 1.58-bit quantizer instance
            use_quantization: Whether to use quantization
            use_adaptive_optimizer: Whether to use NN-guided learning rate
        """
        self.base_learning_rate = base_learning_rate
        self.quantizer = quantizer or Quantizer158Bit()
        self.use_quantization = use_quantization
        self.constrained_step = ConstrainedOptimizationStep(self.quantizer, use_quantization)
        
        # Auxiliary optimizer for NN-guided adaptation
        self.adaptive_optimizer = AdaptiveOptimizer(base_learning_rate) if use_adaptive_optimizer else None
        
        self.step_count = 0
        self.loss_history = []
        
    def step(self,
             parameters: np.ndarray,
             gradients: np.ndarray,
             state: TrainingState,
             current_loss: float) -> ConstrainedUpdateInfo:
        """
        Perform one complete hybrid optimization step.
        
        Args:
            parameters: Current parameters
            gradients: Current gradients
            state: Training state
            current_loss: Current loss value
            
        Returns:
            ConstrainedUpdateInfo with step details
        """
        self.step_count += 1
        
        # Determine constraint radius using auxiliary NN
        if self.adaptive_optimizer:
            # Use NN to predict trust region
            trust_region = self.adaptive_optimizer.auxiliary_nn.predict_trust_region(state)
            effective_lr = self.base_learning_rate * self.adaptive_optimizer.auxiliary_nn.predict_learning_rate(state)
        else:
            # Fixed constraint radius (conservative)
            trust_region = 0.01
            effective_lr = self.base_learning_rate
        
        # Perform constrained step with quantization
        update_info = self.constrained_step.step(
            parameters=parameters,
            gradients=gradients,
            constraint_radius=trust_region,
            learning_rate=effective_lr,
            quantize_gradients=self.use_quantization,
            quantize_parameters=self.use_quantization
        )
        
        # Update auxiliary NN with feedback if available
        if self.adaptive_optimizer and len(self.loss_history) > 0:
            loss_reduction = self.loss_history[-1] - current_loss
            self.adaptive_optimizer.auxiliary_nn.update_with_feedback(
                effective_lr,
                loss_reduction
            )
        
        self.loss_history.append(current_loss)
        
        return update_info
    
    def get_training_summary(self) -> Dict:
        """
        Get summary statistics of training progress.
        
        Returns:
            Dictionary with training statistics
        """
        if not self.step_history:
            return {'steps': 0}
        
        steps = self.constrained_step.step_history
        
        # Calculate statistics
        constraint_active_count = sum(1 for s in steps if s.constraint_active)
        update_magnitudes = [s.update_magnitude for s in steps]
        quant_errors = [s.quantization_error for s in steps]
        
        return {
            'total_steps': len(steps),
            'constraints_active': constraint_active_count,
            'constraint_activation_rate': constraint_active_count / len(steps) if steps else 0,
            'mean_update_magnitude': float(np.mean(update_magnitudes)) if update_magnitudes else 0,
            'std_update_magnitude': float(np.std(update_magnitudes)) if update_magnitudes else 0,
            'mean_quantization_error': float(np.mean(quant_errors)) if quant_errors else 0,
            'total_loss_reduction': float(self.loss_history[0] - self.loss_history[-1]) if self.loss_history else 0,
            'loss_trend': float(np.mean(np.diff(self.loss_history[-10:]))) if len(self.loss_history) > 10 else 0,
        }
    
    @property
    def step_history(self):
        """Access step history."""
        return self.constrained_step.step_history


if __name__ == "__main__":
    print("=== Constrained Optimization Test ===\n")
    
    # Initialize components
    quantizer = Quantizer158Bit(scale=1.0)
    constrained_opt = ConstrainedOptimizationStep(quantizer, use_quantization=True)
    
    # Test constrained update
    print("Test 1: Constrained Update")
    params = np.random.randn(100)
    grads = np.random.randn(100)
    constraint_radius = 0.01
    
    info = constrained_opt.step(
        parameters=params,
        gradients=grads,
        constraint_radius=constraint_radius,
        learning_rate=0.001,
        quantize_gradients=True,
        quantize_parameters=True
    )
    
    print(f"  Update magnitude: {info.update_magnitude:.6f}")
    print(f"  Constraint active: {info.constraint_active}")
    print(f"  Constraint radius: {info.constraint_radius:.6f}")
    print(f"  Quantization error: {info.quantization_error:.6f}")
    
    # Test adaptive constrained optimizer
    print("\nTest 2: Adaptive Constrained Optimizer")
    
    adaptive_opt = AdaptiveConstrainedOptimizer(
        base_learning_rate=0.001,
        use_quantization=True,
        use_adaptive_optimizer=True
    )
    
    params = np.random.randn(100)
    loss = 2.5
    
    for step in range(10):
        grads = np.random.randn(100)
        
        state = TrainingState(
            current_loss=loss,
            gradient_magnitude=np.linalg.norm(grads),
            cluster_id=0,
            loss_trend=0.98,
            gradient_variance=0.01,
            step_number=step
        )
        
        info = adaptive_opt.step(params, grads, state, loss)
        params = info.parameters_updated
        loss -= 0.05  # Simulate loss reduction
        
        print(f"  Step {step+1}: Loss={loss:.4f}, Update={info.update_magnitude:.6f}, "
              f"Constraint active={info.constraint_active}")
    
    # Print summary
    print("\nTraining Summary:")
    summary = adaptive_opt.get_training_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
