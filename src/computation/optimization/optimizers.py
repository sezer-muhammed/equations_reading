"""
Optimization algorithm implementations with detailed mathematical breakdowns.
Implements Adam optimizer and gradient descent with visualization support.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from ...core.models import (
    ColorCodedMatrix, OperationVisualization, HighlightPattern,
    AnimationFrame, ComputationStep, VisualizationData
)


@dataclass
class OptimizationStep:
    """Single step in optimization process."""
    step_number: int
    parameters: np.ndarray
    gradients: np.ndarray
    loss: float
    learning_rate: float
    additional_info: Dict[str, np.ndarray]


@dataclass
class OptimizationResult:
    """Result of optimization process with detailed tracking."""
    final_parameters: np.ndarray
    optimization_steps: List[OptimizationStep]
    computation_steps: List[ComputationStep]
    convergence_info: Dict[str, any]
    visualization: OperationVisualization


class GradientDescentOptimizer:
    """Gradient descent optimizer with step-by-step tracking."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.color_palette = {
            'parameters': '#FF6B6B',    # Red for parameters
            'gradients': '#4ECDC4',     # Teal for gradients
            'updates': '#45B7D1',       # Blue for updates
            'loss': '#96CEB4',          # Green for loss
            'trajectory': '#FFEAA7'     # Yellow for trajectory
        }
    
    def optimize(self, initial_params: np.ndarray, 
                loss_function: Callable[[np.ndarray], float],
                gradient_function: Callable[[np.ndarray], np.ndarray],
                num_steps: int = 100,
                tolerance: float = 1e-6) -> OptimizationResult:
        """
        Perform gradient descent optimization.
        
        Args:
            initial_params: Starting parameter values
            loss_function: Function to compute loss
            gradient_function: Function to compute gradients
            num_steps: Maximum number of optimization steps
            tolerance: Convergence tolerance
            
        Returns:
            OptimizationResult with detailed step tracking
        """
        params = initial_params.copy()
        optimization_steps = []
        computation_steps = []
        
        for step in range(num_steps):
            # Compute loss and gradients
            current_loss = loss_function(params)
            gradients = gradient_function(params)
            
            # Create computation step for gradient computation
            comp_step = ComputationStep(
                step_number=step * 2 + 1,
                operation_name="compute_gradients",
                input_values={'parameters': params.copy()},
                operation_description=f"Compute gradients at step {step}",
                output_values={'gradients': gradients, 'loss': np.array([current_loss])},
                visualization_hints={
                    'gradient_magnitude': np.linalg.norm(gradients),
                    'loss_value': current_loss
                }
            )
            computation_steps.append(comp_step)
            
            # Store optimization step before update
            opt_step = OptimizationStep(
                step_number=step,
                parameters=params.copy(),
                gradients=gradients.copy(),
                loss=current_loss,
                learning_rate=self.learning_rate,
                additional_info={'gradient_norm': np.linalg.norm(gradients)}
            )
            optimization_steps.append(opt_step)
            
            # Check convergence
            if np.linalg.norm(gradients) < tolerance:
                break
            
            # Update parameters: θ = θ - α * ∇θ
            param_update = -self.learning_rate * gradients
            params = params + param_update
            
            # Create computation step for parameter update
            comp_step_update = ComputationStep(
                step_number=step * 2 + 2,
                operation_name="update_parameters",
                input_values={
                    'old_parameters': opt_step.parameters,
                    'gradients': gradients,
                    'learning_rate': np.array([self.learning_rate])
                },
                operation_description=f"Update parameters: θ = θ - α∇θ",
                output_values={'new_parameters': params, 'update': param_update},
                visualization_hints={
                    'update_magnitude': np.linalg.norm(param_update),
                    'learning_rate': self.learning_rate
                }
            )
            computation_steps.append(comp_step_update)
        
        # Create visualization
        visualization = self._create_gd_visualization(optimization_steps)
        
        # Calculate convergence info
        convergence_info = {
            'converged': step < num_steps - 1,
            'final_loss': optimization_steps[-1].loss,
            'total_steps': len(optimization_steps),
            'final_gradient_norm': np.linalg.norm(optimization_steps[-1].gradients),
            'loss_reduction': optimization_steps[0].loss - optimization_steps[-1].loss
        }
        
        return OptimizationResult(
            final_parameters=params,
            optimization_steps=optimization_steps,
            computation_steps=computation_steps,
            convergence_info=convergence_info,
            visualization=visualization
        )
    
    def _create_gd_visualization(self, steps: List[OptimizationStep]) -> OperationVisualization:
        """Create visualization for gradient descent trajectory."""
        
        # Extract parameter trajectory
        param_trajectory = np.array([step.parameters for step in steps])
        loss_trajectory = np.array([step.loss for step in steps])
        
        # Create color-coded matrices for key steps
        initial_params = ColorCodedMatrix(
            matrix_data=steps[0].parameters.reshape(-1, 1),
            color_mapping={'default': self.color_palette['parameters']},
            element_labels={},
            highlight_patterns=[]
        )
        
        final_params = ColorCodedMatrix(
            matrix_data=steps[-1].parameters.reshape(-1, 1),
            color_mapping={'default': self.color_palette['parameters']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="gradient_descent",
            input_matrices=[initial_params],
            intermediate_steps=[],
            output_matrix=final_params,
            animation_sequence=[]
        )


class AdamOptimizer:
    """Adam optimizer with detailed mathematical breakdown."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.color_palette = {
            'parameters': '#FF6B6B',
            'gradients': '#4ECDC4', 
            'momentum': '#45B7D1',
            'velocity': '#96CEB4',
            'updates': '#FFEAA7'
        }
    
    def optimize(self, initial_params: np.ndarray,
                loss_function: Callable[[np.ndarray], float],
                gradient_function: Callable[[np.ndarray], np.ndarray],
                num_steps: int = 100,
                tolerance: float = 1e-6) -> OptimizationResult:
        """
        Perform Adam optimization with detailed step tracking.
        
        Adam update rules:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
        """
        params = initial_params.copy()
        m = np.zeros_like(params)  # First moment estimate
        v = np.zeros_like(params)  # Second moment estimate
        
        optimization_steps = []
        computation_steps = []
        
        for step in range(1, num_steps + 1):  # Adam uses 1-based indexing
            # Compute loss and gradients
            current_loss = loss_function(params)
            gradients = gradient_function(params)
            
            # Step 1: Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * gradients
            
            comp_step_1 = ComputationStep(
                step_number=(step - 1) * 6 + 1,
                operation_name="update_first_moment",
                input_values={'m_prev': m / self.beta1 - (1 - self.beta1) * gradients / self.beta1, 
                             'gradients': gradients},
                operation_description=f"Update first moment: m_t = β₁m_{{t-1}} + (1-β₁)g_t",
                output_values={'m': m},
                visualization_hints={'beta1': self.beta1}
            )
            computation_steps.append(comp_step_1)
            
            # Step 2: Update biased second moment estimate  
            v = self.beta2 * v + (1 - self.beta2) * (gradients ** 2)
            
            comp_step_2 = ComputationStep(
                step_number=(step - 1) * 6 + 2,
                operation_name="update_second_moment",
                input_values={'v_prev': v / self.beta2 - (1 - self.beta2) * (gradients ** 2) / self.beta2,
                             'gradients_squared': gradients ** 2},
                operation_description=f"Update second moment: v_t = β₂v_{{t-1}} + (1-β₂)g_t²",
                output_values={'v': v},
                visualization_hints={'beta2': self.beta2}
            )
            computation_steps.append(comp_step_2)
            
            # Step 3: Compute bias-corrected first moment
            m_hat = m / (1 - self.beta1 ** step)
            
            comp_step_3 = ComputationStep(
                step_number=(step - 1) * 6 + 3,
                operation_name="bias_correct_first_moment",
                input_values={'m': m, 'beta1_power': np.array([self.beta1 ** step])},
                operation_description=f"Bias correction: m̂_t = m_t / (1 - β₁^t)",
                output_values={'m_hat': m_hat},
                visualization_hints={'bias_correction_factor': 1 - self.beta1 ** step}
            )
            computation_steps.append(comp_step_3)
            
            # Step 4: Compute bias-corrected second moment
            v_hat = v / (1 - self.beta2 ** step)
            
            comp_step_4 = ComputationStep(
                step_number=(step - 1) * 6 + 4,
                operation_name="bias_correct_second_moment",
                input_values={'v': v, 'beta2_power': np.array([self.beta2 ** step])},
                operation_description=f"Bias correction: v̂_t = v_t / (1 - β₂^t)",
                output_values={'v_hat': v_hat},
                visualization_hints={'bias_correction_factor': 1 - self.beta2 ** step}
            )
            computation_steps.append(comp_step_4)
            
            # Step 5: Compute parameter update
            param_update = -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            comp_step_5 = ComputationStep(
                step_number=(step - 1) * 6 + 5,
                operation_name="compute_update",
                input_values={'m_hat': m_hat, 'v_hat': v_hat, 'lr': np.array([self.learning_rate])},
                operation_description=f"Compute update: Δθ = -α * m̂_t / (√v̂_t + ε)",
                output_values={'param_update': param_update},
                visualization_hints={'epsilon': self.epsilon}
            )
            computation_steps.append(comp_step_5)
            
            # Store optimization step before update
            opt_step = OptimizationStep(
                step_number=step - 1,
                parameters=params.copy(),
                gradients=gradients.copy(),
                loss=current_loss,
                learning_rate=self.learning_rate,
                additional_info={
                    'm': m.copy(),
                    'v': v.copy(),
                    'm_hat': m_hat.copy(),
                    'v_hat': v_hat.copy(),
                    'update_magnitude': np.linalg.norm(param_update)
                }
            )
            optimization_steps.append(opt_step)
            
            # Step 6: Update parameters
            params = params + param_update
            
            comp_step_6 = ComputationStep(
                step_number=(step - 1) * 6 + 6,
                operation_name="update_parameters",
                input_values={'old_params': opt_step.parameters, 'update': param_update},
                operation_description=f"Update parameters: θ_t = θ_{{t-1}} + Δθ",
                output_values={'new_params': params},
                visualization_hints={'step': step}
            )
            computation_steps.append(comp_step_6)
            
            # Check convergence
            if np.linalg.norm(gradients) < tolerance:
                break
        
        # Create visualization
        visualization = self._create_adam_visualization(optimization_steps)
        
        # Calculate convergence info
        convergence_info = {
            'converged': len(optimization_steps) < num_steps,
            'final_loss': optimization_steps[-1].loss,
            'total_steps': len(optimization_steps),
            'final_gradient_norm': np.linalg.norm(optimization_steps[-1].gradients),
            'loss_reduction': optimization_steps[0].loss - optimization_steps[-1].loss,
            'adam_hyperparameters': {
                'beta1': self.beta1,
                'beta2': self.beta2,
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate
            }
        }
        
        return OptimizationResult(
            final_parameters=params,
            optimization_steps=optimization_steps,
            computation_steps=computation_steps,
            convergence_info=convergence_info,
            visualization=visualization
        )
    
    def _create_adam_visualization(self, steps: List[OptimizationStep]) -> OperationVisualization:
        """Create visualization for Adam optimization trajectory."""
        
        # Extract trajectories
        param_trajectory = np.array([step.parameters for step in steps])
        momentum_trajectory = np.array([step.additional_info['m'] for step in steps])
        velocity_trajectory = np.array([step.additional_info['v'] for step in steps])
        
        # Create color-coded matrices
        initial_params = ColorCodedMatrix(
            matrix_data=steps[0].parameters.reshape(-1, 1),
            color_mapping={'default': self.color_palette['parameters']},
            element_labels={},
            highlight_patterns=[]
        )
        
        final_momentum = ColorCodedMatrix(
            matrix_data=steps[-1].additional_info['m'].reshape(-1, 1),
            color_mapping={'default': self.color_palette['momentum']},
            element_labels={},
            highlight_patterns=[]
        )
        
        final_velocity = ColorCodedMatrix(
            matrix_data=steps[-1].additional_info['v'].reshape(-1, 1),
            color_mapping={'default': self.color_palette['velocity']},
            element_labels={},
            highlight_patterns=[]
        )
        
        final_params = ColorCodedMatrix(
            matrix_data=steps[-1].parameters.reshape(-1, 1),
            color_mapping={'default': self.color_palette['parameters']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="adam_optimization",
            input_matrices=[initial_params],
            intermediate_steps=[final_momentum, final_velocity],
            output_matrix=final_params,
            animation_sequence=[]
        )