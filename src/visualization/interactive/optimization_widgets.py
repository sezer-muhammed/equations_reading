"""
Interactive widgets for optimization algorithm demonstrations.
Allows real-time parameter manipulation and visualization updates.
"""

import numpy as np
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

from ...core.models import VisualizationData
from ...computation.optimization.optimizers import GradientDescentOptimizer, AdamOptimizer


@dataclass
class OptimizerWidget:
    """Interactive widget for optimizer parameter manipulation."""
    widget_type: str
    parameter_name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    description: str


@dataclass
class OptimizationInteractiveDemo:
    """Complete interactive optimization demonstration."""
    loss_function: Callable
    gradient_function: Callable
    parameter_widgets: List[OptimizerWidget]
    visualization_data: Dict[str, any]
    current_results: Dict[str, any]


class OptimizationInteractiveComponents:
    """Creates interactive components for optimization tutorials."""
    
    def __init__(self):
        self.gd_optimizer = GradientDescentOptimizer()
        self.adam_optimizer = AdamOptimizer()
    
    def create_learning_rate_demo(self) -> OptimizationInteractiveDemo:
        """
        Create interactive demo for learning rate effects.
        
        Returns:
            OptimizationInteractiveDemo with learning rate controls
        """
        # Define simple quadratic loss function
        def loss_function(params):
            return np.sum(params**2)
        
        def gradient_function(params):
            return 2 * params
        
        # Create parameter widgets
        widgets = [
            OptimizerWidget(
                widget_type="slider",
                parameter_name="learning_rate",
                current_value=0.1,
                min_value=0.001,
                max_value=1.0,
                step_size=0.001,
                description="Learning rate (α): Controls step size in gradient descent"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="initial_x",
                current_value=2.0,
                min_value=-5.0,
                max_value=5.0,
                step_size=0.1,
                description="Initial x parameter value"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="initial_y",
                current_value=1.5,
                min_value=-5.0,
                max_value=5.0,
                step_size=0.1,
                description="Initial y parameter value"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="num_steps",
                current_value=50,
                min_value=10,
                max_value=200,
                step_size=1,
                description="Number of optimization steps"
            )
        ]
        
        # Initial visualization data
        initial_params = np.array([2.0, 1.5])
        self.gd_optimizer.learning_rate = 0.1
        initial_result = self.gd_optimizer.optimize(
            initial_params, loss_function, gradient_function, num_steps=50
        )
        
        visualization_data = {
            'trajectory': np.array([step.parameters for step in initial_result.optimization_steps]),
            'losses': [step.loss for step in initial_result.optimization_steps],
            'gradients': [step.gradients for step in initial_result.optimization_steps]
        }
        
        return OptimizationInteractiveDemo(
            loss_function=loss_function,
            gradient_function=gradient_function,
            parameter_widgets=widgets,
            visualization_data=visualization_data,
            current_results={'optimization_result': initial_result}
        )
    
    def create_adam_hyperparameter_demo(self) -> OptimizationInteractiveDemo:
        """
        Create interactive demo for Adam hyperparameter effects.
        
        Returns:
            OptimizationInteractiveDemo with Adam hyperparameter controls
        """
        # Define Rosenbrock function for challenging optimization
        def rosenbrock_loss(params):
            a, b = 1, 100
            x, y = params[0], params[1]
            return (a - x)**2 + b * (y - x**2)**2
        
        def rosenbrock_gradient(params):
            a, b = 1, 100
            x, y = params[0], params[1]
            dx = -2*(a - x) - 4*b*x*(y - x**2)
            dy = 2*b*(y - x**2)
            return np.array([dx, dy])
        
        # Create parameter widgets for Adam hyperparameters
        widgets = [
            OptimizerWidget(
                widget_type="slider",
                parameter_name="learning_rate",
                current_value=0.001,
                min_value=0.0001,
                max_value=0.1,
                step_size=0.0001,
                description="Learning rate (α): Base learning rate for Adam"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="beta1",
                current_value=0.9,
                min_value=0.1,
                max_value=0.999,
                step_size=0.001,
                description="Beta1 (β₁): First moment decay rate (momentum)"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="beta2",
                current_value=0.999,
                min_value=0.9,
                max_value=0.9999,
                step_size=0.0001,
                description="Beta2 (β₂): Second moment decay rate (velocity)"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="epsilon",
                current_value=1e-8,
                min_value=1e-10,
                max_value=1e-6,
                step_size=1e-9,
                description="Epsilon (ε): Small constant for numerical stability"
            )
        ]
        
        # Initial visualization data
        initial_params = np.array([-1.0, 1.0])
        initial_result = self.adam_optimizer.optimize(
            initial_params, rosenbrock_loss, rosenbrock_gradient, num_steps=100
        )
        
        visualization_data = {
            'trajectory': np.array([step.parameters for step in initial_result.optimization_steps]),
            'losses': [step.loss for step in initial_result.optimization_steps],
            'momentum': [step.additional_info['m'] for step in initial_result.optimization_steps],
            'velocity': [step.additional_info['v'] for step in initial_result.optimization_steps]
        }
        
        return OptimizationInteractiveDemo(
            loss_function=rosenbrock_loss,
            gradient_function=rosenbrock_gradient,
            parameter_widgets=widgets,
            visualization_data=visualization_data,
            current_results={'optimization_result': initial_result}
        )
    
    def create_optimizer_comparison_demo(self) -> OptimizationInteractiveDemo:
        """
        Create interactive demo comparing different optimizers.
        
        Returns:
            OptimizationInteractiveDemo with optimizer comparison
        """
        # Define a challenging multi-modal function
        def multimodal_loss(params):
            x, y = params[0], params[1]
            return (np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2))
        
        def multimodal_gradient(params):
            x, y = params[0], params[1]
            dx = np.cos(x) * np.cos(y) + 0.2 * x
            dy = -np.sin(x) * np.sin(y) + 0.2 * y
            return np.array([dx, dy])
        
        # Create widgets for comparison parameters
        widgets = [
            OptimizerWidget(
                widget_type="dropdown",
                parameter_name="optimizer_type",
                current_value="both",
                min_value=0,
                max_value=2,
                step_size=1,
                description="Optimizer: Choose which optimizer(s) to display"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="gd_learning_rate",
                current_value=0.01,
                min_value=0.001,
                max_value=0.1,
                step_size=0.001,
                description="Gradient Descent learning rate"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="adam_learning_rate",
                current_value=0.01,
                min_value=0.001,
                max_value=0.1,
                step_size=0.001,
                description="Adam learning rate"
            ),
            OptimizerWidget(
                widget_type="slider",
                parameter_name="num_steps",
                current_value=100,
                min_value=50,
                max_value=300,
                step_size=10,
                description="Number of optimization steps"
            )
        ]
        
        # Run both optimizers for comparison
        initial_params = np.array([2.0, 1.0])
        
        self.gd_optimizer.learning_rate = 0.01
        gd_result = self.gd_optimizer.optimize(
            initial_params.copy(), multimodal_loss, multimodal_gradient, num_steps=100
        )
        
        self.adam_optimizer.learning_rate = 0.01
        adam_result = self.adam_optimizer.optimize(
            initial_params.copy(), multimodal_loss, multimodal_gradient, num_steps=100
        )
        
        visualization_data = {
            'gd_trajectory': np.array([step.parameters for step in gd_result.optimization_steps]),
            'adam_trajectory': np.array([step.parameters for step in adam_result.optimization_steps]),
            'gd_losses': [step.loss for step in gd_result.optimization_steps],
            'adam_losses': [step.loss for step in adam_result.optimization_steps]
        }
        
        return OptimizationInteractiveDemo(
            loss_function=multimodal_loss,
            gradient_function=multimodal_gradient,
            parameter_widgets=widgets,
            visualization_data=visualization_data,
            current_results={
                'gd_result': gd_result,
                'adam_result': adam_result
            }
        )
    
    def update_demo(self, demo: OptimizationInteractiveDemo, 
                   parameter_updates: Dict[str, float]) -> OptimizationInteractiveDemo:
        """
        Update interactive demo with new parameter values.
        
        Args:
            demo: Current demo state
            parameter_updates: Dictionary of parameter name -> new value
            
        Returns:
            Updated OptimizationInteractiveDemo
        """
        # Update widget values
        for widget in demo.parameter_widgets:
            if widget.parameter_name in parameter_updates:
                widget.current_value = parameter_updates[widget.parameter_name]
        
        # Re-run optimization with new parameters
        if 'learning_rate' in parameter_updates:
            if hasattr(demo.current_results, 'optimization_result'):
                # Single optimizer demo
                optimizer = self.gd_optimizer if 'gd' in str(type(demo.current_results)) else self.adam_optimizer
                optimizer.learning_rate = parameter_updates['learning_rate']
                
                initial_params = np.array([
                    parameter_updates.get('initial_x', 2.0),
                    parameter_updates.get('initial_y', 1.5)
                ])
                
                new_result = optimizer.optimize(
                    initial_params, demo.loss_function, demo.gradient_function,
                    num_steps=int(parameter_updates.get('num_steps', 50))
                )
                
                demo.current_results['optimization_result'] = new_result
                demo.visualization_data = {
                    'trajectory': np.array([step.parameters for step in new_result.optimization_steps]),
                    'losses': [step.loss for step in new_result.optimization_steps]
                }
        
        return demo