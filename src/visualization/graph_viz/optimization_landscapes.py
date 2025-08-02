"""
Visualization components for optimization landscapes and convergence trajectories.
Creates interactive plots showing loss landscapes and optimizer paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import VisualizationData, ColorCodedMatrix, OperationVisualization
from ...computation.optimization.optimizers import OptimizationResult, OptimizationStep


@dataclass
class LandscapeVisualization:
    """Visualization data for loss landscapes."""
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    contour_levels: np.ndarray
    optimizer_paths: List[np.ndarray]
    optimizer_names: List[str]
    colormap: str = 'viridis'


class OptimizationLandscapeVisualizer:
    """Creates visualizations for optimization landscapes and trajectories."""
    
    def __init__(self):
        self.color_palette = {
            'gradient_descent': '#FF6B6B',
            'adam': '#4ECDC4',
            'momentum': '#45B7D1',
            'landscape': '#96CEB4',
            'trajectory': '#FFEAA7',
            'minimum': '#FF0000'
        }
    
    def create_loss_landscape(self, landscape_data: Dict[str, np.ndarray],
                            optimizer_results: List[OptimizationResult],
                            optimizer_names: List[str]) -> LandscapeVisualization:
        """
        Create loss landscape visualization with optimizer trajectories.
        
        Args:
            landscape_data: Dictionary with 'X', 'Y', 'Z' mesh data
            optimizer_results: List of optimization results to plot
            optimizer_names: Names of optimizers for legend
            
        Returns:
            LandscapeVisualization object with all visualization data
        """
        X, Y, Z = landscape_data['X'], landscape_data['Y'], landscape_data['Z']
        
        # Create contour levels
        contour_levels = np.logspace(np.log10(Z.min() + 1e-8), 
                                   np.log10(Z.max()), 20)
        
        # Extract optimizer paths
        optimizer_paths = []
        for result in optimizer_results:
            path = np.array([[step.parameters[0], step.parameters[1]] 
                           for step in result.optimization_steps])
            optimizer_paths.append(path)
        
        return LandscapeVisualization(
            X=X, Y=Y, Z=Z,
            contour_levels=contour_levels,
            optimizer_paths=optimizer_paths,
            optimizer_names=optimizer_names
        )
    
    def create_convergence_plot(self, optimizer_results: List[OptimizationResult],
                              optimizer_names: List[str]) -> Dict[str, any]:
        """
        Create convergence plots showing loss vs iteration.
        
        Args:
            optimizer_results: List of optimization results
            optimizer_names: Names of optimizers
            
        Returns:
            Dictionary with convergence plot data
        """
        convergence_data = {
            'optimizers': [],
            'iterations': [],
            'losses': [],
            'gradient_norms': []
        }
        
        for result, name in zip(optimizer_results, optimizer_names):
            iterations = list(range(len(result.optimization_steps)))
            losses = [step.loss for step in result.optimization_steps]
            grad_norms = [np.linalg.norm(step.gradients) for step in result.optimization_steps]
            
            convergence_data['optimizers'].append(name)
            convergence_data['iterations'].append(iterations)
            convergence_data['losses'].append(losses)
            convergence_data['gradient_norms'].append(grad_norms)
        
        return convergence_data
    
    def create_learning_rate_visualization(self, lr_schedule_data: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Create visualization for learning rate schedules.
        
        Args:
            lr_schedule_data: List of learning rate schedule results
            
        Returns:
            Dictionary with learning rate visualization data
        """
        viz_data = {
            'schedules': [],
            'steps': [],
            'learning_rates': [],
            'losses': []
        }
        
        for schedule in lr_schedule_data:
            schedule_type = schedule['schedule_type']
            results = schedule['results']
            
            steps = [r['step'] for r in results]
            lrs = [r['lr'] for r in results]
            losses = [r['loss'] for r in results]
            
            viz_data['schedules'].append(schedule_type)
            viz_data['steps'].append(steps)
            viz_data['learning_rates'].append(lrs)
            viz_data['losses'].append(losses)
        
        return viz_data
    
    def create_adam_components_visualization(self, adam_result: OptimizationResult) -> Dict[str, any]:
        """
        Create visualization showing Adam's momentum and velocity components.
        
        Args:
            adam_result: Adam optimization result with detailed tracking
            
        Returns:
            Dictionary with Adam components visualization data
        """
        steps = adam_result.optimization_steps
        
        # Extract Adam-specific data
        momentum_data = []
        velocity_data = []
        bias_corrected_momentum = []
        bias_corrected_velocity = []
        
        for step in steps:
            if 'm' in step.additional_info:
                momentum_data.append(step.additional_info['m'])
                velocity_data.append(step.additional_info['v'])
                bias_corrected_momentum.append(step.additional_info['m_hat'])
                bias_corrected_velocity.append(step.additional_info['v_hat'])
        
        return {
            'iterations': list(range(len(steps))),
            'parameters': [step.parameters for step in steps],
            'gradients': [step.gradients for step in steps],
            'momentum': momentum_data,
            'velocity': velocity_data,
            'momentum_corrected': bias_corrected_momentum,
            'velocity_corrected': bias_corrected_velocity,
            'losses': [step.loss for step in steps]
        }
    
    def create_parameter_trajectory_3d(self, optimizer_result: OptimizationResult,
                                     loss_function) -> Dict[str, any]:
        """
        Create 3D visualization of parameter trajectory with loss surface.
        
        Args:
            optimizer_result: Optimization result with parameter trajectory
            loss_function: Function to evaluate loss at any point
            
        Returns:
            Dictionary with 3D trajectory visualization data
        """
        steps = optimizer_result.optimization_steps
        
        # Extract parameter trajectory
        trajectory = np.array([step.parameters for step in steps])
        losses = np.array([step.loss for step in steps])
        
        # Create loss surface around trajectory
        x_min, x_max = trajectory[:, 0].min() - 1, trajectory[:, 0].max() + 1
        y_min, y_max = trajectory[:, 1].min() - 1, trajectory[:, 1].max() + 1
        
        x_surface = np.linspace(x_min, x_max, 30)
        y_surface = np.linspace(y_min, y_max, 30)
        X_surface, Y_surface = np.meshgrid(x_surface, y_surface)
        
        Z_surface = np.zeros_like(X_surface)
        for i in range(X_surface.shape[0]):
            for j in range(X_surface.shape[1]):
                Z_surface[i, j] = loss_function(np.array([X_surface[i, j], Y_surface[i, j]]))
        
        return {
            'trajectory_x': trajectory[:, 0],
            'trajectory_y': trajectory[:, 1],
            'trajectory_z': losses,
            'surface_x': X_surface,
            'surface_y': Y_surface,
            'surface_z': Z_surface,
            'start_point': trajectory[0],
            'end_point': trajectory[-1]
        }
    
    def create_gradient_flow_visualization(self, optimizer_result: OptimizationResult) -> Dict[str, any]:
        """
        Create visualization showing gradient flow and parameter updates.
        
        Args:
            optimizer_result: Optimization result with gradient information
            
        Returns:
            Dictionary with gradient flow visualization data
        """
        steps = optimizer_result.optimization_steps
        
        # Extract gradient and parameter data
        parameters = np.array([step.parameters for step in steps])
        gradients = np.array([step.gradients for step in steps])
        
        # Calculate parameter updates
        updates = np.diff(parameters, axis=0)
        
        # Calculate gradient magnitudes and directions
        grad_magnitudes = np.linalg.norm(gradients, axis=1)
        grad_directions = gradients / (grad_magnitudes[:, np.newaxis] + 1e-8)
        
        return {
            'iterations': list(range(len(steps))),
            'parameters': parameters,
            'gradients': gradients,
            'gradient_magnitudes': grad_magnitudes,
            'gradient_directions': grad_directions,
            'parameter_updates': updates,
            'update_magnitudes': np.linalg.norm(updates, axis=1) if len(updates) > 0 else []
        }