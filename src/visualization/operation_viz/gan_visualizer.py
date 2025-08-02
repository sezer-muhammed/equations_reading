"""
GAN-specific visualization components for adversarial training, Nash equilibrium, and mode collapse.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import ColorCodedMatrix, OperationVisualization, AnimationFrame, HighlightPattern


@dataclass
class GANVisualizationConfig:
    """Configuration for GAN visualizations."""
    figsize: Tuple[int, int] = (14, 10)
    color_scheme: Dict[str, str] = None
    animation_speed: int = 1500  # milliseconds
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'generator': '#27ae60',
                'discriminator': '#e74c3c',
                'real_data': '#3498db',
                'fake_data': '#f39c12',
                'minimax': '#9b59b6',
                'nash_equilibrium': '#34495e',
                'mode_collapse': '#e67e22',
                'training_stable': '#27ae60',
                'training_unstable': '#e74c3c'
            }


class GANArchitectureVisualizer:
    """Visualizes GAN architecture and adversarial training flow."""
    
    def __init__(self, config: GANVisualizationConfig = None):
        self.config = config or GANVisualizationConfig()
    
    def create_adversarial_architecture(self, noise_dim: int, data_dim: int,
                                      g_hidden_dim: int, d_hidden_dim: int) -> Dict[str, Dict]:
        """Create GAN adversarial architecture visualization data."""
        
        architecture = {
            'generator': {
                'layers': [
                    {'name': 'Noise Input', 'size': noise_dim, 'pos': (0, 2), 
                     'color': self.config.color_scheme['fake_data']},
                    {'name': 'G Hidden', 'size': g_hidden_dim, 'pos': (2, 2),
                     'color': self.config.color_scheme['generator']},
                    {'name': 'Generated Data', 'size': data_dim, 'pos': (4, 2),
                     'color': self.config.color_scheme['fake_data']}
                ],
                'connections': [
                    {'from': 0, 'to': 1, 'type': 'generator_forward'},
                    {'from': 1, 'to': 2, 'type': 'generator_output'}
                ]
            },
            'discriminator': {
                'layers': [
                    {'name': 'Real Data', 'size': data_dim, 'pos': (4, 0),
                     'color': self.config.color_scheme['real_data']},
                    {'name': 'Fake Data', 'size': data_dim, 'pos': (4, 2),
                     'color': self.config.color_scheme['fake_data']},
                    {'name': 'D Hidden', 'size': d_hidden_dim, 'pos': (6, 1),
                     'color': self.config.color_scheme['discriminator']},
                    {'name': 'Real/Fake', 'size': 1, 'pos': (8, 1),
                     'color': self.config.color_scheme['discriminator']}
                ],
                'connections': [
                    {'from': 0, 'to': 2, 'type': 'real_path'},
                    {'from': 1, 'to': 2, 'type': 'fake_path'},
                    {'from': 2, 'to': 3, 'type': 'discriminator_output'}
                ]
            },
            'adversarial_flow': {
                'generator_loss': {'pos': (4, 3), 'color': self.config.color_scheme['generator']},
                'discriminator_loss': {'pos': (8, 0), 'color': self.config.color_scheme['discriminator']},
                'minimax_game': {'pos': (6, -1), 'color': self.config.color_scheme['minimax']}
            }
        }
        
        return architecture
    
    def create_minimax_game_visualization(self, D_values: np.ndarray, G_values: np.ndarray,
                                        value_function: np.ndarray) -> OperationVisualization:
        """Visualize the minimax game dynamics."""
        
        # Create matrices for game components
        D_matrix = ColorCodedMatrix(
            matrix_data=D_values.reshape(-1, 1),
            color_mapping={'discriminator': self.config.color_scheme['discriminator']},
            element_labels={(i, 0): f'D_{i}' for i in range(len(D_values))}
        )
        
        G_matrix = ColorCodedMatrix(
            matrix_data=G_values.reshape(-1, 1),
            color_mapping={'generator': self.config.color_scheme['generator']},
            element_labels={(i, 0): f'G_{i}' for i in range(len(G_values))}
        )
        
        V_matrix = ColorCodedMatrix(
            matrix_data=value_function.reshape(-1, 1),
            color_mapping={'value_function': self.config.color_scheme['minimax']},
            element_labels={(i, 0): f'V_{i}' for i in range(len(value_function))}
        )
        
        # Create animation frames showing minimax dynamics
        frames = []
        
        # Frame 1: Discriminator maximization
        frames.append(AnimationFrame(
            frame_number=1,
            matrix_state=D_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(i, 0) for i in range(len(D_values))],
                                        self.config.color_scheme['discriminator'], "Discriminator maximizes V(D,G)")],
            description="Discriminator step: max_D V(D,G)",
            duration_ms=self.config.animation_speed
        ))
        
        # Frame 2: Generator minimization
        frames.append(AnimationFrame(
            frame_number=2,
            matrix_state=G_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(i, 0) for i in range(len(G_values))],
                                        self.config.color_scheme['generator'], "Generator minimizes V(D,G)")],
            description="Generator step: min_G V(D,G)",
            duration_ms=self.config.animation_speed
        ))
        
        # Frame 3: Value function evolution
        frames.append(AnimationFrame(
            frame_number=3,
            matrix_state=V_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(i, 0) for i in range(len(value_function))],
                                        self.config.color_scheme['minimax'], "Minimax value function")],
            description="V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]",
            duration_ms=self.config.animation_speed
        ))
        
        return OperationVisualization(
            operation_type="minimax_game",
            input_matrices=[D_matrix, G_matrix],
            intermediate_steps=[],
            output_matrix=V_matrix,
            animation_sequence=frames
        )


class GANTrainingDynamicsVisualizer:
    """Visualizes GAN training dynamics and convergence analysis."""
    
    def __init__(self, config: GANVisualizationConfig = None):
        self.config = config or GANVisualizationConfig()
    
    def create_training_curves(self, training_history: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        """Create training dynamics visualization data."""
        
        iterations = np.arange(len(training_history['D_loss']))
        
        return {
            'iterations': iterations,
            'discriminator_loss': np.array(training_history['D_loss']),
            'generator_loss': np.array(training_history['G_loss']),
            'D_real_output': np.array(training_history['D_real']),
            'D_fake_output': np.array(training_history['D_fake']),
            'value_function': np.array(training_history['value_function']),
            'nash_equilibrium_line': np.full_like(iterations, 0.5, dtype=float)  # D(x) = 0.5 at equilibrium
        }
    
    def analyze_convergence_patterns(self, training_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze convergence patterns and stability metrics."""
        
        D_real = training_data['D_real_output']
        D_fake = training_data['D_fake_output']
        
        # Convergence metrics
        final_window = max(10, len(D_real) // 10)  # Last 10% of training
        
        D_real_final = np.mean(D_real[-final_window:])
        D_fake_final = np.mean(D_fake[-final_window:])
        
        # Nash equilibrium proximity (closer to 0.5 is better)
        nash_distance = abs(D_real_final - 0.5) + abs(D_fake_final - 0.5)
        
        # Training stability (lower variance is better)
        D_real_stability = np.std(D_real[-final_window:])
        D_fake_stability = np.std(D_fake[-final_window:])
        
        # Mode collapse indicator (D_fake too low suggests collapse)
        mode_collapse_risk = 1.0 if D_fake_final < 0.1 else 0.0
        
        return {
            'nash_distance': nash_distance,
            'D_real_final': D_real_final,
            'D_fake_final': D_fake_final,
            'stability_score': 1.0 / (1.0 + D_real_stability + D_fake_stability),
            'mode_collapse_risk': mode_collapse_risk,
            'convergence_score': 1.0 / (1.0 + nash_distance)
        }
    
    def create_phase_portrait(self, D_loss_history: np.ndarray, 
                            G_loss_history: np.ndarray) -> Dict[str, np.ndarray]:
        """Create phase portrait of training dynamics."""
        
        # Create phase space trajectory
        trajectory = np.column_stack([D_loss_history, G_loss_history])
        
        # Compute vector field (simplified)
        D_gradient = np.gradient(D_loss_history)
        G_gradient = np.gradient(G_loss_history)
        
        return {
            'trajectory': trajectory,
            'D_gradient': D_gradient,
            'G_gradient': G_gradient,
            'equilibrium_point': np.array([np.mean(D_loss_history[-10:]), 
                                         np.mean(G_loss_history[-10:])])
        }


class GANModeCollapseVisualizer:
    """Visualizes mode collapse and diversity analysis."""
    
    def __init__(self, config: GANVisualizationConfig = None):
        self.config = config or GANVisualizationConfig()
    
    def create_mode_collapse_analysis(self, real_samples: np.ndarray, 
                                    generated_samples: np.ndarray,
                                    num_modes: int = 8) -> Dict[str, np.ndarray]:
        """Create mode collapse visualization data."""
        
        # Reduce to 2D for visualization if needed
        if real_samples.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            real_2d = pca.fit_transform(real_samples)
            generated_2d = pca.transform(generated_samples)
        else:
            real_2d = real_samples
            generated_2d = generated_samples
        
        # Identify modes in real data (simplified clustering)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_modes, random_state=42)
        real_modes = kmeans.fit_predict(real_2d)
        mode_centers = kmeans.cluster_centers_
        
        # Analyze mode coverage by generated samples
        generated_modes = kmeans.predict(generated_2d)
        mode_coverage = np.bincount(generated_modes, minlength=num_modes) / len(generated_samples)
        
        return {
            'real_samples_2d': real_2d,
            'generated_samples_2d': generated_2d,
            'real_mode_labels': real_modes,
            'generated_mode_labels': generated_modes,
            'mode_centers': mode_centers,
            'mode_coverage': mode_coverage,
            'collapsed_modes': np.where(mode_coverage < 0.05)[0]  # Modes with <5% coverage
        }
    
    def compute_diversity_metrics(self, generated_samples: np.ndarray) -> Dict[str, float]:
        """Compute diversity metrics for generated samples."""
        
        # Pairwise distances
        n_samples = len(generated_samples)
        pairwise_distances = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(generated_samples[i] - generated_samples[j])
                pairwise_distances.append(dist)
        
        pairwise_distances = np.array(pairwise_distances)
        
        # Diversity metrics
        mean_distance = np.mean(pairwise_distances)
        std_distance = np.std(pairwise_distances)
        min_distance = np.min(pairwise_distances)
        
        # Diversity score (higher is better)
        diversity_score = mean_distance / (std_distance + 1e-8)
        
        # Mode collapse indicator (very low min distance suggests collapse)
        collapse_indicator = 1.0 if min_distance < 0.01 else 0.0
        
        return {
            'mean_pairwise_distance': mean_distance,
            'std_pairwise_distance': std_distance,
            'min_pairwise_distance': min_distance,
            'diversity_score': diversity_score,
            'collapse_indicator': collapse_indicator,
            'effective_sample_size': len(np.unique(generated_samples.round(2), axis=0))
        }
    
    def create_diversity_evolution(self, sample_history: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Track diversity evolution during training."""
        
        diversity_scores = []
        collapse_indicators = []
        
        for samples in sample_history:
            metrics = self.compute_diversity_metrics(samples)
            diversity_scores.append(metrics['diversity_score'])
            collapse_indicators.append(metrics['collapse_indicator'])
        
        return {
            'training_steps': np.arange(len(sample_history)),
            'diversity_evolution': np.array(diversity_scores),
            'collapse_evolution': np.array(collapse_indicators),
            'diversity_trend': np.gradient(diversity_scores)
        }


class GANInteractiveComponents:
    """Interactive components for GAN tutorial."""
    
    def __init__(self, config: GANVisualizationConfig = None):
        self.config = config or GANVisualizationConfig()
    
    def create_training_controller(self) -> Dict[str, Dict]:
        """Create interactive training controller."""
        
        return {
            'hyperparameters': {
                'learning_rate_g': {'min': 1e-5, 'max': 1e-2, 'default': 2e-4, 'step': 1e-5},
                'learning_rate_d': {'min': 1e-5, 'max': 1e-2, 'default': 2e-4, 'step': 1e-5},
                'batch_size': {'min': 16, 'max': 256, 'default': 32, 'step': 16},
                'noise_dim': {'min': 10, 'max': 512, 'default': 100, 'step': 10}
            },
            'training_controls': {
                'num_d_steps': {'min': 1, 'max': 5, 'default': 1, 'step': 1},
                'num_g_steps': {'min': 1, 'max': 5, 'default': 1, 'step': 1},
                'training_iterations': {'min': 100, 'max': 10000, 'default': 1000, 'step': 100}
            },
            'visualization_options': {
                'show_loss_curves': True,
                'show_sample_evolution': True,
                'show_mode_analysis': True,
                'update_frequency': {'min': 10, 'max': 100, 'default': 50, 'step': 10}
            }
        }
    
    def create_architecture_explorer(self) -> Dict[str, Dict]:
        """Create interactive architecture exploration interface."""
        
        return {
            'generator_architecture': {
                'hidden_layers': {'min': 1, 'max': 5, 'default': 2, 'step': 1},
                'hidden_dim': {'min': 64, 'max': 1024, 'default': 256, 'step': 64},
                'activation': {'options': ['relu', 'leaky_relu', 'tanh'], 'default': 'relu'},
                'normalization': {'options': ['none', 'batch_norm', 'layer_norm'], 'default': 'batch_norm'}
            },
            'discriminator_architecture': {
                'hidden_layers': {'min': 1, 'max': 5, 'default': 2, 'step': 1},
                'hidden_dim': {'min': 64, 'max': 1024, 'default': 256, 'step': 64},
                'activation': {'options': ['relu', 'leaky_relu'], 'default': 'leaky_relu'},
                'dropout_rate': {'min': 0.0, 'max': 0.8, 'default': 0.3, 'step': 0.1}
            },
            'loss_variants': {
                'generator_loss': {'options': ['original', 'alternative', 'least_squares'], 'default': 'alternative'},
                'discriminator_loss': {'options': ['standard', 'least_squares', 'wasserstein'], 'default': 'standard'},
                'regularization': {'options': ['none', 'gradient_penalty', 'spectral_norm'], 'default': 'none'}
            }
        }
    
    def create_mode_collapse_simulator(self) -> Dict[str, Dict]:
        """Create mode collapse simulation interface."""
        
        return {
            'data_distribution': {
                'num_modes': {'min': 2, 'max': 10, 'default': 8, 'step': 1},
                'mode_separation': {'min': 1.0, 'max': 5.0, 'default': 2.0, 'step': 0.5},
                'mode_variance': {'min': 0.1, 'max': 1.0, 'default': 0.3, 'step': 0.1}
            },
            'collapse_factors': {
                'learning_rate_imbalance': {'min': 0.1, 'max': 10.0, 'default': 1.0, 'step': 0.1},
                'discriminator_strength': {'min': 0.5, 'max': 3.0, 'default': 1.0, 'step': 0.1},
                'generator_capacity': {'min': 0.1, 'max': 2.0, 'default': 1.0, 'step': 0.1}
            },
            'recovery_strategies': {
                'unrolled_optimization': {'enabled': False, 'steps': 5},
                'experience_replay': {'enabled': False, 'buffer_size': 1000},
                'minibatch_discrimination': {'enabled': False, 'feature_dim': 50}
            }
        }


def create_gan_visualization_suite() -> Dict[str, object]:
    """Create complete GAN visualization suite."""
    
    config = GANVisualizationConfig()
    
    return {
        'architecture_visualizer': GANArchitectureVisualizer(config),
        'training_dynamics_visualizer': GANTrainingDynamicsVisualizer(config),
        'mode_collapse_visualizer': GANModeCollapseVisualizer(config),
        'interactive_components': GANInteractiveComponents(config),
        'config': config
    }