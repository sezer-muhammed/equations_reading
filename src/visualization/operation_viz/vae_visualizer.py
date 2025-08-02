"""
VAE-specific visualization components for architecture, latent space, and operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import ColorCodedMatrix, OperationVisualization, AnimationFrame, HighlightPattern


@dataclass
class VAEVisualizationConfig:
    """Configuration for VAE visualizations."""
    figsize: Tuple[int, int] = (12, 8)
    color_scheme: Dict[str, str] = None
    animation_speed: int = 1000  # milliseconds
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'encoder': '#3498db',
                'decoder': '#e74c3c', 
                'latent': '#f39c12',
                'reconstruction': '#27ae60',
                'kl_divergence': '#9b59b6',
                'prior': '#95a5a6'
            }


class VAEArchitectureVisualizer:
    """Visualizes VAE architecture and information flow."""
    
    def __init__(self, config: VAEVisualizationConfig = None):
        self.config = config or VAEVisualizationConfig()
    
    def create_architecture_diagram(self, input_dim: int, latent_dim: int, 
                                  hidden_dim: int) -> Dict[str, np.ndarray]:
        """Create VAE architecture visualization data."""
        
        # Define layer positions and sizes
        layers = {
            'input': {'pos': (0, 0), 'size': input_dim, 'color': self.config.color_scheme['encoder']},
            'encoder_hidden': {'pos': (2, 0), 'size': hidden_dim, 'color': self.config.color_scheme['encoder']},
            'mu': {'pos': (4, 0.5), 'size': latent_dim, 'color': self.config.color_scheme['latent']},
            'log_var': {'pos': (4, -0.5), 'size': latent_dim, 'color': self.config.color_scheme['latent']},
            'z_sample': {'pos': (6, 0), 'size': latent_dim, 'color': self.config.color_scheme['latent']},
            'decoder_hidden': {'pos': (8, 0), 'size': hidden_dim, 'color': self.config.color_scheme['decoder']},
            'reconstruction': {'pos': (10, 0), 'size': input_dim, 'color': self.config.color_scheme['reconstruction']}
        }
        
        return layers
    
    def create_reparameterization_visualization(self, mu: np.ndarray, log_var: np.ndarray,
                                              epsilon: np.ndarray) -> OperationVisualization:
        """Visualize the reparameterization trick step by step."""
        
        # Create color-coded matrices for each component
        mu_matrix = ColorCodedMatrix(
            matrix_data=mu.reshape(-1, 1),
            color_mapping={'mu': self.config.color_scheme['latent']},
            element_labels={(i, 0): f'μ_{i}' for i in range(len(mu))}
        )
        
        sigma = np.exp(0.5 * log_var)
        sigma_matrix = ColorCodedMatrix(
            matrix_data=sigma.reshape(-1, 1),
            color_mapping={'sigma': self.config.color_scheme['kl_divergence']},
            element_labels={(i, 0): f'σ_{i}' for i in range(len(sigma))}
        )
        
        epsilon_matrix = ColorCodedMatrix(
            matrix_data=epsilon.reshape(-1, 1),
            color_mapping={'epsilon': self.config.color_scheme['prior']},
            element_labels={(i, 0): f'ε_{i}' for i in range(len(epsilon))}
        )
        
        # Compute z = mu + sigma * epsilon
        z = mu + sigma * epsilon
        z_matrix = ColorCodedMatrix(
            matrix_data=z.reshape(-1, 1),
            color_mapping={'z': self.config.color_scheme['latent']},
            element_labels={(i, 0): f'z_{i}' for i in range(len(z))}
        )
        
        # Create animation frames
        frames = []
        
        # Frame 1: Show mu
        frames.append(AnimationFrame(
            frame_number=1,
            matrix_state=mu_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(i, 0) for i in range(len(mu))], 
                                        self.config.color_scheme['latent'], "Mean parameters")],
            description="Start with encoder mean output μ",
            duration_ms=self.config.animation_speed
        ))
        
        # Frame 2: Show sigma computation
        frames.append(AnimationFrame(
            frame_number=2,
            matrix_state=sigma_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(i, 0) for i in range(len(sigma))], 
                                        self.config.color_scheme['kl_divergence'], "Standard deviation")],
            description="Compute σ = exp(0.5 * log_var)",
            duration_ms=self.config.animation_speed
        ))
        
        # Frame 3: Show epsilon sampling
        frames.append(AnimationFrame(
            frame_number=3,
            matrix_state=epsilon_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(i, 0) for i in range(len(epsilon))], 
                                        self.config.color_scheme['prior'], "Random noise")],
            description="Sample ε ~ N(0, I)",
            duration_ms=self.config.animation_speed
        ))
        
        # Frame 4: Show final z computation
        frames.append(AnimationFrame(
            frame_number=4,
            matrix_state=z_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(i, 0) for i in range(len(z))], 
                                        self.config.color_scheme['latent'], "Latent sample")],
            description="Compute z = μ + σ ⊙ ε",
            duration_ms=self.config.animation_speed
        ))
        
        return OperationVisualization(
            operation_type="reparameterization_trick",
            input_matrices=[mu_matrix, sigma_matrix, epsilon_matrix],
            intermediate_steps=[],
            output_matrix=z_matrix,
            animation_sequence=frames
        )


class VAELatentSpaceVisualizer:
    """Visualizes VAE latent space and interpolations."""
    
    def __init__(self, config: VAEVisualizationConfig = None):
        self.config = config or VAEVisualizationConfig()
    
    def create_latent_space_plot(self, latent_samples: np.ndarray, 
                               labels: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Create 2D latent space visualization."""
        
        if latent_samples.shape[1] > 2:
            # Use PCA for dimensionality reduction if needed
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_samples)
        else:
            latent_2d = latent_samples
        
        # Create prior distribution grid for background
        x_range = np.linspace(-3, 3, 50)
        y_range = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Standard normal prior density
        prior_density = np.exp(-0.5 * (X**2 + Y**2)) / (2 * np.pi)
        
        return {
            'latent_points': latent_2d,
            'prior_grid_x': X,
            'prior_grid_y': Y,
            'prior_density': prior_density,
            'labels': labels
        }
    
    def create_interpolation_path(self, z1: np.ndarray, z2: np.ndarray, 
                                num_steps: int = 10) -> np.ndarray:
        """Create smooth interpolation path between two latent points."""
        
        alphas = np.linspace(0, 1, num_steps)
        interpolation_path = np.array([
            (1 - alpha) * z1 + alpha * z2 for alpha in alphas
        ])
        
        return interpolation_path
    
    def visualize_elbo_components(self, reconstruction_losses: np.ndarray,
                                kl_divergences: np.ndarray) -> Dict[str, np.ndarray]:
        """Visualize ELBO components over training."""
        
        total_loss = reconstruction_losses + kl_divergences
        elbo = -(total_loss)  # ELBO is negative of total loss
        
        return {
            'reconstruction_loss': reconstruction_losses,
            'kl_divergence': kl_divergences,
            'total_loss': total_loss,
            'elbo': elbo,
            'iterations': np.arange(len(reconstruction_losses))
        }


class VAEInteractiveComponents:
    """Interactive components for VAE tutorial."""
    
    def __init__(self, config: VAEVisualizationConfig = None):
        self.config = config or VAEVisualizationConfig()
    
    def create_parameter_slider_data(self, param_ranges: Dict[str, Tuple[float, float]]) -> Dict:
        """Create data for interactive parameter sliders."""
        
        sliders = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            sliders[param_name] = {
                'min': min_val,
                'max': max_val,
                'default': (min_val + max_val) / 2,
                'step': (max_val - min_val) / 100,
                'description': self._get_parameter_description(param_name)
            }
        
        return sliders
    
    def _get_parameter_description(self, param_name: str) -> str:
        """Get description for parameter sliders."""
        descriptions = {
            'beta': 'β-VAE weighting factor for KL divergence term',
            'latent_dim': 'Dimensionality of latent space',
            'learning_rate': 'Learning rate for optimization',
            'batch_size': 'Batch size for training'
        }
        return descriptions.get(param_name, f'Parameter: {param_name}')
    
    def create_latent_manipulation_interface(self, latent_dim: int) -> Dict:
        """Create interface for direct latent space manipulation."""
        
        return {
            'latent_sliders': [
                {
                    'name': f'z_{i}',
                    'min': -3.0,
                    'max': 3.0,
                    'default': 0.0,
                    'step': 0.1
                } for i in range(latent_dim)
            ],
            'preset_points': {
                'origin': np.zeros(latent_dim),
                'random_1': np.random.randn(latent_dim),
                'random_2': np.random.randn(latent_dim)
            },
            'interpolation_controls': {
                'num_steps': {'min': 5, 'max': 50, 'default': 10},
                'interpolation_type': ['linear', 'spherical']
            }
        }


def create_vae_visualization_suite() -> Dict[str, object]:
    """Create complete VAE visualization suite."""
    
    config = VAEVisualizationConfig()
    
    return {
        'architecture_visualizer': VAEArchitectureVisualizer(config),
        'latent_space_visualizer': VAELatentSpaceVisualizer(config),
        'interactive_components': VAEInteractiveComponents(config),
        'config': config
    }