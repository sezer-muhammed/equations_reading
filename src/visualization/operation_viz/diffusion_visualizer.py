"""
Diffusion Models-specific visualization components for forward/reverse processes, 
noise schedules, and sampling procedures.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import ColorCodedMatrix, OperationVisualization, AnimationFrame, HighlightPattern


@dataclass
class DiffusionVisualizationConfig:
    """Configuration for diffusion model visualizations."""
    figsize: Tuple[int, int] = (16, 10)
    color_scheme: Dict[str, str] = None
    animation_speed: int = 800  # milliseconds
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'clean_data': '#3498db',
                'noisy_data': '#e67e22',
                'pure_noise': '#95a5a6',
                'forward_process': '#e74c3c',
                'reverse_process': '#27ae60',
                'denoising_model': '#9b59b6',
                'noise_schedule': '#f39c12',
                'sampling': '#16a085',
                'loss_function': '#8e44ad'
            }


class DiffusionProcessVisualizer:
    """Visualizes forward and reverse diffusion processes."""
    
    def __init__(self, config: DiffusionVisualizationConfig = None):
        self.config = config or DiffusionVisualizationConfig()
    
    def create_diffusion_process_animation(self, x_0: np.ndarray, timesteps: List[int],
                                         alpha_bars: np.ndarray) -> OperationVisualization:
        """Create animation showing forward and reverse diffusion processes."""
        
        # Generate forward diffusion trajectory
        forward_samples = []
        for t in timesteps:
            if t == 0:
                x_t = x_0.copy()
            else:
                alpha_bar_t = alpha_bars[t]
                epsilon = np.random.randn(*x_0.shape)
                x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * epsilon
            forward_samples.append(x_t)
        
        # Create color-coded matrices for each timestep
        input_matrices = []
        for i, (t, x_t) in enumerate(zip(timesteps, forward_samples)):
            # Color based on noise level
            noise_level = np.sqrt(1 - alpha_bars[t]) if t > 0 else 0.0
            color_intensity = min(noise_level * 2, 1.0)  # Scale for visibility
            
            matrix = ColorCodedMatrix(
                matrix_data=x_t.reshape(-1, 1) if x_t.ndim == 1 else x_t,
                color_mapping={f'timestep_{t}': self._interpolate_color(
                    self.config.color_scheme['clean_data'],
                    self.config.color_scheme['pure_noise'],
                    color_intensity
                )},
                element_labels={(0, 0): f't={t}'}
            )
            input_matrices.append(matrix)
        
        # Create animation frames
        frames = []
        for i, (t, x_t) in enumerate(zip(timesteps, forward_samples)):
            # Forward process frame
            frames.append(AnimationFrame(
                frame_number=i * 2 + 1,
                matrix_state=input_matrices[i].matrix_data,
                highlights=[HighlightPattern(
                    "element", [(0, 0)], 
                    self.config.color_scheme['forward_process'],
                    f"Forward diffusion at t={t}"
                )],
                description=f"Forward: x_{t} = √ᾱ_{t} x_0 + √(1-ᾱ_{t}) ε",
                duration_ms=self.config.animation_speed
            ))
            
            # Reverse process frame (conceptual)
            if i < len(timesteps) - 1:
                frames.append(AnimationFrame(
                    frame_number=i * 2 + 2,
                    matrix_state=input_matrices[i].matrix_data,
                    highlights=[HighlightPattern(
                        "element", [(0, 0)],
                        self.config.color_scheme['reverse_process'],
                        f"Reverse denoising from t={t}"
                    )],
                    description=f"Reverse: x_{t-1} = μ_θ(x_{t}, {t}) + σ_{t} z",
                    duration_ms=self.config.animation_speed
                ))
        
        return OperationVisualization(
            operation_type="diffusion_process",
            input_matrices=input_matrices,
            intermediate_steps=[],
            output_matrix=input_matrices[0],  # Clean data
            animation_sequence=frames
        )
    
    def _interpolate_color(self, color1: str, color2: str, t: float) -> str:
        """Interpolate between two hex colors."""
        # Simple interpolation (could be improved with proper color space)
        return color1 if t < 0.5 else color2
    
    def create_noise_schedule_visualization(self, num_timesteps: int = 1000) -> Dict[str, np.ndarray]:
        """Create noise schedule visualization data."""
        
        timesteps = np.arange(num_timesteps)
        
        # Linear schedule
        beta_linear = np.linspace(1e-4, 0.02, num_timesteps)
        alpha_linear = 1.0 - beta_linear
        alpha_bar_linear = np.cumprod(alpha_linear)
        
        # Cosine schedule
        def cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = np.linspace(0, timesteps, steps)
            alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0, 0.999)
        
        beta_cosine = cosine_beta_schedule(num_timesteps)
        alpha_cosine = 1.0 - beta_cosine
        alpha_bar_cosine = np.cumprod(alpha_cosine)
        
        # Compute derived quantities
        snr_linear = alpha_bar_linear / (1 - alpha_bar_linear)
        snr_cosine = alpha_bar_cosine / (1 - alpha_bar_cosine)
        
        noise_level_linear = np.sqrt(1 - alpha_bar_linear)
        noise_level_cosine = np.sqrt(1 - alpha_bar_cosine)
        
        return {
            'timesteps': timesteps,
            'beta_linear': beta_linear,
            'beta_cosine': beta_cosine,
            'alpha_bar_linear': alpha_bar_linear,
            'alpha_bar_cosine': alpha_bar_cosine,
            'snr_linear': snr_linear,
            'snr_cosine': snr_cosine,
            'noise_level_linear': noise_level_linear,
            'noise_level_cosine': noise_level_cosine
        }


class DiffusionSamplingVisualizer:
    """Visualizes different sampling procedures for diffusion models."""
    
    def __init__(self, config: DiffusionVisualizationConfig = None):
        self.config = config or DiffusionVisualizationConfig()
    
    def create_sampling_comparison(self, sample_trajectories: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """Create visualization comparing different sampling methods."""
        
        comparison_data = {}
        
        for method_name, trajectory in sample_trajectories.items():
            # Compute trajectory statistics
            trajectory_array = np.array(trajectory)
            
            # Sample quality metrics (simplified)
            final_sample = trajectory[-1]
            sample_variance = np.var(final_sample)
            sample_mean = np.mean(final_sample)
            
            # Trajectory smoothness
            if len(trajectory) > 1:
                differences = [np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                             for i in range(len(trajectory)-1)]
                trajectory_smoothness = np.mean(differences)
            else:
                trajectory_smoothness = 0.0
            
            comparison_data[method_name] = {
                'trajectory': trajectory_array,
                'final_sample': final_sample,
                'sample_variance': sample_variance,
                'sample_mean': sample_mean,
                'trajectory_smoothness': trajectory_smoothness,
                'num_steps': len(trajectory)
            }
        
        return comparison_data
    
    def create_ddpm_vs_ddim_analysis(self, num_timesteps: int = 1000) -> Dict[str, np.ndarray]:
        """Create analysis comparing DDPM and DDIM sampling characteristics."""
        
        timesteps = np.arange(num_timesteps)
        
        # DDPM: uses all timesteps
        ddpm_steps = timesteps
        ddpm_stochasticity = np.ones_like(ddpm_steps)  # Fully stochastic
        
        # DDIM: uses subset of timesteps
        ddim_num_steps = 50
        ddim_steps = np.linspace(0, num_timesteps-1, ddim_num_steps, dtype=int)
        ddim_stochasticity = np.zeros_like(ddim_steps)  # Deterministic (η=0)
        
        # Quality vs speed trade-off (conceptual)
        ddpm_quality = 1.0  # Reference quality
        ddpm_speed = 1.0 / len(ddpm_steps)  # Inversely proportional to steps
        
        ddim_quality = 0.95  # Slightly lower quality
        ddim_speed = 1.0 / len(ddim_steps)  # Much faster
        
        return {
            'ddpm_steps': ddpm_steps,
            'ddim_steps': ddim_steps,
            'ddpm_stochasticity': ddpm_stochasticity,
            'ddim_stochasticity': ddim_stochasticity,
            'quality_comparison': np.array([ddpm_quality, ddim_quality]),
            'speed_comparison': np.array([ddpm_speed, ddim_speed]),
            'method_names': ['DDPM', 'DDIM']
        }
    
    def create_sampling_trajectory_visualization(self, trajectory: List[np.ndarray], 
                                               method_name: str) -> Dict[str, np.ndarray]:
        """Create detailed visualization of a single sampling trajectory."""
        
        trajectory_array = np.array(trajectory)
        num_steps = len(trajectory)
        
        # Compute step-wise changes
        step_changes = []
        for i in range(1, num_steps):
            change = np.linalg.norm(trajectory[i] - trajectory[i-1])
            step_changes.append(change)
        
        step_changes = np.array(step_changes)
        
        # Compute cumulative denoising
        if len(trajectory) > 1:
            initial_noise_level = np.linalg.norm(trajectory[0])
            cumulative_denoising = []
            for sample in trajectory:
                current_noise = np.linalg.norm(sample - trajectory[-1])  # Distance from final
                denoising_progress = 1.0 - (current_noise / initial_noise_level)
                cumulative_denoising.append(max(0, denoising_progress))
            cumulative_denoising = np.array(cumulative_denoising)
        else:
            cumulative_denoising = np.array([1.0])
        
        return {
            'trajectory': trajectory_array,
            'step_indices': np.arange(num_steps),
            'step_changes': step_changes,
            'cumulative_denoising': cumulative_denoising,
            'method_name': method_name,
            'total_steps': num_steps
        }


class DiffusionLossVisualizer:
    """Visualizes diffusion model loss functions and training dynamics."""
    
    def __init__(self, config: DiffusionVisualizationConfig = None):
        self.config = config or DiffusionVisualizationConfig()
    
    def create_denoising_loss_visualization(self, true_noise: np.ndarray, 
                                          predicted_noise: np.ndarray,
                                          timestep: int) -> OperationVisualization:
        """Visualize the denoising loss computation."""
        
        # Create color-coded matrices
        true_noise_matrix = ColorCodedMatrix(
            matrix_data=true_noise.reshape(-1, 1) if true_noise.ndim == 1 else true_noise,
            color_mapping={'true_noise': self.config.color_scheme['pure_noise']},
            element_labels={(0, 0): 'ε_true'}
        )
        
        predicted_noise_matrix = ColorCodedMatrix(
            matrix_data=predicted_noise.reshape(-1, 1) if predicted_noise.ndim == 1 else predicted_noise,
            color_mapping={'predicted_noise': self.config.color_scheme['denoising_model']},
            element_labels={(0, 0): 'ε_θ'}
        )
        
        # Compute loss
        loss = np.mean((true_noise - predicted_noise) ** 2)
        loss_matrix = ColorCodedMatrix(
            matrix_data=np.array([[loss]]),
            color_mapping={'loss': self.config.color_scheme['loss_function']},
            element_labels={(0, 0): f'L_t={timestep}'}
        )
        
        # Create animation frames
        frames = []
        
        # Frame 1: Show true noise
        frames.append(AnimationFrame(
            frame_number=1,
            matrix_state=true_noise_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(0, 0)], 
                                        self.config.color_scheme['pure_noise'],
                                        "True noise added in forward process")],
            description="True noise ε sampled from N(0, I)",
            duration_ms=self.config.animation_speed
        ))
        
        # Frame 2: Show predicted noise
        frames.append(AnimationFrame(
            frame_number=2,
            matrix_state=predicted_noise_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(0, 0)],
                                        self.config.color_scheme['denoising_model'],
                                        "Noise predicted by denoising model")],
            description=f"Predicted noise ε_θ(x_t, {timestep})",
            duration_ms=self.config.animation_speed
        ))
        
        # Frame 3: Show loss computation
        frames.append(AnimationFrame(
            frame_number=3,
            matrix_state=loss_matrix.matrix_data,
            highlights=[HighlightPattern("element", [(0, 0)],
                                        self.config.color_scheme['loss_function'],
                                        "MSE loss between true and predicted noise")],
            description="L_simple = ||ε - ε_θ(x_t, t)||²",
            duration_ms=self.config.animation_speed
        ))
        
        return OperationVisualization(
            operation_type="denoising_loss",
            input_matrices=[true_noise_matrix, predicted_noise_matrix],
            intermediate_steps=[],
            output_matrix=loss_matrix,
            animation_sequence=frames
        )
    
    def create_loss_landscape_analysis(self, timesteps: np.ndarray, 
                                     losses: np.ndarray) -> Dict[str, np.ndarray]:
        """Create loss landscape analysis across timesteps."""
        
        # Compute loss statistics
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        # Identify challenging timesteps (high loss)
        loss_threshold = mean_loss + std_loss
        challenging_timesteps = timesteps[losses > loss_threshold]
        
        # Compute loss gradient (rate of change)
        if len(losses) > 1:
            loss_gradient = np.gradient(losses)
        else:
            loss_gradient = np.zeros_like(losses)
        
        # Smooth loss curve for visualization
        if len(losses) > 10:
            from scipy.ndimage import gaussian_filter1d
            smooth_losses = gaussian_filter1d(losses, sigma=2)
        else:
            smooth_losses = losses.copy()
        
        return {
            'timesteps': timesteps,
            'raw_losses': losses,
            'smooth_losses': smooth_losses,
            'loss_gradient': loss_gradient,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'challenging_timesteps': challenging_timesteps,
            'loss_threshold': loss_threshold
        }
    
    def create_vlb_decomposition(self, num_timesteps: int = 1000) -> Dict[str, np.ndarray]:
        """Create variational lower bound decomposition visualization."""
        
        timesteps = np.arange(1, num_timesteps + 1)
        
        # Simulate VLB components (in practice, these would be computed from actual training)
        # L_T term (prior matching)
        L_T = 0.1  # Typically small
        
        # L_t terms (denoising)
        # Higher loss at intermediate timesteps where noise level is moderate
        L_t = 2.0 * np.exp(-0.5 * ((timesteps - num_timesteps/2) / (num_timesteps/4))**2)
        
        # L_0 term (reconstruction)
        L_0 = 0.5
        
        # Total VLB
        total_vlb = L_T + np.sum(L_t) + L_0
        
        # Simplified objective (uniform weighting)
        L_simple = np.mean(L_t)
        
        return {
            'timesteps': timesteps,
            'L_t_terms': L_t,
            'L_T_term': L_T,
            'L_0_term': L_0,
            'total_vlb': total_vlb,
            'L_simple': L_simple,
            'vlb_components': ['L_T', 'L_t', 'L_0'],
            'component_values': [L_T, np.sum(L_t), L_0]
        }


class DiffusionInteractiveComponents:
    """Interactive components for diffusion models tutorial."""
    
    def __init__(self, config: DiffusionVisualizationConfig = None):
        self.config = config or DiffusionVisualizationConfig()
    
    def create_noise_schedule_controller(self) -> Dict[str, Dict]:
        """Create interactive noise schedule controller."""
        
        return {
            'schedule_parameters': {
                'schedule_type': {'options': ['linear', 'cosine', 'sigmoid'], 'default': 'linear'},
                'beta_start': {'min': 1e-6, 'max': 1e-2, 'default': 1e-4, 'step': 1e-6},
                'beta_end': {'min': 1e-3, 'max': 0.1, 'default': 0.02, 'step': 1e-3},
                'num_timesteps': {'min': 100, 'max': 2000, 'default': 1000, 'step': 100}
            },
            'cosine_parameters': {
                's_offset': {'min': 0.001, 'max': 0.02, 'default': 0.008, 'step': 0.001}
            },
            'visualization_options': {
                'show_beta_schedule': True,
                'show_alpha_bar': True,
                'show_snr': True,
                'show_noise_level': True,
                'log_scale': False
            }
        }
    
    def create_sampling_controller(self) -> Dict[str, Dict]:
        """Create interactive sampling procedure controller."""
        
        return {
            'sampling_method': {
                'method': {'options': ['DDPM', 'DDIM', 'DPM-Solver'], 'default': 'DDPM'},
                'num_steps': {'min': 10, 'max': 1000, 'default': 50, 'step': 10},
                'eta': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 0.1}  # For DDIM
            },
            'generation_parameters': {
                'batch_size': {'min': 1, 'max': 16, 'default': 4, 'step': 1},
                'guidance_scale': {'min': 1.0, 'max': 20.0, 'default': 7.5, 'step': 0.5},
                'seed': {'min': 0, 'max': 999999, 'default': 42, 'step': 1}
            },
            'visualization_options': {
                'show_trajectory': True,
                'show_intermediate_steps': True,
                'animate_process': True,
                'step_interval': {'min': 1, 'max': 100, 'default': 10, 'step': 1}
            }
        }
    
    def create_training_analyzer(self) -> Dict[str, Dict]:
        """Create training analysis interface."""
        
        return {
            'loss_analysis': {
                'loss_type': {'options': ['L_simple', 'L_vlb', 'weighted_vlb'], 'default': 'L_simple'},
                'timestep_sampling': {'options': ['uniform', 'importance', 'cosine'], 'default': 'uniform'},
                'show_per_timestep': True,
                'show_gradient_norms': True
            },
            'model_analysis': {
                'architecture': {'options': ['UNet', 'Transformer', 'CNN'], 'default': 'UNet'},
                'attention_layers': {'min': 0, 'max': 8, 'default': 4, 'step': 1},
                'hidden_dim': {'min': 64, 'max': 1024, 'default': 256, 'step': 64}
            },
            'convergence_metrics': {
                'track_fid': True,
                'track_is': True,
                'track_lpips': True,
                'evaluation_frequency': {'min': 100, 'max': 5000, 'default': 1000, 'step': 100}
            }
        }
    
    def create_comparison_interface(self) -> Dict[str, Dict]:
        """Create interface for comparing with other generative models."""
        
        return {
            'model_comparison': {
                'models': {'options': ['Diffusion', 'GAN', 'VAE', 'Flow'], 'default': ['Diffusion', 'GAN']},
                'metrics': {'options': ['FID', 'IS', 'LPIPS', 'Precision', 'Recall'], 'default': ['FID', 'IS']},
                'datasets': {'options': ['CIFAR-10', 'CelebA', 'ImageNet'], 'default': 'CIFAR-10'}
            },
            'trade_off_analysis': {
                'quality_vs_speed': True,
                'memory_usage': True,
                'training_stability': True,
                'mode_coverage': True
            },
            'visualization_options': {
                'radar_chart': True,
                'bar_comparison': True,
                'scatter_plot': True,
                'table_view': True
            }
        }


def create_diffusion_visualization_suite() -> Dict[str, object]:
    """Create complete diffusion models visualization suite."""
    
    config = DiffusionVisualizationConfig()
    
    return {
        'process_visualizer': DiffusionProcessVisualizer(config),
        'sampling_visualizer': DiffusionSamplingVisualizer(config),
        'loss_visualizer': DiffusionLossVisualizer(config),
        'interactive_components': DiffusionInteractiveComponents(config),
        'config': config
    }