"""
Diffusion Models mathematical operations and computations.
Implements forward and reverse diffusion processes, denoising objectives, and sampling procedures.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import ComputationStep, NumericalExample, VisualizationData


@dataclass
class DiffusionParameters:
    """Parameters for diffusion model computation."""
    data_dim: int
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    model_hidden_dim: int = 256
    batch_size: int = 32


class DiffusionOperations:
    """Core diffusion model mathematical operations with step-by-step computations."""
    
    def __init__(self, params: DiffusionParameters):
        self.params = params
        
        # Compute noise schedule
        self.betas = self._compute_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        
        # Initialize simple denoising model weights
        self.model_weights = self._initialize_model_weights()
    
    def _compute_noise_schedule(self) -> np.ndarray:
        """Compute noise schedule β_t for forward diffusion process."""
        # Linear schedule (can be extended to cosine, etc.)
        return np.linspace(self.params.beta_start, self.params.beta_end, self.params.num_timesteps)
    
    def _initialize_model_weights(self) -> Dict[str, np.ndarray]:
        """Initialize simple denoising model weights."""
        return {
            'W1': np.random.randn(self.params.data_dim + 1, self.params.model_hidden_dim) * 0.1,  # +1 for time embedding
            'b1': np.zeros(self.params.model_hidden_dim),
            'W2': np.random.randn(self.params.model_hidden_dim, self.params.data_dim) * 0.1,
            'b2': np.zeros(self.params.data_dim)
        }
    
    def forward_diffusion_step(self, x_t_minus_1: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single forward diffusion step: q(x_t | x_{t-1})
        x_t = √(1 - β_t) * x_{t-1} + √β_t * ε, where ε ~ N(0, I)
        """
        beta_t = self.betas[t]
        epsilon = np.random.randn(*x_t_minus_1.shape)
        
        x_t = np.sqrt(1 - beta_t) * x_t_minus_1 + np.sqrt(beta_t) * epsilon
        
        return x_t, epsilon
    
    def forward_diffusion_direct(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct forward diffusion: q(x_t | x_0)
        x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε, where ᾱ_t = ∏_{s=1}^t α_s
        """
        alpha_bar_t = self.alpha_bars[t]
        epsilon = np.random.randn(*x_0.shape)
        
        x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * epsilon
        
        return x_t, epsilon
    
    def reverse_diffusion_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        """
        Single reverse diffusion step: p_θ(x_{t-1} | x_t)
        Using DDPM parameterization: x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
        """
        if t == 0:
            return x_t  # No noise at t=0
        
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        
        # Compute mean of reverse process
        mean = (1.0 / np.sqrt(alpha_t)) * (x_t - (beta_t / np.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        # Compute variance (simplified)
        if t > 1:
            alpha_bar_t_minus_1 = self.alpha_bars[t-1]
            variance = beta_t * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)
        else:
            variance = 0.0
        
        # Sample from reverse process
        if variance > 0:
            z = np.random.randn(*x_t.shape)
            x_t_minus_1 = mean + np.sqrt(variance) * z
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    def predict_noise(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """
        Predict noise using simple denoising model: ε_θ(x_t, t)
        """
        # Simple time embedding (sinusoidal)
        time_emb = np.sin(t / self.params.num_timesteps * 2 * np.pi)
        
        # Concatenate time embedding to input
        batch_size = x_t.shape[0]
        time_input = np.full((batch_size, 1), time_emb)
        model_input = np.concatenate([x_t, time_input], axis=1)
        
        # Forward pass through simple MLP
        h1 = np.maximum(0, np.dot(model_input, self.model_weights['W1']) + self.model_weights['b1'])
        predicted_noise = np.dot(h1, self.model_weights['W2']) + self.model_weights['b2']
        
        return predicted_noise
    
    def compute_denoising_loss(self, x_0: np.ndarray, t: int) -> Dict[str, float]:
        """
        Compute denoising objective: L_simple = E[||ε - ε_θ(x_t, t)||²]
        """
        # Forward diffusion to get noisy sample
        x_t, true_noise = self.forward_diffusion_direct(x_0, t)
        
        # Predict noise
        predicted_noise = self.predict_noise(x_t, t)
        
        # Compute MSE loss
        mse_loss = np.mean((true_noise - predicted_noise) ** 2)
        
        # Compute signal-to-noise ratio
        signal_power = np.mean(x_0 ** 2)
        noise_power = np.mean(true_noise ** 2)
        snr = signal_power / (noise_power + 1e-8)
        
        return {
            'mse_loss': mse_loss,
            'signal_to_noise_ratio': snr,
            'timestep': t,
            'alpha_bar_t': self.alpha_bars[t],
            'noise_level': np.sqrt(1 - self.alpha_bars[t])
        }
    
    def compute_variational_lower_bound(self, x_0: np.ndarray) -> Dict[str, float]:
        """
        Compute variational lower bound (ELBO) for diffusion models.
        L_vlb = L_T + L_{T-1} + ... + L_1 + L_0
        """
        total_loss = 0.0
        timestep_losses = []
        
        for t in range(1, self.params.num_timesteps):
            # Sample timestep uniformly
            loss_info = self.compute_denoising_loss(x_0, t)
            timestep_losses.append(loss_info['mse_loss'])
            total_loss += loss_info['mse_loss']
        
        # L_0 term (reconstruction)
        x_1, _ = self.forward_diffusion_direct(x_0, 1)
        predicted_noise_1 = self.predict_noise(x_1, 1)
        x_0_pred = self.reverse_diffusion_step(x_1, 1, predicted_noise_1)
        reconstruction_loss = np.mean((x_0 - x_0_pred) ** 2)
        
        total_loss += reconstruction_loss
        
        return {
            'total_vlb_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'denoising_losses': timestep_losses,
            'mean_denoising_loss': np.mean(timestep_losses)
        }
    
    def sample_ddpm(self, shape: Tuple[int, ...], num_steps: Optional[int] = None) -> List[np.ndarray]:
        """
        DDPM sampling procedure: start from noise and iteratively denoise.
        """
        if num_steps is None:
            num_steps = self.params.num_timesteps
        
        # Start from pure noise
        x_t = np.random.randn(*shape)
        samples = [x_t.copy()]
        
        # Reverse diffusion process
        for t in reversed(range(1, num_steps)):
            predicted_noise = self.predict_noise(x_t, t)
            x_t = self.reverse_diffusion_step(x_t, t, predicted_noise)
            samples.append(x_t.copy())
        
        return samples
    
    def sample_ddim(self, shape: Tuple[int, ...], num_steps: int = 50, eta: float = 0.0) -> List[np.ndarray]:
        """
        DDIM sampling: deterministic sampling with fewer steps.
        """
        # Create subset of timesteps
        timesteps = np.linspace(0, self.params.num_timesteps - 1, num_steps, dtype=int)
        
        # Start from pure noise
        x_t = np.random.randn(*shape)
        samples = [x_t.copy()]
        
        for i in reversed(range(1, len(timesteps))):
            t = timesteps[i]
            t_prev = timesteps[i-1] if i > 0 else 0
            
            # Predict noise
            predicted_noise = self.predict_noise(x_t, t)
            
            # DDIM update
            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_t_prev = self.alpha_bars[t_prev] if t_prev > 0 else 1.0
            
            # Predict x_0
            x_0_pred = (x_t - np.sqrt(1 - alpha_bar_t) * predicted_noise) / np.sqrt(alpha_bar_t)
            
            # Compute direction to x_t
            direction = np.sqrt(1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)) * predicted_noise
            
            # Add noise (eta controls stochasticity)
            noise = eta * np.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * np.sqrt(1 - alpha_bar_t / alpha_bar_t_prev) * np.random.randn(*shape)
            
            # Update
            x_t = np.sqrt(alpha_bar_t_prev) * x_0_pred + direction + noise
            samples.append(x_t.copy())
        
        return samples
    
    def analyze_noise_schedule(self) -> Dict[str, np.ndarray]:
        """Analyze the noise schedule and its properties."""
        
        # Signal-to-noise ratio over time
        snr = self.alpha_bars / (1 - self.alpha_bars)
        
        # Noise level over time
        noise_levels = np.sqrt(1 - self.alpha_bars)
        
        # Signal preservation over time
        signal_levels = np.sqrt(self.alpha_bars)
        
        return {
            'timesteps': np.arange(self.params.num_timesteps),
            'betas': self.betas,
            'alphas': self.alphas,
            'alpha_bars': self.alpha_bars,
            'snr': snr,
            'noise_levels': noise_levels,
            'signal_levels': signal_levels
        }
    
    def generate_diffusion_process_steps(self) -> List[ComputationStep]:
        """Generate step-by-step diffusion process computation."""
        
        # Create sample data
        x_0 = np.random.randn(4, self.params.data_dim)
        t = self.params.num_timesteps // 2  # Middle timestep
        
        steps = []
        
        # Step 1: Forward diffusion
        x_t, true_noise = self.forward_diffusion_direct(x_0, t)
        steps.append(ComputationStep(
            step_number=1,
            operation_name="forward_diffusion",
            input_values={"x_0": x_0, "t": np.array([t])},
            operation_description=f"Forward diffusion: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε",
            output_values={"x_t": x_t, "true_noise": true_noise, "alpha_bar_t": np.array([self.alpha_bars[t]])},
            visualization_hints={"highlight": "forward_process"}
        ))
        
        # Step 2: Noise prediction
        predicted_noise = self.predict_noise(x_t, t)
        steps.append(ComputationStep(
            step_number=2,
            operation_name="noise_prediction",
            input_values={"x_t": x_t, "t": np.array([t])},
            operation_description="Denoising model predicts noise: ε_θ(x_t, t)",
            output_values={"predicted_noise": predicted_noise},
            visualization_hints={"highlight": "denoising_model"}
        ))
        
        # Step 3: Denoising loss
        loss_info = self.compute_denoising_loss(x_0, t)
        steps.append(ComputationStep(
            step_number=3,
            operation_name="denoising_loss",
            input_values={"true_noise": true_noise, "predicted_noise": predicted_noise},
            operation_description="L_simple = E[||ε - ε_θ(x_t, t)||²]",
            output_values=loss_info,
            visualization_hints={"highlight": "loss_computation"}
        ))
        
        # Step 4: Reverse diffusion
        x_t_minus_1 = self.reverse_diffusion_step(x_t, t, predicted_noise)
        steps.append(ComputationStep(
            step_number=4,
            operation_name="reverse_diffusion",
            input_values={"x_t": x_t, "predicted_noise": predicted_noise},
            operation_description="Reverse step: x_{t-1} = μ_θ(x_t, t) + σ_t * z",
            output_values={"x_t_minus_1": x_t_minus_1},
            visualization_hints={"highlight": "reverse_process"}
        ))
        
        # Step 5: Sampling process
        samples = self.sample_ddpm((2, self.params.data_dim), num_steps=10)
        steps.append(ComputationStep(
            step_number=5,
            operation_name="sampling_process",
            input_values={"noise": samples[0]},
            operation_description="DDPM sampling: iterative denoising from pure noise",
            output_values={"samples": np.array(samples), "final_sample": samples[-1]},
            visualization_hints={"highlight": "sampling_trajectory"}
        ))
        
        return steps


def create_diffusion_numerical_example() -> NumericalExample:
    """Create a comprehensive diffusion models numerical example."""
    
    params = DiffusionParameters(
        data_dim=784,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        model_hidden_dim=256,
        batch_size=32
    )
    diffusion_ops = DiffusionOperations(params)
    
    # Generate computation steps
    computation_steps = diffusion_ops.generate_diffusion_process_steps()
    
    # Create visualization data
    viz_data = VisualizationData(
        visualization_type="diffusion_process",
        data={
            "num_timesteps": params.num_timesteps,
            "data_dim": params.data_dim,
            "noise_schedule": "linear",
            "sampling_methods": ["ddpm", "ddim"]
        },
        color_mappings={
            "forward_process": "#e74c3c",
            "reverse_process": "#27ae60",
            "denoising_model": "#3498db",
            "noise_schedule": "#f39c12",
            "sampling": "#9b59b6"
        },
        interactive_elements=["noise_schedule_control", "sampling_visualization", "timestep_analysis"]
    )
    
    return NumericalExample(
        example_id="diffusion_denoising_process",
        description="Complete diffusion model with forward/reverse processes and denoising objective",
        input_values={"clean_data": np.random.randn(params.batch_size, params.data_dim)},
        computation_steps=computation_steps,
        output_values={"denoised_sample": np.random.randn(params.data_dim)},
        visualization_data=viz_data,
        educational_notes=[
            "Forward diffusion gradually adds Gaussian noise over T timesteps",
            "Reverse diffusion learns to denoise by predicting the added noise",
            "The denoising objective L_simple = E[||ε - ε_θ(x_t, t)||²] is key to training",
            "DDPM and DDIM offer different sampling trade-offs between quality and speed",
            "Noise schedule β_t controls the rate of noise addition in forward process"
        ]
    )