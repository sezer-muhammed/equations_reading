"""
Variational Autoencoder (VAE) mathematical operations and computations.
Implements ELBO derivation, encoder-decoder architecture, and reparameterization trick.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import ComputationStep, NumericalExample, VisualizationData


@dataclass
class VAEParameters:
    """Parameters for VAE computation."""
    input_dim: int
    latent_dim: int
    hidden_dim: int
    batch_size: int = 32


class VAEOperations:
    """Core VAE mathematical operations with step-by-step computations."""
    
    def __init__(self, params: VAEParameters):
        self.params = params
        
    def compute_elbo_components(self, x: np.ndarray, mu: np.ndarray, 
                               log_var: np.ndarray, x_recon: np.ndarray) -> Dict[str, float]:
        """
        Compute ELBO components: reconstruction loss and KL divergence.
        
        ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
        """
        # Reconstruction loss (negative log-likelihood)
        reconstruction_loss = np.mean(np.sum((x - x_recon) ** 2, axis=1))
        
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = 0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_divergence = -0.5 * np.mean(np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1))
        
        # ELBO (Evidence Lower BOund)
        elbo = -reconstruction_loss - kl_divergence
        
        return {
            'reconstruction_loss': reconstruction_loss,
            'kl_divergence': kl_divergence,
            'elbo': elbo,
            'negative_elbo': -elbo  # This is what we minimize
        }
    
    def reparameterization_trick(self, mu: np.ndarray, log_var: np.ndarray, 
                                epsilon: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Implement reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)
        """
        if epsilon is None:
            epsilon = np.random.standard_normal(mu.shape)
        
        # σ = exp(0.5 * log_var)
        sigma = np.exp(0.5 * log_var)
        
        # z = μ + σ * ε
        z = mu + sigma * epsilon
        
        return z, epsilon, sigma
    
    def encoder_forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified encoder forward pass returning mean and log variance.
        """
        # Simplified linear transformation for demonstration
        # In practice, this would be a neural network
        W_mu = np.random.randn(self.params.input_dim, self.params.latent_dim) * 0.1
        W_logvar = np.random.randn(self.params.input_dim, self.params.latent_dim) * 0.1
        
        mu = np.dot(x, W_mu)
        log_var = np.dot(x, W_logvar)
        
        return mu, log_var
    
    def decoder_forward(self, z: np.ndarray) -> np.ndarray:
        """
        Simplified decoder forward pass.
        """
        # Simplified linear transformation for demonstration
        W_dec = np.random.randn(self.params.latent_dim, self.params.input_dim) * 0.1
        x_recon = np.dot(z, W_dec)
        
        return x_recon
    
    def generate_elbo_derivation_steps(self) -> List[ComputationStep]:
        """Generate step-by-step ELBO derivation with numerical examples."""
        
        # Create sample data
        x = np.random.randn(4, self.params.input_dim)
        mu, log_var = self.encoder_forward(x)
        z, epsilon, sigma = self.reparameterization_trick(mu, log_var)
        x_recon = self.decoder_forward(z)
        
        steps = []
        
        # Step 1: Encoder output
        steps.append(ComputationStep(
            step_number=1,
            operation_name="encoder_output",
            input_values={"x": x},
            operation_description="Encoder produces mean μ and log variance log(σ²)",
            output_values={"mu": mu, "log_var": log_var},
            visualization_hints={"highlight": "encoder_params"}
        ))
        
        # Step 2: Reparameterization trick
        steps.append(ComputationStep(
            step_number=2,
            operation_name="reparameterization",
            input_values={"mu": mu, "log_var": log_var, "epsilon": epsilon},
            operation_description="z = μ + σ * ε, where σ = exp(0.5 * log(σ²))",
            output_values={"z": z, "sigma": sigma},
            visualization_hints={"highlight": "sampling_process"}
        ))
        
        # Step 3: Decoder reconstruction
        steps.append(ComputationStep(
            step_number=3,
            operation_name="decoder_reconstruction",
            input_values={"z": z},
            operation_description="Decoder reconstructs x̂ from latent z",
            output_values={"x_recon": x_recon},
            visualization_hints={"highlight": "decoder_params"}
        ))
        
        # Step 4: ELBO computation
        elbo_components = self.compute_elbo_components(x, mu, log_var, x_recon)
        steps.append(ComputationStep(
            step_number=4,
            operation_name="elbo_computation",
            input_values={"x": x, "x_recon": x_recon, "mu": mu, "log_var": log_var},
            operation_description="ELBO = -Reconstruction_Loss - KL_Divergence",
            output_values=elbo_components,
            visualization_hints={"highlight": "loss_components"}
        ))
        
        return steps
    
    def generate_latent_interpolation(self, z1: np.ndarray, z2: np.ndarray, 
                                    num_steps: int = 10) -> List[np.ndarray]:
        """Generate latent space interpolation between two points."""
        interpolations = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = self.decoder_forward(z_interp)
            interpolations.append(x_interp)
        
        return interpolations


def create_vae_numerical_example() -> NumericalExample:
    """Create a comprehensive VAE numerical example."""
    
    params = VAEParameters(input_dim=784, latent_dim=20, hidden_dim=400)
    vae_ops = VAEOperations(params)
    
    # Generate computation steps
    computation_steps = vae_ops.generate_elbo_derivation_steps()
    
    # Create visualization data
    viz_data = VisualizationData(
        visualization_type="vae_architecture",
        data={
            "encoder_dims": [params.input_dim, params.hidden_dim, params.latent_dim * 2],
            "decoder_dims": [params.latent_dim, params.hidden_dim, params.input_dim],
            "latent_dim": params.latent_dim
        },
        color_mappings={
            "encoder": "#3498db",
            "decoder": "#e74c3c", 
            "latent": "#f39c12",
            "reconstruction": "#27ae60"
        },
        interactive_elements=["latent_interpolation", "parameter_adjustment"]
    )
    
    return NumericalExample(
        example_id="vae_elbo_derivation",
        description="Complete VAE ELBO derivation with encoder-decoder architecture",
        input_values={"input_data": np.random.randn(32, params.input_dim)},
        computation_steps=computation_steps,
        output_values={"elbo": np.array([-150.5])},
        visualization_data=viz_data,
        educational_notes=[
            "The reparameterization trick enables backpropagation through stochastic nodes",
            "KL divergence acts as a regularizer, preventing posterior collapse",
            "ELBO provides a tractable lower bound for the intractable marginal likelihood"
        ]
    )