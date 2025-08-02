"""
Generative Adversarial Network (GAN) mathematical operations and computations.
Implements min-max game theory, adversarial training dynamics, and Nash equilibrium analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import ComputationStep, NumericalExample, VisualizationData


@dataclass
class GANParameters:
    """Parameters for GAN computation."""
    noise_dim: int
    data_dim: int
    generator_hidden_dim: int
    discriminator_hidden_dim: int
    batch_size: int = 32
    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0002


class GANOperations:
    """Core GAN mathematical operations with step-by-step computations."""
    
    def __init__(self, params: GANParameters):
        self.params = params
        # Initialize simple linear weights for demonstration
        self.G_weights = self._initialize_generator_weights()
        self.D_weights = self._initialize_discriminator_weights()
        
    def _initialize_generator_weights(self) -> Dict[str, np.ndarray]:
        """Initialize generator network weights."""
        return {
            'W1': np.random.randn(self.params.noise_dim, self.params.generator_hidden_dim) * 0.1,
            'b1': np.zeros(self.params.generator_hidden_dim),
            'W2': np.random.randn(self.params.generator_hidden_dim, self.params.data_dim) * 0.1,
            'b2': np.zeros(self.params.data_dim)
        }
    
    def _initialize_discriminator_weights(self) -> Dict[str, np.ndarray]:
        """Initialize discriminator network weights."""
        return {
            'W1': np.random.randn(self.params.data_dim, self.params.discriminator_hidden_dim) * 0.1,
            'b1': np.zeros(self.params.discriminator_hidden_dim),
            'W2': np.random.randn(self.params.discriminator_hidden_dim, 1) * 0.1,
            'b2': np.zeros(1)
        }
    
    def generator_forward(self, z: np.ndarray) -> np.ndarray:
        """
        Generator forward pass: G(z) -> fake data
        """
        # Hidden layer with ReLU activation
        h1 = np.maximum(0, np.dot(z, self.G_weights['W1']) + self.G_weights['b1'])
        # Output layer with tanh activation
        output = np.tanh(np.dot(h1, self.G_weights['W2']) + self.G_weights['b2'])
        return output
    
    def discriminator_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Discriminator forward pass: D(x) -> probability of being real
        """
        # Hidden layer with LeakyReLU activation
        h1 = np.maximum(0.2 * (np.dot(x, self.D_weights['W1']) + self.D_weights['b1']),
                       np.dot(x, self.D_weights['W1']) + self.D_weights['b1'])
        # Output layer with sigmoid activation
        output = 1 / (1 + np.exp(-(np.dot(h1, self.D_weights['W2']) + self.D_weights['b2'])))
        return output
    
    def compute_discriminator_loss(self, real_data: np.ndarray, fake_data: np.ndarray) -> Dict[str, float]:
        """
        Compute discriminator loss: max E[log D(x)] + E[log(1 - D(G(z)))]
        """
        # Discriminator predictions
        D_real = self.discriminator_forward(real_data)
        D_fake = self.discriminator_forward(fake_data)
        
        # Binary cross-entropy loss components
        real_loss = -np.mean(np.log(D_real + 1e-8))  # -E[log D(x)]
        fake_loss = -np.mean(np.log(1 - D_fake + 1e-8))  # -E[log(1 - D(G(z)))]
        
        # Total discriminator loss (we minimize, so negate the max objective)
        total_loss = real_loss + fake_loss
        
        return {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'total_loss': total_loss,
            'D_real_mean': np.mean(D_real),
            'D_fake_mean': np.mean(D_fake)
        }
    
    def compute_generator_loss(self, fake_data: np.ndarray) -> Dict[str, float]:
        """
        Compute generator loss: min E[log(1 - D(G(z)))] or max E[log D(G(z))]
        """
        D_fake = self.discriminator_forward(fake_data)
        
        # Original formulation: min E[log(1 - D(G(z)))]
        original_loss = -np.mean(np.log(1 - D_fake + 1e-8))
        
        # Alternative formulation: max E[log D(G(z))] (better gradients)
        alternative_loss = -np.mean(np.log(D_fake + 1e-8))
        
        return {
            'original_loss': original_loss,
            'alternative_loss': alternative_loss,
            'D_fake_mean': np.mean(D_fake)
        }
    
    def compute_minimax_objective(self, real_data: np.ndarray, noise: np.ndarray) -> Dict[str, float]:
        """
        Compute the full minimax objective: min_G max_D V(D,G)
        V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
        """
        # Generate fake data
        fake_data = self.generator_forward(noise)
        
        # Discriminator predictions
        D_real = self.discriminator_forward(real_data)
        D_fake = self.discriminator_forward(fake_data)
        
        # Value function V(D,G)
        V_DG = np.mean(np.log(D_real + 1e-8)) + np.mean(np.log(1 - D_fake + 1e-8))
        
        return {
            'value_function': V_DG,
            'D_real_mean': np.mean(D_real),
            'D_fake_mean': np.mean(D_fake),
            'real_log_prob': np.mean(np.log(D_real + 1e-8)),
            'fake_log_prob': np.mean(np.log(1 - D_fake + 1e-8))
        }
    
    def analyze_nash_equilibrium(self, real_data: np.ndarray, num_iterations: int = 100) -> Dict[str, List[float]]:
        """
        Analyze convergence to Nash equilibrium through training dynamics.
        """
        history = {
            'D_loss': [],
            'G_loss': [],
            'D_real': [],
            'D_fake': [],
            'value_function': []
        }
        
        for iteration in range(num_iterations):
            # Sample noise
            noise = np.random.randn(self.params.batch_size, self.params.noise_dim)
            fake_data = self.generator_forward(noise)
            
            # Compute losses
            D_loss = self.compute_discriminator_loss(real_data, fake_data)
            G_loss = self.compute_generator_loss(fake_data)
            minimax = self.compute_minimax_objective(real_data, noise)
            
            # Store history
            history['D_loss'].append(D_loss['total_loss'])
            history['G_loss'].append(G_loss['alternative_loss'])
            history['D_real'].append(D_loss['D_real_mean'])
            history['D_fake'].append(D_loss['D_fake_mean'])
            history['value_function'].append(minimax['value_function'])
            
            # Simple gradient updates (for demonstration)
            self._update_discriminator_weights(D_loss)
            self._update_generator_weights(G_loss)
        
        return history
    
    def _update_discriminator_weights(self, loss_info: Dict[str, float]):
        """Simplified discriminator weight update."""
        # Simple gradient descent step (placeholder)
        lr = self.params.learning_rate_d
        for key in self.D_weights:
            self.D_weights[key] -= lr * np.random.randn(*self.D_weights[key].shape) * 0.01
    
    def _update_generator_weights(self, loss_info: Dict[str, float]):
        """Simplified generator weight update."""
        # Simple gradient descent step (placeholder)
        lr = self.params.learning_rate_g
        for key in self.G_weights:
            self.G_weights[key] -= lr * np.random.randn(*self.G_weights[key].shape) * 0.01
    
    def demonstrate_mode_collapse(self, real_data: np.ndarray, num_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Demonstrate mode collapse by analyzing generator output diversity.
        """
        # Generate multiple samples
        noise_samples = np.random.randn(num_samples, self.params.noise_dim)
        generated_samples = self.generator_forward(noise_samples)
        
        # Compute diversity metrics
        pairwise_distances = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                dist = np.linalg.norm(generated_samples[i] - generated_samples[j])
                pairwise_distances.append(dist)
        
        pairwise_distances = np.array(pairwise_distances)
        
        # Compare with real data diversity
        real_distances = []
        for i in range(min(num_samples, len(real_data))):
            for j in range(i + 1, min(num_samples, len(real_data))):
                dist = np.linalg.norm(real_data[i] - real_data[j])
                real_distances.append(dist)
        
        real_distances = np.array(real_distances)
        
        return {
            'generated_samples': generated_samples,
            'generated_distances': pairwise_distances,
            'real_distances': real_distances,
            'diversity_ratio': np.std(pairwise_distances) / (np.std(real_distances) + 1e-8),
            'mean_generated_distance': np.mean(pairwise_distances),
            'mean_real_distance': np.mean(real_distances)
        }
    
    def generate_adversarial_training_steps(self) -> List[ComputationStep]:
        """Generate step-by-step adversarial training computation."""
        
        # Create sample data
        real_data = np.random.randn(self.params.batch_size, self.params.data_dim)
        noise = np.random.randn(self.params.batch_size, self.params.noise_dim)
        
        steps = []
        
        # Step 1: Generate fake data
        fake_data = self.generator_forward(noise)
        steps.append(ComputationStep(
            step_number=1,
            operation_name="generator_forward",
            input_values={"noise": noise},
            operation_description="Generator creates fake data: G(z) → x_fake",
            output_values={"fake_data": fake_data},
            visualization_hints={"highlight": "generator_network"}
        ))
        
        # Step 2: Discriminator evaluation
        D_real = self.discriminator_forward(real_data)
        D_fake = self.discriminator_forward(fake_data)
        steps.append(ComputationStep(
            step_number=2,
            operation_name="discriminator_evaluation",
            input_values={"real_data": real_data, "fake_data": fake_data},
            operation_description="Discriminator evaluates real and fake data",
            output_values={"D_real": D_real, "D_fake": D_fake},
            visualization_hints={"highlight": "discriminator_network"}
        ))
        
        # Step 3: Compute discriminator loss
        D_loss = self.compute_discriminator_loss(real_data, fake_data)
        steps.append(ComputationStep(
            step_number=3,
            operation_name="discriminator_loss",
            input_values={"D_real": D_real, "D_fake": D_fake},
            operation_description="L_D = -E[log D(x)] - E[log(1 - D(G(z)))]",
            output_values=D_loss,
            visualization_hints={"highlight": "discriminator_loss"}
        ))
        
        # Step 4: Compute generator loss
        G_loss = self.compute_generator_loss(fake_data)
        steps.append(ComputationStep(
            step_number=4,
            operation_name="generator_loss",
            input_values={"D_fake": D_fake},
            operation_description="L_G = -E[log D(G(z))] (alternative formulation)",
            output_values=G_loss,
            visualization_hints={"highlight": "generator_loss"}
        ))
        
        # Step 5: Minimax objective
        minimax = self.compute_minimax_objective(real_data, noise)
        steps.append(ComputationStep(
            step_number=5,
            operation_name="minimax_objective",
            input_values={"real_data": real_data, "fake_data": fake_data},
            operation_description="V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]",
            output_values=minimax,
            visualization_hints={"highlight": "minimax_game"}
        ))
        
        return steps


def create_gan_numerical_example() -> NumericalExample:
    """Create a comprehensive GAN numerical example."""
    
    params = GANParameters(
        noise_dim=100, 
        data_dim=784, 
        generator_hidden_dim=256,
        discriminator_hidden_dim=256,
        batch_size=32
    )
    gan_ops = GANOperations(params)
    
    # Generate computation steps
    computation_steps = gan_ops.generate_adversarial_training_steps()
    
    # Create visualization data
    viz_data = VisualizationData(
        visualization_type="gan_architecture",
        data={
            "generator_dims": [params.noise_dim, params.generator_hidden_dim, params.data_dim],
            "discriminator_dims": [params.data_dim, params.discriminator_hidden_dim, 1],
            "adversarial_flow": True
        },
        color_mappings={
            "generator": "#27ae60",
            "discriminator": "#e74c3c",
            "real_data": "#3498db",
            "fake_data": "#f39c12",
            "minimax": "#9b59b6"
        },
        interactive_elements=["training_dynamics", "loss_visualization", "mode_collapse_demo"]
    )
    
    return NumericalExample(
        example_id="gan_adversarial_training",
        description="Complete GAN adversarial training with minimax game theory",
        input_values={
            "noise": np.random.randn(params.batch_size, params.noise_dim),
            "real_data": np.random.randn(params.batch_size, params.data_dim)
        },
        computation_steps=computation_steps,
        output_values={"nash_equilibrium_reached": np.array([False])},
        visualization_data=viz_data,
        educational_notes=[
            "The minimax game creates adversarial dynamics between generator and discriminator",
            "Nash equilibrium occurs when D(x) = 0.5 for all x in the data distribution",
            "Mode collapse happens when generator produces limited diversity",
            "Training instability arises from the non-convex minimax optimization"
        ]
    )