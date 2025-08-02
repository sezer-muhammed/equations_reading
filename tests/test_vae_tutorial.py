"""
Tests for VAE tutorial implementation.
Validates mathematical correctness and educational content quality.
"""

import pytest
import numpy as np
from src.computation.generative_ops.vae_operations import VAEOperations, VAEParameters, create_vae_numerical_example
from src.content.advanced.vae_tutorial import create_vae_tutorial
from src.visualization.operation_viz.vae_visualizer import create_vae_visualization_suite


class TestVAEOperations:
    """Test VAE mathematical operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = VAEParameters(input_dim=784, latent_dim=20, hidden_dim=400, batch_size=32)
        self.vae_ops = VAEOperations(self.params)
    
    def test_elbo_computation(self):
        """Test ELBO computation with known values."""
        # Create test data
        x = np.random.randn(4, self.params.input_dim)
        mu = np.random.randn(4, self.params.latent_dim)
        log_var = np.random.randn(4, self.params.latent_dim)
        x_recon = np.random.randn(4, self.params.input_dim)
        
        # Compute ELBO components
        elbo_components = self.vae_ops.compute_elbo_components(x, mu, log_var, x_recon)
        
        # Validate structure
        assert 'reconstruction_loss' in elbo_components
        assert 'kl_divergence' in elbo_components
        assert 'elbo' in elbo_components
        assert 'negative_elbo' in elbo_components
        
        # Validate mathematical properties
        assert elbo_components['reconstruction_loss'] >= 0
        assert elbo_components['kl_divergence'] >= 0  # KL divergence is non-negative
        assert elbo_components['elbo'] == -(elbo_components['reconstruction_loss'] + elbo_components['kl_divergence'])
        assert elbo_components['negative_elbo'] == -elbo_components['elbo']
    
    def test_reparameterization_trick(self):
        """Test reparameterization trick implementation."""
        mu = np.array([1.0, 2.0, -0.5])
        log_var = np.array([0.0, 0.693, -1.0])  # log(1), log(2), log(0.368)
        epsilon = np.array([0.0, 1.0, -1.0])
        
        z, returned_epsilon, sigma = self.vae_ops.reparameterization_trick(mu, log_var, epsilon)
        
        # Validate shapes
        assert z.shape == mu.shape
        assert sigma.shape == mu.shape
        assert np.array_equal(returned_epsilon, epsilon)
        
        # Validate mathematical correctness
        expected_sigma = np.exp(0.5 * log_var)
        np.testing.assert_array_almost_equal(sigma, expected_sigma)
        
        expected_z = mu + sigma * epsilon
        np.testing.assert_array_almost_equal(z, expected_z)
    
    def test_encoder_decoder_consistency(self):
        """Test encoder-decoder forward pass consistency."""
        x = np.random.randn(8, self.params.input_dim)
        
        # Forward pass
        mu, log_var = self.vae_ops.encoder_forward(x)
        z, _, _ = self.vae_ops.reparameterization_trick(mu, log_var)
        x_recon = self.vae_ops.decoder_forward(z)
        
        # Validate shapes
        assert mu.shape == (8, self.params.latent_dim)
        assert log_var.shape == (8, self.params.latent_dim)
        assert z.shape == (8, self.params.latent_dim)
        assert x_recon.shape == (8, self.params.input_dim)
    
    def test_latent_interpolation(self):
        """Test latent space interpolation."""
        z1 = np.random.randn(self.params.latent_dim)
        z2 = np.random.randn(self.params.latent_dim)
        
        interpolations = self.vae_ops.generate_latent_interpolation(z1, z2, num_steps=5)
        
        # Validate structure
        assert len(interpolations) == 5
        assert all(interp.shape == (self.params.input_dim,) for interp in interpolations)
        
        # Validate that interpolation is monotonic (differences are consistent)
        diffs = []
        for i in range(len(interpolations) - 1):
            diff = np.linalg.norm(interpolations[i+1] - interpolations[i])
            diffs.append(diff)
        
        # Check that differences are reasonably consistent (smooth interpolation)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        # Standard deviation should be small relative to mean (smooth progression)
        assert std_diff / mean_diff < 0.5  # Coefficient of variation < 0.5
    
    def test_derivation_steps_generation(self):
        """Test ELBO derivation steps generation."""
        steps = self.vae_ops.generate_elbo_derivation_steps()
        
        # Validate structure
        assert len(steps) == 4
        assert all(hasattr(step, 'step_number') for step in steps)
        assert all(hasattr(step, 'operation_name') for step in steps)
        assert all(hasattr(step, 'input_values') for step in steps)
        assert all(hasattr(step, 'output_values') for step in steps)
        
        # Validate step sequence
        step_names = [step.operation_name for step in steps]
        expected_names = ['encoder_output', 'reparameterization', 'decoder_reconstruction', 'elbo_computation']
        assert step_names == expected_names


class TestVAETutorialContent:
    """Test VAE tutorial content structure and quality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.vae_concept = create_vae_tutorial()
    
    def test_concept_structure(self):
        """Test VAE concept structure."""
        assert self.vae_concept.concept_id == "vae_tutorial"
        assert self.vae_concept.title == "Variational Autoencoders (VAE)"
        assert self.vae_concept.difficulty_level == 4
        
        # Validate prerequisites
        expected_prereqs = ["probability_theory", "neural_networks", "variational_inference", "information_theory"]
        assert all(prereq in self.vae_concept.prerequisites for prereq in expected_prereqs)
    
    def test_equations_content(self):
        """Test VAE equations content."""
        equations = self.vae_concept.equations
        assert len(equations) >= 2
        
        # Find ELBO equation
        elbo_eq = next((eq for eq in equations if eq.equation_id == "vae_elbo"), None)
        assert elbo_eq is not None
        assert "ELBO" in elbo_eq.latex_expression or "mathcal{L}" in elbo_eq.latex_expression
        assert len(elbo_eq.derivation_steps) >= 3
        
        # Find reparameterization equation
        reparam_eq = next((eq for eq in equations if eq.equation_id == "vae_reparameterization"), None)
        assert reparam_eq is not None
        assert "epsilon" in reparam_eq.variables or "ε" in reparam_eq.latex_expression
    
    def test_explanations_quality(self):
        """Test quality and completeness of explanations."""
        explanations = self.vae_concept.explanations
        assert len(explanations) >= 3
        
        # Check for different explanation types
        explanation_types = [exp.explanation_type for exp in explanations]
        assert "intuitive" in explanation_types
        assert "formal" in explanation_types
        assert "practical" in explanation_types
        
        # Validate content length and quality
        for explanation in explanations:
            assert len(explanation.content) > 100  # Substantial content
            assert explanation.mathematical_level in range(1, 6)  # Valid difficulty level
    
    def test_visualizations_structure(self):
        """Test VAE visualizations structure."""
        visualizations = self.vae_concept.visualizations
        assert len(visualizations) >= 2
        
        # Check for architecture visualization
        arch_viz = next((viz for viz in visualizations if "architecture" in viz.visualization_id), None)
        assert arch_viz is not None
        assert arch_viz.interactive is True
        
        # Check for latent space visualization
        latent_viz = next((viz for viz in visualizations if "latent" in viz.visualization_id), None)
        assert latent_viz is not None
    
    def test_learning_objectives(self):
        """Test learning objectives completeness."""
        objectives = self.vae_concept.learning_objectives
        assert len(objectives) >= 4
        
        # Check for key learning objectives
        objectives_text = " ".join(objectives).lower()
        assert "elbo" in objectives_text
        assert "reparameterization" in objectives_text
        assert "encoder" in objectives_text or "decoder" in objectives_text
        assert "latent" in objectives_text


class TestVAEVisualization:
    """Test VAE visualization components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.viz_suite = create_vae_visualization_suite()
    
    def test_visualization_suite_structure(self):
        """Test visualization suite structure."""
        assert 'architecture_visualizer' in self.viz_suite
        assert 'latent_space_visualizer' in self.viz_suite
        assert 'interactive_components' in self.viz_suite
        assert 'config' in self.viz_suite
    
    def test_architecture_visualization(self):
        """Test architecture visualization creation."""
        arch_viz = self.viz_suite['architecture_visualizer']
        layers = arch_viz.create_architecture_diagram(784, 20, 400)
        
        # Validate layer structure
        expected_layers = ['input', 'encoder_hidden', 'mu', 'log_var', 'z_sample', 'decoder_hidden', 'reconstruction']
        assert all(layer in layers for layer in expected_layers)
        
        # Validate layer properties
        for layer_name, layer_info in layers.items():
            assert 'pos' in layer_info
            assert 'size' in layer_info
            assert 'color' in layer_info
            assert isinstance(layer_info['size'], int)
    
    def test_reparameterization_visualization(self):
        """Test reparameterization trick visualization."""
        arch_viz = self.viz_suite['architecture_visualizer']
        
        mu = np.array([1.0, 0.5, -0.2])
        log_var = np.array([0.0, -1.0, 0.5])
        epsilon = np.array([0.5, -1.0, 1.5])
        
        viz = arch_viz.create_reparameterization_visualization(mu, log_var, epsilon)
        
        # Validate visualization structure
        assert viz.operation_type == "reparameterization_trick"
        assert len(viz.input_matrices) == 3  # mu, sigma, epsilon
        assert len(viz.animation_sequence) == 4  # 4 animation frames
        
        # Validate animation frames
        for i, frame in enumerate(viz.animation_sequence):
            assert frame.frame_number == i + 1
            assert len(frame.description) > 0
    
    def test_latent_space_visualization(self):
        """Test latent space visualization."""
        latent_viz = self.viz_suite['latent_space_visualizer']
        
        # Test with 2D latent samples
        latent_samples = np.random.randn(100, 2)
        plot_data = latent_viz.create_latent_space_plot(latent_samples)
        
        # Validate plot data structure
        assert 'latent_points' in plot_data
        assert 'prior_grid_x' in plot_data
        assert 'prior_grid_y' in plot_data
        assert 'prior_density' in plot_data
        
        # Validate data shapes
        assert plot_data['latent_points'].shape == (100, 2)
        assert plot_data['prior_grid_x'].shape == plot_data['prior_grid_y'].shape
    
    def test_interpolation_path(self):
        """Test latent interpolation path creation."""
        latent_viz = self.viz_suite['latent_space_visualizer']
        
        z1 = np.array([1.0, 2.0])
        z2 = np.array([-1.0, 0.5])
        
        path = latent_viz.create_interpolation_path(z1, z2, num_steps=5)
        
        # Validate interpolation path
        assert path.shape == (5, 2)
        np.testing.assert_array_almost_equal(path[0], z1)
        np.testing.assert_array_almost_equal(path[-1], z2)
        
        # Check intermediate points are between endpoints
        for i in range(1, 4):
            assert np.all(np.minimum(z1, z2) <= path[i])
            assert np.all(path[i] <= np.maximum(z1, z2))


class TestVAENumericalExample:
    """Test VAE numerical example generation."""
    
    def test_numerical_example_creation(self):
        """Test VAE numerical example creation."""
        example = create_vae_numerical_example()
        
        # Validate example structure
        assert example.example_id == "vae_elbo_derivation"
        assert len(example.computation_steps) == 4
        assert example.visualization_data is not None
        
        # Validate computation steps
        step_names = [step.operation_name for step in example.computation_steps]
        expected_names = ['encoder_output', 'reparameterization', 'decoder_reconstruction', 'elbo_computation']
        assert step_names == expected_names
        
        # Validate educational notes
        assert len(example.educational_notes) >= 3
        notes_text = " ".join(example.educational_notes).lower()
        assert "reparameterization" in notes_text
        assert "kl divergence" in notes_text or "kl" in notes_text
        assert "elbo" in notes_text


if __name__ == "__main__":
    pytest.main([__file__])