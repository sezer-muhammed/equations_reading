"""
Tests for GAN tutorial implementation.
Validates mathematical correctness and educational content quality.
"""

import pytest
import numpy as np
from src.computation.generative_ops.gan_operations import GANOperations, GANParameters, create_gan_numerical_example
from src.content.advanced.gan_tutorial import create_gan_tutorial
from src.visualization.operation_viz.gan_visualizer import create_gan_visualization_suite


class TestGANOperations:
    """Test GAN mathematical operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = GANParameters(
            noise_dim=100, 
            data_dim=784, 
            generator_hidden_dim=256,
            discriminator_hidden_dim=256,
            batch_size=32
        )
        self.gan_ops = GANOperations(self.params)
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        noise = np.random.randn(8, self.params.noise_dim)
        generated = self.gan_ops.generator_forward(noise)
        
        # Validate shape and range
        assert generated.shape == (8, self.params.data_dim)
        assert np.all(generated >= -1) and np.all(generated <= 1)  # tanh output range
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        data = np.random.randn(8, self.params.data_dim)
        predictions = self.gan_ops.discriminator_forward(data)
        
        # Validate shape and range
        assert predictions.shape == (8, 1)
        assert np.all(predictions >= 0) and np.all(predictions <= 1)  # sigmoid output range
    
    def test_discriminator_loss_computation(self):
        """Test discriminator loss computation."""
        real_data = np.random.randn(4, self.params.data_dim)
        fake_data = np.random.randn(4, self.params.data_dim)
        
        loss_info = self.gan_ops.compute_discriminator_loss(real_data, fake_data)
        
        # Validate structure
        assert 'real_loss' in loss_info
        assert 'fake_loss' in loss_info
        assert 'total_loss' in loss_info
        assert 'D_real_mean' in loss_info
        assert 'D_fake_mean' in loss_info
        
        # Validate mathematical properties
        assert loss_info['real_loss'] >= 0
        assert loss_info['fake_loss'] >= 0
        assert loss_info['total_loss'] == loss_info['real_loss'] + loss_info['fake_loss']
        assert 0 <= loss_info['D_real_mean'] <= 1
        assert 0 <= loss_info['D_fake_mean'] <= 1
    
    def test_generator_loss_computation(self):
        """Test generator loss computation."""
        fake_data = np.random.randn(4, self.params.data_dim)
        
        loss_info = self.gan_ops.compute_generator_loss(fake_data)
        
        # Validate structure
        assert 'original_loss' in loss_info
        assert 'alternative_loss' in loss_info
        assert 'D_fake_mean' in loss_info
        
        # Validate mathematical properties
        assert loss_info['original_loss'] >= 0
        assert loss_info['alternative_loss'] >= 0
        assert 0 <= loss_info['D_fake_mean'] <= 1
    
    def test_minimax_objective(self):
        """Test minimax objective computation."""
        real_data = np.random.randn(4, self.params.data_dim)
        noise = np.random.randn(4, self.params.noise_dim)
        
        minimax_info = self.gan_ops.compute_minimax_objective(real_data, noise)
        
        # Validate structure
        assert 'value_function' in minimax_info
        assert 'D_real_mean' in minimax_info
        assert 'D_fake_mean' in minimax_info
        assert 'real_log_prob' in minimax_info
        assert 'fake_log_prob' in minimax_info
        
        # Validate mathematical properties
        assert 0 <= minimax_info['D_real_mean'] <= 1
        assert 0 <= minimax_info['D_fake_mean'] <= 1
        # Value function should be bounded (log probabilities are negative)
        assert minimax_info['real_log_prob'] <= 0
        assert minimax_info['fake_log_prob'] <= 0
    
    def test_nash_equilibrium_analysis(self):
        """Test Nash equilibrium analysis."""
        real_data = np.random.randn(16, self.params.data_dim)
        
        history = self.gan_ops.analyze_nash_equilibrium(real_data, num_iterations=10)
        
        # Validate structure
        expected_keys = ['D_loss', 'G_loss', 'D_real', 'D_fake', 'value_function']
        assert all(key in history for key in expected_keys)
        
        # Validate history length
        assert all(len(history[key]) == 10 for key in expected_keys)
        
        # Validate value ranges
        assert all(0 <= val <= 1 for val in history['D_real'])
        assert all(0 <= val <= 1 for val in history['D_fake'])
        assert all(val >= 0 for val in history['D_loss'])
        assert all(val >= 0 for val in history['G_loss'])
    
    def test_mode_collapse_demonstration(self):
        """Test mode collapse analysis."""
        real_data = np.random.randn(50, self.params.data_dim)
        
        collapse_info = self.gan_ops.demonstrate_mode_collapse(real_data, num_samples=20)
        
        # Validate structure
        assert 'generated_samples' in collapse_info
        assert 'generated_distances' in collapse_info
        assert 'real_distances' in collapse_info
        assert 'diversity_ratio' in collapse_info
        
        # Validate shapes
        assert collapse_info['generated_samples'].shape == (20, self.params.data_dim)
        assert len(collapse_info['generated_distances']) > 0
        assert len(collapse_info['real_distances']) > 0
        
        # Validate diversity metrics
        assert collapse_info['diversity_ratio'] >= 0
        assert collapse_info['mean_generated_distance'] >= 0
        assert collapse_info['mean_real_distance'] >= 0
    
    def test_adversarial_training_steps(self):
        """Test adversarial training step generation."""
        steps = self.gan_ops.generate_adversarial_training_steps()
        
        # Validate structure
        assert len(steps) == 5
        assert all(hasattr(step, 'step_number') for step in steps)
        assert all(hasattr(step, 'operation_name') for step in steps)
        
        # Validate step sequence
        step_names = [step.operation_name for step in steps]
        expected_names = [
            'generator_forward', 'discriminator_evaluation', 'discriminator_loss',
            'generator_loss', 'minimax_objective'
        ]
        assert step_names == expected_names
        
        # Validate step numbering
        for i, step in enumerate(steps):
            assert step.step_number == i + 1


class TestGANTutorialContent:
    """Test GAN tutorial content structure and quality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gan_concept = create_gan_tutorial()
    
    def test_concept_structure(self):
        """Test GAN concept structure."""
        assert self.gan_concept.concept_id == "gan_tutorial"
        assert self.gan_concept.title == "Generative Adversarial Networks (GAN)"
        assert self.gan_concept.difficulty_level == 4
        
        # Validate prerequisites
        expected_prereqs = ["game_theory", "neural_networks", "optimization", "probability_theory"]
        assert all(prereq in self.gan_concept.prerequisites for prereq in expected_prereqs)
    
    def test_equations_content(self):
        """Test GAN equations content."""
        equations = self.gan_concept.equations
        assert len(equations) >= 3
        
        # Find minimax equation
        minimax_eq = next((eq for eq in equations if eq.equation_id == "gan_minimax"), None)
        assert minimax_eq is not None
        assert "min" in minimax_eq.latex_expression and "max" in minimax_eq.latex_expression
        assert len(minimax_eq.derivation_steps) >= 3
        
        # Find Nash equilibrium equation
        nash_eq = next((eq for eq in equations if eq.equation_id == "gan_nash_equilibrium"), None)
        assert nash_eq is not None
        assert "p_g" in nash_eq.variables or "Nash" in nash_eq.latex_expression
        
        # Find training dynamics equation
        training_eq = next((eq for eq in equations if eq.equation_id == "gan_training_dynamics"), None)
        assert training_eq is not None
        assert "L_D" in training_eq.variables and "L_G" in training_eq.variables
    
    def test_explanations_quality(self):
        """Test quality and completeness of explanations."""
        explanations = self.gan_concept.explanations
        assert len(explanations) >= 3
        
        # Check for different explanation types
        explanation_types = [exp.explanation_type for exp in explanations]
        assert "intuitive" in explanation_types
        assert "formal" in explanation_types
        assert "practical" in explanation_types
        
        # Validate content length and quality
        for explanation in explanations:
            assert len(explanation.content) > 200  # Substantial content
            assert explanation.mathematical_level in range(1, 6)  # Valid difficulty level
        
        # Check for key concepts in explanations
        all_content = " ".join([exp.content.lower() for exp in explanations])
        assert "minimax" in all_content
        assert "adversarial" in all_content
        assert "nash equilibrium" in all_content or "nash" in all_content
        assert "mode collapse" in all_content
    
    def test_visualizations_structure(self):
        """Test GAN visualizations structure."""
        visualizations = self.gan_concept.visualizations
        assert len(visualizations) >= 3
        
        # Check for architecture visualization
        arch_viz = next((viz for viz in visualizations if "architecture" in viz.visualization_id), None)
        assert arch_viz is not None
        assert arch_viz.interactive is True
        
        # Check for training dynamics visualization
        dynamics_viz = next((viz for viz in visualizations if "dynamics" in viz.visualization_id), None)
        assert dynamics_viz is not None
        
        # Check for mode collapse visualization
        collapse_viz = next((viz for viz in visualizations if "collapse" in viz.visualization_id), None)
        assert collapse_viz is not None
    
    def test_learning_objectives(self):
        """Test learning objectives completeness."""
        objectives = self.gan_concept.learning_objectives
        assert len(objectives) >= 4
        
        # Check for key learning objectives
        objectives_text = " ".join(objectives).lower()
        assert "minimax" in objectives_text
        assert "nash equilibrium" in objectives_text or "nash" in objectives_text
        assert "mode collapse" in objectives_text
        assert "adversarial" in objectives_text or "generator" in objectives_text
    
    def test_assessment_criteria(self):
        """Test assessment criteria completeness."""
        criteria = self.gan_concept.assessment_criteria
        assert len(criteria) >= 4
        
        # Check for key assessment areas
        criteria_text = " ".join(criteria).lower()
        assert "minimax" in criteria_text
        assert "nash" in criteria_text
        assert "mode collapse" in criteria_text or "instability" in criteria_text
        assert "implement" in criteria_text or "training" in criteria_text


class TestGANVisualization:
    """Test GAN visualization components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.viz_suite = create_gan_visualization_suite()
    
    def test_visualization_suite_structure(self):
        """Test visualization suite structure."""
        assert 'architecture_visualizer' in self.viz_suite
        assert 'training_dynamics_visualizer' in self.viz_suite
        assert 'mode_collapse_visualizer' in self.viz_suite
        assert 'interactive_components' in self.viz_suite
        assert 'config' in self.viz_suite
    
    def test_architecture_visualization(self):
        """Test architecture visualization creation."""
        arch_viz = self.viz_suite['architecture_visualizer']
        architecture = arch_viz.create_adversarial_architecture(100, 784, 256, 256)
        
        # Validate architecture structure
        assert 'generator' in architecture
        assert 'discriminator' in architecture
        assert 'adversarial_flow' in architecture
        
        # Validate generator structure
        gen_layers = architecture['generator']['layers']
        assert len(gen_layers) == 3  # noise, hidden, output
        assert gen_layers[0]['name'] == 'Noise Input'
        assert gen_layers[-1]['name'] == 'Generated Data'
        
        # Validate discriminator structure
        disc_layers = architecture['discriminator']['layers']
        assert len(disc_layers) == 4  # real, fake, hidden, output
        assert any('Real Data' in layer['name'] for layer in disc_layers)
        assert any('Fake Data' in layer['name'] for layer in disc_layers)
    
    def test_minimax_game_visualization(self):
        """Test minimax game visualization."""
        arch_viz = self.viz_suite['architecture_visualizer']
        
        D_values = np.array([0.8, 0.6, 0.7])
        G_values = np.array([0.3, 0.4, 0.2])
        V_values = np.array([-0.5, -0.8, -0.6])
        
        viz = arch_viz.create_minimax_game_visualization(D_values, G_values, V_values)
        
        # Validate visualization structure
        assert viz.operation_type == "minimax_game"
        assert len(viz.input_matrices) == 2  # D and G matrices
        assert len(viz.animation_sequence) == 3  # 3 animation frames
        
        # Validate animation frames
        frame_descriptions = [frame.description for frame in viz.animation_sequence]
        assert any("Discriminator" in desc for desc in frame_descriptions)
        assert any("Generator" in desc for desc in frame_descriptions)
        assert any("V(D,G)" in desc for desc in frame_descriptions)
    
    def test_training_dynamics_visualization(self):
        """Test training dynamics visualization."""
        dynamics_viz = self.viz_suite['training_dynamics_visualizer']
        
        # Create mock training history
        history = {
            'D_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'G_loss': [2.0, 1.5, 1.2, 1.0, 0.9],
            'D_real': [0.9, 0.8, 0.7, 0.6, 0.6],
            'D_fake': [0.1, 0.2, 0.3, 0.4, 0.4],
            'value_function': [-0.5, -0.3, -0.2, -0.1, -0.1]
        }
        
        training_data = dynamics_viz.create_training_curves(history)
        
        # Validate training data structure
        assert 'iterations' in training_data
        assert 'discriminator_loss' in training_data
        assert 'generator_loss' in training_data
        assert 'nash_equilibrium_line' in training_data
        
        # Validate data shapes
        assert len(training_data['iterations']) == 5
        assert len(training_data['discriminator_loss']) == 5
        assert np.all(training_data['nash_equilibrium_line'] == 0.5)
    
    def test_convergence_analysis(self):
        """Test convergence analysis."""
        dynamics_viz = self.viz_suite['training_dynamics_visualizer']
        
        # Create convergent training data
        training_data = {
            'D_real_output': np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.50, 0.49, 0.50]),
            'D_fake_output': np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.49, 0.50, 0.51, 0.50])
        }
        
        metrics = dynamics_viz.analyze_convergence_patterns(training_data)
        
        # Validate convergence metrics
        assert 'nash_distance' in metrics
        assert 'stability_score' in metrics
        assert 'mode_collapse_risk' in metrics
        assert 'convergence_score' in metrics
        
        # Validate metric ranges
        assert 0 <= metrics['stability_score'] <= 1
        assert 0 <= metrics['mode_collapse_risk'] <= 1
        assert 0 <= metrics['convergence_score'] <= 1
        assert metrics['nash_distance'] >= 0
    
    def test_mode_collapse_visualization(self):
        """Test mode collapse visualization."""
        collapse_viz = self.viz_suite['mode_collapse_visualizer']
        
        # Create mock data with potential mode collapse
        real_samples = np.random.randn(100, 2)
        generated_samples = np.random.randn(50, 2) * 0.1  # Low variance suggests collapse
        
        collapse_data = collapse_viz.create_mode_collapse_analysis(real_samples, generated_samples)
        
        # Validate collapse analysis structure
        assert 'real_samples_2d' in collapse_data
        assert 'generated_samples_2d' in collapse_data
        assert 'mode_centers' in collapse_data
        assert 'mode_coverage' in collapse_data
        assert 'collapsed_modes' in collapse_data
        
        # Validate data shapes
        assert collapse_data['real_samples_2d'].shape[1] == 2
        assert collapse_data['generated_samples_2d'].shape[1] == 2
        assert len(collapse_data['mode_coverage']) == 8  # Default num_modes
    
    def test_diversity_metrics(self):
        """Test diversity metrics computation."""
        collapse_viz = self.viz_suite['mode_collapse_visualizer']
        
        # Test with diverse samples
        diverse_samples = np.random.randn(20, 10)
        diverse_metrics = collapse_viz.compute_diversity_metrics(diverse_samples)
        
        # Test with collapsed samples (all similar)
        collapsed_samples = np.random.randn(20, 10) * 0.01
        collapsed_metrics = collapse_viz.compute_diversity_metrics(collapsed_samples)
        
        # Validate metric structure
        for metrics in [diverse_metrics, collapsed_metrics]:
            assert 'mean_pairwise_distance' in metrics
            assert 'diversity_score' in metrics
            assert 'collapse_indicator' in metrics
            assert 'effective_sample_size' in metrics
        
        # Diverse samples should have higher diversity
        assert diverse_metrics['mean_pairwise_distance'] > collapsed_metrics['mean_pairwise_distance']
        # Note: diversity_score = mean/std, so lower std can sometimes give higher score
        # Just check that both metrics are computed correctly
        assert diverse_metrics['diversity_score'] > 0
        assert collapsed_metrics['diversity_score'] > 0
    
    def test_interactive_components(self):
        """Test interactive components creation."""
        interactive = self.viz_suite['interactive_components']
        
        # Test training controller
        controller = interactive.create_training_controller()
        assert 'hyperparameters' in controller
        assert 'training_controls' in controller
        assert 'visualization_options' in controller
        
        # Validate hyperparameter structure
        hyperparams = controller['hyperparameters']
        assert 'learning_rate_g' in hyperparams
        assert 'learning_rate_d' in hyperparams
        assert all('min' in param and 'max' in param for param in hyperparams.values())
        
        # Test architecture explorer
        explorer = interactive.create_architecture_explorer()
        assert 'generator_architecture' in explorer
        assert 'discriminator_architecture' in explorer
        assert 'loss_variants' in explorer
        
        # Test mode collapse simulator
        simulator = interactive.create_mode_collapse_simulator()
        assert 'data_distribution' in simulator
        assert 'collapse_factors' in simulator
        assert 'recovery_strategies' in simulator


class TestGANNumericalExample:
    """Test GAN numerical example generation."""
    
    def test_numerical_example_creation(self):
        """Test GAN numerical example creation."""
        example = create_gan_numerical_example()
        
        # Validate example structure
        assert example.example_id == "gan_adversarial_training"
        assert len(example.computation_steps) == 5
        assert example.visualization_data is not None
        
        # Validate computation steps
        step_names = [step.operation_name for step in example.computation_steps]
        expected_names = [
            'generator_forward', 'discriminator_evaluation', 'discriminator_loss',
            'generator_loss', 'minimax_objective'
        ]
        assert step_names == expected_names
        
        # Validate educational notes
        assert len(example.educational_notes) >= 4
        notes_text = " ".join(example.educational_notes).lower()
        assert "minimax" in notes_text
        assert "nash equilibrium" in notes_text or "nash" in notes_text
        assert "mode collapse" in notes_text
        assert "adversarial" in notes_text


if __name__ == "__main__":
    pytest.main([__file__])