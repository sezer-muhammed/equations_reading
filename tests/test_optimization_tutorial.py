"""
Tests for the optimization algorithms tutorial implementation.
Validates gradient descent, Adam optimizer, learning rate scheduling, and loss landscapes.
"""

import pytest
import numpy as np
from src.content.foundations.optimization_tutorial import OptimizationTutorial, OptimizationTutorialResult
from src.visualization.graph_viz.optimization_landscapes import OptimizationLandscapeVisualizer
from src.visualization.interactive.optimization_widgets import OptimizationInteractiveComponents


class TestOptimizationTutorial:
    """Test suite for optimization algorithms tutorial."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tutorial = OptimizationTutorial()
    
    def test_tutorial_creation(self):
        """Test that the complete tutorial can be created."""
        result = self.tutorial.create_complete_tutorial()
        
        assert isinstance(result, OptimizationTutorialResult)
        assert result.concept is not None
        assert len(result.gradient_descent_examples) > 0
        assert len(result.adam_examples) > 0
        assert len(result.learning_rate_examples) > 0
        assert len(result.loss_landscape_examples) > 0
        assert len(result.tutorial_sections) > 0
    
    def test_optimization_concept_structure(self):
        """Test the mathematical concept structure."""
        result = self.tutorial.create_complete_tutorial()
        concept = result.concept
        
        assert concept.concept_id == "optimization_algorithms"
        assert concept.title == "Optimization Algorithms for Deep Learning"
        assert concept.difficulty_level == 3
        assert "calculus" in concept.prerequisites
        assert "linear_algebra" in concept.prerequisites
        assert len(concept.equations) >= 2  # At least GD and Adam
        assert len(concept.explanations) >= 2  # Intuitive and formal
    
    def test_gradient_descent_examples(self):
        """Test gradient descent examples generation."""
        result = self.tutorial.create_complete_tutorial()
        gd_examples = result.gradient_descent_examples
        
        assert len(gd_examples) >= 3  # At least quadratic, Rosenbrock, and LR comparison
        
        # Check first example (quadratic)
        first_example = gd_examples[0]
        assert first_example.final_parameters is not None
        assert len(first_example.optimization_steps) > 0
        assert len(first_example.computation_steps) > 0
        assert 'converged' in first_example.convergence_info
        assert 'final_loss' in first_example.convergence_info
        assert 'total_steps' in first_example.convergence_info
    
    def test_adam_examples(self):
        """Test Adam optimizer examples generation."""
        result = self.tutorial.create_complete_tutorial()
        adam_examples = result.adam_examples
        
        assert len(adam_examples) >= 3  # At least quadratic, Rosenbrock, and hyperparameter comparison
        
        # Check first example
        first_example = adam_examples[0]
        assert first_example.final_parameters is not None
        assert len(first_example.optimization_steps) > 0
        assert len(first_example.computation_steps) > 0
        assert 'adam_hyperparameters' in first_example.convergence_info
        
        # Check Adam-specific data in optimization steps
        first_step = first_example.optimization_steps[0]
        assert 'm' in first_step.additional_info
        assert 'v' in first_step.additional_info
        assert 'm_hat' in first_step.additional_info
        assert 'v_hat' in first_step.additional_info
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling examples."""
        result = self.tutorial.create_complete_tutorial()
        lr_examples = result.learning_rate_examples
        
        assert len(lr_examples) >= 3  # Constant, exponential decay, step decay
        
        schedule_types = [ex['schedule_type'] for ex in lr_examples]
        assert 'constant' in schedule_types
        assert 'exponential_decay' in schedule_types
        assert 'step_decay' in schedule_types
        
        # Check structure of each example
        for example in lr_examples:
            assert 'schedule_type' in example
            assert 'results' in example
            assert 'description' in example
            assert len(example['results']) > 0
            
            # Check result structure
            first_result = example['results'][0]
            assert 'step' in first_result
            assert 'lr' in first_result
            assert 'loss' in first_result
    
    def test_loss_landscapes(self):
        """Test loss landscape examples generation."""
        result = self.tutorial.create_complete_tutorial()
        landscape_examples = result.loss_landscape_examples
        
        assert len(landscape_examples) >= 3  # Quadratic, Rosenbrock, multimodal
        
        landscape_types = [ex['landscape_type'] for ex in landscape_examples]
        assert 'quadratic' in landscape_types
        assert 'rosenbrock' in landscape_types
        assert 'multimodal' in landscape_types
        
        # Check structure of each landscape
        for example in landscape_examples:
            assert 'landscape_type' in example
            assert 'X' in example
            assert 'Y' in example
            assert 'Z' in example
            assert 'description' in example
            assert 'properties' in example
            
            # Check data shapes
            assert example['X'].shape == example['Y'].shape == example['Z'].shape
    
    def test_tutorial_sections_structure(self):
        """Test the structure of tutorial sections."""
        result = self.tutorial.create_complete_tutorial()
        sections = result.tutorial_sections
        
        assert len(sections) >= 6  # Introduction, GD, Adam, LR, Landscapes, Guidelines
        
        # Check required sections exist
        section_titles = [section['title'] for section in sections]
        assert any('Introduction' in title for title in section_titles)
        assert any('Gradient Descent' in title for title in section_titles)
        assert any('Adam' in title for title in section_titles)
        assert any('Learning Rate' in title for title in section_titles)
        assert any('Loss Landscape' in title for title in section_titles)
        
        # Check section structure
        for section in sections:
            assert 'title' in section
            assert 'content' in section
    
    def test_mathematical_equations(self):
        """Test the mathematical equations in the concept."""
        result = self.tutorial.create_complete_tutorial()
        equations = result.concept.equations
        
        # Check gradient descent equation
        gd_eq = next((eq for eq in equations if eq.equation_id == "gradient_descent"), None)
        assert gd_eq is not None
        assert r"\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)" in gd_eq.latex_expression
        assert 'θ' in gd_eq.variables
        assert 'α' in gd_eq.variables
        assert len(gd_eq.derivation_steps) > 0
        
        # Check Adam equation
        adam_eq = next((eq for eq in equations if eq.equation_id == "adam_optimizer"), None)
        assert adam_eq is not None
        assert r"\hat{v}_t" in adam_eq.latex_expression
        assert r"\hat{m}_t" in adam_eq.latex_expression
        assert 'm_t' in adam_eq.variables
        assert 'v_t' in adam_eq.variables
        assert len(adam_eq.derivation_steps) >= 4  # At least 4 steps for Adam
    
    def test_computation_steps_detail(self):
        """Test the detailed computation steps in examples."""
        result = self.tutorial.create_complete_tutorial()
        
        # Test gradient descent computation steps
        gd_example = result.gradient_descent_examples[0]
        gd_steps = gd_example.computation_steps
        
        assert len(gd_steps) > 0
        
        # Check step structure
        first_step = gd_steps[0]
        assert hasattr(first_step, 'step_number')
        assert hasattr(first_step, 'operation_name')
        assert hasattr(first_step, 'operation_description')
        assert hasattr(first_step, 'input_values')
        assert hasattr(first_step, 'output_values')
        
        # Test Adam computation steps (should have more detailed breakdown)
        adam_example = result.adam_examples[0]
        adam_steps = adam_example.computation_steps
        
        assert len(adam_steps) > len(gd_steps)  # Adam has more steps per iteration
        
        # Check for Adam-specific operations
        operation_names = [step.operation_name for step in adam_steps]
        assert 'update_first_moment' in operation_names
        assert 'update_second_moment' in operation_names
        assert 'bias_correct_first_moment' in operation_names
        assert 'bias_correct_second_moment' in operation_names


class TestOptimizationVisualization:
    """Test suite for optimization visualization components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = OptimizationLandscapeVisualizer()
        self.tutorial = OptimizationTutorial()
    
    def test_loss_landscape_visualization(self):
        """Test loss landscape visualization creation."""
        # Get tutorial data
        result = self.tutorial.create_complete_tutorial()
        landscape_data = result.loss_landscape_examples[0]  # Quadratic
        optimizer_results = result.gradient_descent_examples[:2]
        
        # Create visualization
        viz = self.visualizer.create_loss_landscape(
            landscape_data, optimizer_results, ['GD-1', 'GD-2']
        )
        
        assert viz.X.shape == landscape_data['X'].shape
        assert viz.Y.shape == landscape_data['Y'].shape
        assert viz.Z.shape == landscape_data['Z'].shape
        assert len(viz.optimizer_paths) == 2
        assert len(viz.optimizer_names) == 2
        assert len(viz.contour_levels) > 0
    
    def test_convergence_plot_creation(self):
        """Test convergence plot data creation."""
        result = self.tutorial.create_complete_tutorial()
        optimizer_results = result.gradient_descent_examples[:2]
        
        convergence_data = self.visualizer.create_convergence_plot(
            optimizer_results, ['GD-1', 'GD-2']
        )
        
        assert len(convergence_data['optimizers']) == 2
        assert len(convergence_data['iterations']) == 2
        assert len(convergence_data['losses']) == 2
        assert len(convergence_data['gradient_norms']) == 2
        
        # Check data structure
        assert len(convergence_data['iterations'][0]) > 0
        assert len(convergence_data['losses'][0]) > 0
        assert len(convergence_data['gradient_norms'][0]) > 0
    
    def test_adam_components_visualization(self):
        """Test Adam components visualization."""
        result = self.tutorial.create_complete_tutorial()
        adam_result = result.adam_examples[0]
        
        adam_viz = self.visualizer.create_adam_components_visualization(adam_result)
        
        assert 'iterations' in adam_viz
        assert 'parameters' in adam_viz
        assert 'gradients' in adam_viz
        assert 'momentum' in adam_viz
        assert 'velocity' in adam_viz
        assert 'momentum_corrected' in adam_viz
        assert 'velocity_corrected' in adam_viz
        assert 'losses' in adam_viz
        
        # Check data consistency
        num_steps = len(adam_viz['iterations'])
        assert len(adam_viz['parameters']) == num_steps
        assert len(adam_viz['momentum']) == num_steps
        assert len(adam_viz['velocity']) == num_steps
    
    def test_learning_rate_visualization(self):
        """Test learning rate schedule visualization."""
        result = self.tutorial.create_complete_tutorial()
        lr_data = result.learning_rate_examples
        
        lr_viz = self.visualizer.create_learning_rate_visualization(lr_data)
        
        assert 'schedules' in lr_viz
        assert 'steps' in lr_viz
        assert 'learning_rates' in lr_viz
        assert 'losses' in lr_viz
        
        assert len(lr_viz['schedules']) == len(lr_data)
        assert len(lr_viz['steps']) == len(lr_data)
        assert len(lr_viz['learning_rates']) == len(lr_data)
        assert len(lr_viz['losses']) == len(lr_data)


class TestOptimizationInteractiveComponents:
    """Test suite for optimization interactive components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interactive = OptimizationInteractiveComponents()
    
    def test_learning_rate_demo_creation(self):
        """Test learning rate interactive demo creation."""
        demo = self.interactive.create_learning_rate_demo()
        
        assert demo.loss_function is not None
        assert demo.gradient_function is not None
        assert len(demo.parameter_widgets) >= 4  # LR, initial_x, initial_y, num_steps
        assert 'trajectory' in demo.visualization_data
        assert 'losses' in demo.visualization_data
        assert 'optimization_result' in demo.current_results
        
        # Check widget structure
        widget_names = [w.parameter_name for w in demo.parameter_widgets]
        assert 'learning_rate' in widget_names
        assert 'initial_x' in widget_names
        assert 'initial_y' in widget_names
        assert 'num_steps' in widget_names
    
    def test_adam_hyperparameter_demo_creation(self):
        """Test Adam hyperparameter interactive demo creation."""
        demo = self.interactive.create_adam_hyperparameter_demo()
        
        assert demo.loss_function is not None
        assert demo.gradient_function is not None
        assert len(demo.parameter_widgets) >= 4  # LR, beta1, beta2, epsilon
        assert 'trajectory' in demo.visualization_data
        assert 'momentum' in demo.visualization_data
        assert 'velocity' in demo.visualization_data
        
        # Check Adam-specific widgets
        widget_names = [w.parameter_name for w in demo.parameter_widgets]
        assert 'learning_rate' in widget_names
        assert 'beta1' in widget_names
        assert 'beta2' in widget_names
        assert 'epsilon' in widget_names
    
    def test_optimizer_comparison_demo_creation(self):
        """Test optimizer comparison interactive demo creation."""
        demo = self.interactive.create_optimizer_comparison_demo()
        
        assert demo.loss_function is not None
        assert demo.gradient_function is not None
        assert len(demo.parameter_widgets) >= 4
        assert 'gd_trajectory' in demo.visualization_data
        assert 'adam_trajectory' in demo.visualization_data
        assert 'gd_result' in demo.current_results
        assert 'adam_result' in demo.current_results
        
        # Check comparison-specific widgets
        widget_names = [w.parameter_name for w in demo.parameter_widgets]
        assert 'gd_learning_rate' in widget_names
        assert 'adam_learning_rate' in widget_names
    
    def test_demo_update_functionality(self):
        """Test demo update functionality."""
        demo = self.interactive.create_learning_rate_demo()
        original_lr = demo.parameter_widgets[0].current_value
        
        # Update parameters
        new_params = {'learning_rate': 0.05}
        updated_demo = self.interactive.update_demo(demo, new_params)
        
        # Check that widget was updated
        lr_widget = next(w for w in updated_demo.parameter_widgets if w.parameter_name == 'learning_rate')
        assert lr_widget.current_value == 0.05
        assert lr_widget.current_value != original_lr


class TestOptimizationIntegration:
    """Integration tests for the complete optimization tutorial system."""
    
    def test_complete_tutorial_pipeline(self):
        """Test the complete tutorial generation and visualization pipeline."""
        # Create tutorial
        tutorial = OptimizationTutorial()
        result = tutorial.create_complete_tutorial()
        
        # Create visualizations
        visualizer = OptimizationLandscapeVisualizer()
        landscape_viz = visualizer.create_loss_landscape(
            result.loss_landscape_examples[0],
            result.gradient_descent_examples[:2],
            ['GD-1', 'GD-2']
        )
        
        # Create interactive components
        interactive = OptimizationInteractiveComponents()
        lr_demo = interactive.create_learning_rate_demo()
        
        # Verify integration
        assert result is not None
        assert landscape_viz is not None
        assert lr_demo is not None
        
        # Check that all components work together
        assert len(result.tutorial_sections) > 0
        assert len(landscape_viz.optimizer_paths) > 0
        assert len(lr_demo.parameter_widgets) > 0
    
    def test_mathematical_correctness(self):
        """Test mathematical correctness of optimization implementations."""
        tutorial = OptimizationTutorial()
        result = tutorial.create_complete_tutorial()
        
        # Test gradient descent on quadratic function (should converge to zero)
        gd_example = result.gradient_descent_examples[0]  # Quadratic function
        final_loss = gd_example.convergence_info['final_loss']
        initial_loss = gd_example.optimization_steps[0].loss
        
        # Loss should decrease
        assert final_loss < initial_loss
        
        # Test Adam optimizer
        adam_example = result.adam_examples[0]  # Same quadratic function
        adam_final_loss = adam_example.convergence_info['final_loss']
        adam_initial_loss = adam_example.optimization_steps[0].loss
        
        # Loss should decrease for Adam too
        assert adam_final_loss < adam_initial_loss
        
        # Test learning rate scheduling (loss should generally decrease)
        for lr_example in result.learning_rate_examples:
            results = lr_example['results']
            initial_loss = results[0]['loss']
            final_loss = results[-1]['loss']
            assert final_loss < initial_loss  # Should improve over time