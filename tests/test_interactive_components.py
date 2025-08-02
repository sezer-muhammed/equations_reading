"""
Tests for interactive visualization components.
Verifies parameter widgets, hover effects, and animation controls.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

from src.core.models import (
    VariableDefinition, ColorCodedMatrix, OperationVisualization,
    NumericalExample, MathematicalConcept, Equation, AnimationFrame,
    HighlightPattern
)
from src.visualization.interactive import (
    InteractiveParameterWidget, MathematicalParameterWidget, ParameterConfig,
    HoverEffectManager, VariableHighlighter, HoverRegion,
    AnimationController, StepByStepAnimationController,
    InteractiveMathVisualization, InteractiveVisualizationConfig
)


class TestParameterWidgets:
    """Test parameter manipulation widgets."""
    
    def test_parameter_config_creation(self):
        """Test parameter configuration creation."""
        config = ParameterConfig(
            name="test_param",
            display_name="Test Parameter",
            min_value=0.0,
            max_value=10.0,
            initial_value=5.0
        )
        
        assert config.name == "test_param"
        assert config.display_name == "Test Parameter"
        assert config.min_value == 0.0
        assert config.max_value == 10.0
        assert config.initial_value == 5.0
    
    def test_interactive_parameter_widget_creation(self):
        """Test interactive parameter widget creation."""
        parameters = {
            "param1": ParameterConfig("param1", "Parameter 1", 0, 10, 5),
            "param2": ParameterConfig("param2", "Parameter 2", -5, 5, 0)
        }
        
        callback = Mock()
        
        # Mock the entire widget creation process
        with patch.object(InteractiveParameterWidget, '_create_widgets'), \
             patch.object(InteractiveParameterWidget, '_setup_callbacks'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot2grid'):
            
            widget = InteractiveParameterWidget(parameters, callback)
        
        assert widget.parameters == parameters
        assert widget.update_callback == callback
        assert len(widget.current_values) == 2
        assert widget.current_values["param1"] == 5
        assert widget.current_values["param2"] == 0
    
    def test_mathematical_parameter_widget_variable_conversion(self):
        """Test conversion of VariableDefinition to ParameterConfig."""
        variables = {
            "x": VariableDefinition(
                name="x", description="Test variable", data_type="scalar",
                constraints="positive"
            ),
            "y": VariableDefinition(
                name="y", description="Test matrix", data_type="matrix",
                shape=(3, 3)
            )
        }
        
        callback = Mock()
        
        # Mock the entire widget creation process
        with patch.object(MathematicalParameterWidget, '_create_widgets'), \
             patch.object(MathematicalParameterWidget, '_setup_callbacks'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot2grid'):
            
            widget = MathematicalParameterWidget(variables, callback)
            
            assert "x" in widget.parameters
            assert "y" in widget.parameters
            
            # Check that positive constraint affects range
            x_config = widget.parameters["x"]
            assert x_config.min_value > 0


class TestHoverEffects:
    """Test hover effects for variable highlighting."""
    
    def test_hover_region_creation(self):
        """Test hover region creation."""
        region = HoverRegion(
            region_id="test_region",
            bounds=(0, 0, 1, 1),
            variable_name="x",
            description="Test variable"
        )
        
        assert region.region_id == "test_region"
        assert region.bounds == (0, 0, 1, 1)
        assert region.variable_name == "x"
        assert region.description == "Test variable"
    
    @patch('matplotlib.pyplot.subplots')
    def test_hover_effect_manager_creation(self, mock_subplots):
        """Test hover effect manager creation."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock canvas and mpl_connect
        mock_canvas = Mock()
        mock_fig.canvas = mock_canvas
        
        manager = HoverEffectManager(mock_fig, mock_ax)
        
        assert manager.figure == mock_fig
        assert manager.axes == mock_ax
        assert len(manager.hover_regions) == 0
        
        # Verify event connections were made
        assert mock_canvas.mpl_connect.call_count == 2
    
    def test_variable_highlighter_color_assignment(self):
        """Test variable color assignment."""
        mock_manager = Mock()
        highlighter = VariableHighlighter(mock_manager)
        
        variables = {
            "Q": VariableDefinition("Q", "Query", "matrix", color_code="#FF0000"),
            "K": VariableDefinition("K", "Key", "matrix"),
            "V": VariableDefinition("V", "Value", "matrix")
        }
        
        highlighter._assign_variable_colors(variables)
        
        assert highlighter.variable_colors["Q"] == "#FF0000"
        assert "K" in highlighter.variable_colors
        assert "V" in highlighter.variable_colors


class TestAnimationControls:
    """Test animation controls for step-by-step demonstrations."""
    
    def test_animation_frame_creation(self):
        """Test animation frame creation."""
        matrix_data = np.random.randn(3, 3)
        highlights = [
            HighlightPattern("element", [(0, 0)], "red", "Test highlight")
        ]
        
        frame = AnimationFrame(
            frame_number=0,
            matrix_state=matrix_data,
            highlights=highlights,
            description="Test frame"
        )
        
        assert frame.frame_number == 0
        assert np.array_equal(frame.matrix_state, matrix_data)
        assert len(frame.highlights) == 1
        assert frame.description == "Test frame"
    
    def test_animation_controller_creation(self):
        """Test animation controller creation."""
        mock_fig = Mock()
        mock_fig.subplots_adjust = Mock()
        
        frames = [
            AnimationFrame(0, np.random.randn(2, 2), [], "Frame 1"),
            AnimationFrame(1, np.random.randn(2, 2), [], "Frame 2")
        ]
        
        callback = Mock()
        
        # Mock the entire control setup process
        with patch.object(AnimationController, '_create_control_panel'), \
             patch.object(AnimationController, '_setup_controls'):
            
            controller = AnimationController(mock_fig, frames, callback)
            
            assert controller.figure == mock_fig
            assert len(controller.frames) == 2
            assert controller.render_callback == callback
            assert controller.current_frame == 0
            assert not controller.is_playing
    
    def test_step_by_step_controller_frame_creation(self):
        """Test step-by-step controller frame creation from operation."""
        # Create sample operation visualization
        input_matrix = ColorCodedMatrix(np.random.randn(2, 2), {})
        output_matrix = ColorCodedMatrix(np.random.randn(2, 2), {})
        
        operation_viz = OperationVisualization(
            operation_type="matrix_multiply",
            input_matrices=[input_matrix],
            intermediate_steps=[],
            output_matrix=output_matrix
        )
        
        callback = Mock()
        
        mock_fig = Mock()
        mock_fig.subplots_adjust = Mock()
        
        # Mock the entire control setup process
        with patch.object(StepByStepAnimationController, '_create_control_panel'), \
             patch.object(StepByStepAnimationController, '_setup_controls'), \
             patch.object(StepByStepAnimationController, '_create_step_description_display'):
            
            controller = StepByStepAnimationController(
                mock_fig, operation_viz, callback
            )
            
            # Should create frames from operation
            assert len(controller.frames) >= 2  # At least input and output


class TestInteractiveComponents:
    """Test main interactive visualization components."""
    
    def test_interactive_visualization_config(self):
        """Test interactive visualization configuration."""
        config = InteractiveVisualizationConfig(
            enable_parameter_widgets=True,
            enable_hover_effects=True,
            enable_animation_controls=False,
            figure_size=(12, 8)
        )
        
        assert config.enable_parameter_widgets
        assert config.enable_hover_effects
        assert not config.enable_animation_controls
        assert config.figure_size == (12, 8)
    
    def test_interactive_math_visualization_creation(self):
        """Test interactive math visualization creation."""
        # Create sample concept
        variables = {
            "x": VariableDefinition("x", "Variable x", "scalar")
        }
        
        equation = Equation(
            equation_id="test_eq",
            latex_expression="y = x^2",
            variables=variables
        )
        
        example = NumericalExample(
            example_id="test_example",
            description="Test example",
            input_values={"x": np.array(2.0)},
            computation_steps=[],
            output_values={"y": np.array(4.0)}
        )
        
        concept = MathematicalConcept(
            concept_id="test_concept",
            title="Test Concept",
            prerequisites=[],
            equations=[equation],
            explanations=[],
            examples=[example],
            visualizations=[],
            difficulty_level=1
        )
        
        # Mock the widget creation parts but allow basic initialization
        with patch.object(InteractiveMathVisualization, '_initialize_parameter_widgets'), \
             patch.object(InteractiveMathVisualization, '_initialize_hover_effects'), \
             patch.object(InteractiveMathVisualization, '_initialize_animation_controls'):
            
            viz = InteractiveMathVisualization(concept)
            
            assert viz.concept == concept
            assert viz.current_example == example
    
    def test_parameter_application_to_example(self):
        """Test applying parameter changes to numerical examples."""
        # Create sample example
        example = NumericalExample(
            example_id="test",
            description="Test",
            input_values={"x": np.array(2.0), "matrix": np.ones((2, 2))},
            computation_steps=[],
            output_values={"y": np.array(4.0)}
        )
        
        concept = MathematicalConcept(
            concept_id="test",
            title="Test",
            prerequisites=[],
            equations=[],
            explanations=[],
            examples=[example],
            visualizations=[],
            difficulty_level=1
        )
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.axes'), \
             patch('matplotlib.pyplot.subplot2grid'):
            
            viz = InteractiveMathVisualization(concept)
            
            # Apply parameter changes
            new_params = {"x": 3.0}
            updated_example = viz._apply_parameters_to_example(example, new_params)
            
            assert updated_example.input_values["x"] == 3.0
            assert "x_result" in updated_example.output_values


class TestIntegration:
    """Integration tests for interactive components."""
    
    @patch('matplotlib.pyplot.show')
    def test_attention_demo_creation(self, mock_show):
        """Test creation of attention mechanism demo."""
        # Mock the entire initialization process
        with patch.object(InteractiveMathVisualization, '_initialize_components'):
            # This should not raise any exceptions
            from src.visualization.interactive.interactive_components import create_attention_interactive_demo
            
            demo = create_attention_interactive_demo()
            
            assert demo.concept.concept_id == "attention_mechanism"
            assert len(demo.concept.equations) == 1
            assert len(demo.concept.examples) == 1
    
    def test_matrix_operations_with_hover_effects(self):
        """Test matrix operations with hover effects integration."""
        # Create sample matrix
        matrix_data = np.random.randn(3, 3)
        matrix = ColorCodedMatrix(matrix_data, {})
        
        # Create variables
        variables = {
            "A": VariableDefinition("A", "Matrix A", "matrix", shape=(3, 3))
        }
        
        with patch('matplotlib.pyplot.subplots'):
            mock_fig = Mock()
            mock_ax = Mock()
            mock_fig.canvas = Mock()
            
            # Create hover manager
            hover_manager = HoverEffectManager(mock_fig, mock_ax)
            
            # Create variable highlighter
            highlighter = VariableHighlighter(hover_manager)
            
            # Setup highlighting - should not raise exceptions
            highlighter.setup_variable_highlighting(variables, matrix)
            
            assert len(highlighter.variable_colors) == 1
            assert "A" in highlighter.variable_colors


if __name__ == "__main__":
    pytest.main([__file__])