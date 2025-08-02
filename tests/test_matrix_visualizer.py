"""
Tests for the color-coded matrix visualizer components.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing

import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from src.core.models import ColorCodedMatrix, HighlightPattern
from src.visualization.matrix_viz import (
    ColorCodedMatrixVisualizer,
    MatrixVisualizationStyle,
    MatrixAnimationEngine,
    ConceptType,
    get_concept_colors,
    create_sample_matrix
)


class TestColorCodedMatrixVisualizer:
    """Test cases for the ColorCodedMatrixVisualizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = ColorCodedMatrixVisualizer()
        self.sample_matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
    
    def test_visualizer_initialization(self):
        """Test that visualizer initializes correctly."""
        assert self.visualizer is not None
        assert self.visualizer.style is not None
        assert self.visualizer.config is not None
        assert 'matrix' in self.visualizer.color_schemes
    
    def test_create_matrix_visualization(self):
        """Test basic matrix visualization creation."""
        matrix = ColorCodedMatrix(
            matrix_data=self.sample_matrix,
            color_mapping={"default": "viridis"}
        )
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            fig = self.visualizer.create_matrix_visualization(matrix, "Test Matrix")
            
            assert mock_subplots.called
            mock_ax.set_xlim.assert_called_once()
            mock_ax.set_ylim.assert_called_once()
    
    def test_color_mapping_creation(self):
        """Test color mapping creation for different schemes."""
        matrix = ColorCodedMatrix(
            matrix_data=self.sample_matrix,
            color_mapping={"default": "viridis"}
        )
        
        colors = self.visualizer._create_color_mapping(matrix, "matrix")
        
        assert colors is not None
        assert colors.shape == self.sample_matrix.shape or colors.shape == self.sample_matrix.shape + (4,)
    
    def test_highlight_patterns(self):
        """Test that highlight patterns are applied correctly."""
        highlights = [
            HighlightPattern(
                pattern_type="row",
                indices=[(0,)],
                color="red",
                description="First row"
            ),
            HighlightPattern(
                pattern_type="element",
                indices=[(1, 1)],
                color="blue", 
                description="Center element"
            )
        ]
        
        matrix = ColorCodedMatrix(
            matrix_data=self.sample_matrix,
            color_mapping={"default": "viridis"},
            highlight_patterns=highlights
        )
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            self.visualizer.create_matrix_visualization(matrix)
            
            # Verify that add_patch was called for highlights
            assert mock_ax.add_patch.call_count >= len(highlights)
    
    def test_element_labels(self):
        """Test that element labels are displayed correctly."""
        matrix = ColorCodedMatrix(
            matrix_data=self.sample_matrix,
            color_mapping={"default": "viridis"},
            element_labels={(0, 0): "A", (2, 2): "B"}
        )
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            self.visualizer.create_matrix_visualization(matrix)
            
            # Verify that text was added for labels
            assert mock_ax.text.call_count >= len(matrix.element_labels)
    
    def test_custom_style(self):
        """Test visualization with custom style settings."""
        custom_style = MatrixVisualizationStyle(
            cell_size=2.0,
            font_size=14,
            show_values=False,
            colorbar=False
        )
        
        visualizer = ColorCodedMatrixVisualizer(style=custom_style)
        matrix = ColorCodedMatrix(
            matrix_data=self.sample_matrix,
            color_mapping={"default": "viridis"}
        )
        
        assert visualizer.style.cell_size == 2.0
        assert visualizer.style.font_size == 14
        assert not visualizer.style.show_values
        assert not visualizer.style.colorbar


class TestMatrixAnimationEngine:
    """Test cases for the MatrixAnimationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MatrixAnimationEngine()
        self.matrix_a = np.array([[1, 2], [3, 4]])
        self.matrix_b = np.array([[5, 6], [7, 8]])
    
    def test_engine_initialization(self):
        """Test that animation engine initializes correctly."""
        assert self.engine is not None
        assert self.engine.visualizer is not None
        assert self.engine.config is not None
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.animation.FuncAnimation')
    def test_matrix_multiplication_animation(self, mock_animation, mock_subplots):
        """Test matrix multiplication animation creation."""
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        animation = self.engine.create_matrix_multiplication_animation(
            self.matrix_a, self.matrix_b
        )
        
        mock_subplots.assert_called_once()
        mock_animation.assert_called_once()
    
    def test_incompatible_matrix_dimensions(self):
        """Test error handling for incompatible matrix dimensions."""
        incompatible_matrix = np.array([[1, 2, 3]])  # 1x3 matrix
        
        with pytest.raises(ValueError, match="Matrix dimensions incompatible"):
            self.engine.create_matrix_multiplication_animation(
                self.matrix_a, incompatible_matrix
            )
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.animation.FuncAnimation')
    def test_attention_animation(self, mock_animation, mock_subplots):
        """Test attention mechanism animation creation."""
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        query = np.random.randn(3, 4)
        key = np.random.randn(5, 4)
        value = np.random.randn(5, 6)
        
        animation = self.engine.create_attention_animation(query, key, value)
        
        mock_subplots.assert_called_once()
        mock_animation.assert_called_once()
    
    def test_softmax_calculation(self):
        """Test softmax calculation in attention animation."""
        test_scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        result = self.engine._softmax(test_scores)
        
        # Check that softmax properties hold
        assert np.allclose(np.sum(result, axis=-1), 1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestColorSchemes:
    """Test cases for color scheme management."""
    
    def test_concept_color_retrieval(self):
        """Test retrieving colors for different concept types."""
        linear_algebra_scheme = get_concept_colors(ConceptType.LINEAR_ALGEBRA)
        attention_scheme = get_concept_colors(ConceptType.ATTENTION)
        
        assert linear_algebra_scheme is not None
        assert attention_scheme is not None
        assert linear_algebra_scheme.concept_type == ConceptType.LINEAR_ALGEBRA
        assert attention_scheme.concept_type == ConceptType.ATTENTION
    
    def test_color_consistency(self):
        """Test that related concepts have consistent colors."""
        from src.visualization.matrix_viz.color_schemes import color_manager
        
        # Test that related concepts share some color elements
        la_scheme = color_manager.get_scheme(ConceptType.LINEAR_ALGEBRA)
        nn_scheme = color_manager.get_scheme(ConceptType.NEURAL_NETWORK)
        
        assert la_scheme is not None
        assert nn_scheme is not None
        
        # Both should have matrix-related colors
        la_matrix_color = la_scheme.get_color("matrix")
        nn_weight_color = nn_scheme.get_color("weight")
        
        assert la_matrix_color.startswith("#")
        assert nn_weight_color.startswith("#")
    
    def test_gradient_color_generation(self):
        """Test gradient color generation."""
        scheme = get_concept_colors(ConceptType.ATTENTION)
        
        color_low = scheme.get_gradient_color(0.0)
        color_mid = scheme.get_gradient_color(0.5)
        color_high = scheme.get_gradient_color(1.0)
        
        assert color_low.startswith("#")
        assert color_mid.startswith("#")
        assert color_high.startswith("#")
        assert color_low != color_high  # Should be different colors


class TestIntegration:
    """Integration tests for the complete matrix visualization system."""
    
    def test_sample_matrix_creation(self):
        """Test that sample matrix creation works end-to-end."""
        sample_matrix = create_sample_matrix()
        
        assert isinstance(sample_matrix, ColorCodedMatrix)
        assert sample_matrix.matrix_data is not None
        assert sample_matrix.matrix_data.shape == (4, 4)
        assert len(sample_matrix.highlight_patterns) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_complete_visualization_pipeline(self, mock_show):
        """Test the complete visualization pipeline."""
        # Create sample data
        sample_matrix = create_sample_matrix()
        
        # Create visualizer
        visualizer = ColorCodedMatrixVisualizer()
        
        # Generate visualization
        fig = visualizer.create_matrix_visualization(
            sample_matrix, 
            "Integration Test Matrix"
        )
        
        assert fig is not None
    
    def test_animation_sample_creation(self):
        """Test that sample animation creation works."""
        from src.visualization.matrix_viz import create_sample_multiplication_animation
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            with patch('matplotlib.animation.FuncAnimation') as mock_animation:
                mock_fig = MagicMock()
                mock_axes = [MagicMock(), MagicMock(), MagicMock()]
                mock_subplots.return_value = (mock_fig, mock_axes)
                
                animation = create_sample_multiplication_animation()
                
                mock_subplots.assert_called_once()
                mock_animation.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])