"""
Tests for matrix operation visualizations.
Validates step-by-step operation breakdowns and visual components.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

from src.visualization.operation_viz.matrix_operations import (
    MatrixOperationVisualizer,
    OperationStep,
    create_sample_matrix_multiplication,
    create_sample_attention_visualization
)
from src.core.models import ColorCodedMatrix, HighlightPattern, OperationVisualization
from src.visualization.matrix_viz import ConceptType


class TestMatrixOperationVisualizer:
    """Test suite for MatrixOperationVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = MatrixOperationVisualizer()
        
        # Sample matrices for testing
        self.matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
        self.matrix_b = np.array([[7, 8], [9, 10], [11, 12]])
        self.expected_result = np.array([[58, 64], [139, 154]])
        
        # Sample attention matrices
        self.query = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        self.key = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        self.value = np.array([[1.0, 0.9, 0.8], [0.7, 0.6, 0.5], [0.4, 0.3, 0.2]])
    
    def test_initialization(self):
        """Test visualizer initialization."""
        assert self.visualizer.matrix_viz is not None
        assert self.visualizer.concept_colors is not None
        assert len(self.visualizer.concept_colors.primary_colors) > 0
    
    def test_matrix_multiplication_basic(self):
        """Test basic matrix multiplication visualization."""
        result = self.visualizer.visualize_matrix_multiplication(
            self.matrix_a, self.matrix_b, show_steps=False
        )
        
        assert isinstance(result, OperationVisualization)
        assert result.operation_type == "matrix_multiply"
        assert len(result.input_matrices) == 2
        assert result.output_matrix is not None
        
        # Verify computation correctness
        np.testing.assert_array_equal(
            result.output_matrix.matrix_data, 
            self.expected_result
        )
    
    def test_matrix_multiplication_with_steps(self):
        """Test matrix multiplication with step-by-step breakdown."""
        result = self.visualizer.visualize_matrix_multiplication(
            self.matrix_a, self.matrix_b, show_steps=True
        )
        
        assert len(result.intermediate_steps) > 0
        assert len(result.animation_sequence) > 0
        
        # Should have steps for each element in result matrix
        expected_steps = self.matrix_a.shape[0] * self.matrix_b.shape[1]
        assert len(result.animation_sequence) == expected_steps
    
    def test_matrix_multiplication_dimension_error(self):
        """Test error handling for incompatible matrix dimensions."""
        # matrix_a is 2x3, so incompatible matrix should have different first dimension than 3
        incompatible_matrix = np.array([[1, 2], [3, 4]])  # 2x2, incompatible with 2x3
        
        with pytest.raises(ValueError, match="Matrix dimensions incompatible"):
            self.visualizer.visualize_matrix_multiplication(
                self.matrix_a, incompatible_matrix
            )
    
    def test_attention_computation(self):
        """Test attention mechanism visualization."""
        result = self.visualizer.visualize_attention_computation(
            self.query, self.key, self.value
        )
        
        assert isinstance(result, OperationVisualization)
        assert result.operation_type == "attention"
        assert len(result.input_matrices) == 3  # Q, K, V
        assert len(result.intermediate_steps) == 3  # QK^T, scaled, attention weights
        assert result.output_matrix is not None
        
        # Verify attention weights sum to 1 (approximately)
        attention_weights = result.intermediate_steps[2].matrix_data
        row_sums = np.sum(attention_weights, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)), decimal=5)
    
    def test_attention_with_custom_scale(self):
        """Test attention computation with custom scale factor."""
        custom_scale = 0.5
        result = self.visualizer.visualize_attention_computation(
            self.query, self.key, self.value, scale_factor=custom_scale
        )
        
        # Verify scaling was applied
        qk_scores = np.dot(self.query, self.key.T)
        expected_scaled = qk_scores * custom_scale
        
        # The scaled scores should be in intermediate steps
        scaled_matrix = result.intermediate_steps[1].matrix_data
        np.testing.assert_array_almost_equal(scaled_matrix, expected_scaled)
    
    def test_gradient_flow_visualization(self):
        """Test gradient flow visualization."""
        # Create sample forward pass and gradients
        forward_matrices = [
            np.random.randn(3, 4),
            np.random.randn(4, 2),
            np.random.randn(2, 1)
        ]
        gradients = [
            np.random.randn(3, 4) * 0.1,
            np.random.randn(4, 2) * 0.05,
            np.random.randn(2, 1) * 0.02
        ]
        layer_names = ["input", "hidden", "output"]
        
        result = self.visualizer.visualize_gradient_flow(
            forward_matrices, gradients, layer_names
        )
        
        assert isinstance(result, OperationVisualization)
        assert result.operation_type == "gradient_flow"
        assert len(result.input_matrices) == len(forward_matrices)
        assert len(result.intermediate_steps) == len(gradients)
    
    def test_gradient_flow_dimension_error(self):
        """Test error handling for mismatched gradient flow inputs."""
        forward_matrices = [np.random.randn(3, 4)]
        gradients = [np.random.randn(3, 4), np.random.randn(4, 2)]  # Different length
        layer_names = ["input"]
        
        with pytest.raises(ValueError, match="All input lists must have the same length"):
            self.visualizer.visualize_gradient_flow(forward_matrices, gradients, layer_names)
    
    def test_create_operation_plot(self):
        """Test operation plot creation."""
        result = self.visualizer.visualize_matrix_multiplication(
            self.matrix_a, self.matrix_b, show_steps=True
        )
        
        # Just test that the method runs without error
        fig = self.visualizer.create_operation_plot(result)
        
        # Verify we got a figure back
        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up
    
    def test_matrix_labels_creation(self):
        """Test matrix label creation."""
        matrix = np.array([[1, 2], [3, 4]])
        labels = self.visualizer._create_matrix_labels(matrix, "A")
        
        expected_labels = {
            (0, 0): "A[0,0]",
            (0, 1): "A[0,1]", 
            (1, 0): "A[1,0]",
            (1, 1): "A[1,1]"
        }
        
        assert labels == expected_labels
    
    def test_softmax_computation(self):
        """Test softmax computation with numerical stability."""
        # Test with large values that could cause overflow
        large_values = np.array([[1000, 1001, 1002], [2000, 2001, 2002]])
        result = self.visualizer._softmax(large_values)
        
        # Check that rows sum to 1
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))
        
        # Check that all values are positive
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_gradient_color_mapping(self):
        """Test gradient color mapping based on magnitude."""
        # High gradient
        high_grad = np.array([[2.0, 3.0], [1.5, 2.5]])
        high_color = self.visualizer._get_gradient_color(high_grad)
        assert high_color == self.visualizer.concept_colors.get_color("high_gradient")
        
        # Medium gradient
        medium_grad = np.array([[0.5, 0.3], [0.2, 0.4]])
        medium_color = self.visualizer._get_gradient_color(medium_grad)
        assert medium_color == self.visualizer.concept_colors.get_color("medium_gradient")
        
        # Low gradient
        low_grad = np.array([[0.01, 0.02], [0.005, 0.03]])
        low_color = self.visualizer._get_gradient_color(low_grad)
        assert low_color == self.visualizer.concept_colors.get_color("low_gradient")


class TestSampleFunctions:
    """Test sample creation functions."""
    
    def test_sample_matrix_multiplication(self):
        """Test sample matrix multiplication creation."""
        result = create_sample_matrix_multiplication()
        
        assert isinstance(result, OperationVisualization)
        assert result.operation_type == "matrix_multiply"
        assert len(result.input_matrices) == 2
        assert len(result.intermediate_steps) > 0
        assert len(result.animation_sequence) > 0
    
    def test_sample_attention_visualization(self):
        """Test sample attention visualization creation."""
        result = create_sample_attention_visualization()
        
        assert isinstance(result, OperationVisualization)
        assert result.operation_type == "attention"
        assert len(result.input_matrices) == 3
        assert len(result.intermediate_steps) == 3
        assert len(result.animation_sequence) > 0


class TestOperationStep:
    """Test OperationStep dataclass."""
    
    def test_operation_step_creation(self):
        """Test OperationStep creation and attributes."""
        matrix = ColorCodedMatrix(
            matrix_data=np.array([[1, 2], [3, 4]]),
            color_mapping={"default": "#FF0000"}
        )
        
        highlight = HighlightPattern(
            pattern_type="row",
            indices=[(0, 0), (0, 1)],
            color="#FFFF00",
            description="First row"
        )
        
        step = OperationStep(
            step_number=1,
            description="Test step",
            input_matrices=[matrix],
            output_matrix=matrix,
            highlights=[highlight],
            mathematical_expression="A × B = C"
        )
        
        assert step.step_number == 1
        assert step.description == "Test step"
        assert len(step.input_matrices) == 1
        assert step.output_matrix == matrix
        assert len(step.highlights) == 1
        assert step.mathematical_expression == "A × B = C"


if __name__ == "__main__":
    pytest.main([__file__])