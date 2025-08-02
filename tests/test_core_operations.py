"""
Tests for core mathematical operations in the computational backend.
"""

import numpy as np
import pytest
from src.computation.matrix_ops import CoreMatrixOperations
from src.computation.attention_ops import AttentionMechanisms
from src.computation.optimization import GradientDescentOptimizer, AdamOptimizer


class TestCoreMatrixOperations:
    """Test matrix operations with visualization data."""
    
    def setup_method(self):
        self.matrix_ops = CoreMatrixOperations()
    
    def test_matrix_multiply(self):
        """Test matrix multiplication with visualization."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        result = self.matrix_ops.matrix_multiply(A, B)
        
        # Check mathematical correctness
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result.result, expected)
        
        # Check visualization components
        assert result.visualization.operation_type == "matrix_multiply"
        assert len(result.visualization.input_matrices) == 2
        assert result.visualization.output_matrix.matrix_data.shape == (2, 2)
        
        # Check computation steps
        assert len(result.computation_steps) == 4  # 2x2 result = 4 steps
        assert all(step.operation_name.startswith("dot_product") for step in result.computation_steps)
        
        # Check properties
        assert result.properties['operation'] == 'matrix_multiplication'
        assert result.properties['input_shapes'] == [(2, 2), (2, 2)]
        assert result.properties['output_shape'] == (2, 2)
    
    def test_matrix_transpose(self):
        """Test matrix transpose operation."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        
        result = self.matrix_ops.matrix_transpose(A)
        
        # Check mathematical correctness
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result.result, expected)
        
        # Check properties
        assert result.properties['operation'] == 'transpose'
        assert result.properties['input_shape'] == (2, 3)
        assert result.properties['output_shape'] == (3, 2)
    
    def test_elementwise_operations(self):
        """Test element-wise operations."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        # Test addition
        result_add = self.matrix_ops.matrix_elementwise_ops(A, B, 'add')
        expected_add = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(result_add.result, expected_add)
        
        # Test multiplication
        result_mul = self.matrix_ops.matrix_elementwise_ops(A, B, 'multiply')
        expected_mul = np.array([[5, 12], [21, 32]])
        np.testing.assert_array_equal(result_mul.result, expected_mul)
    
    def test_incompatible_dimensions(self):
        """Test error handling for incompatible matrix dimensions."""
        A = np.array([[1, 2]])  # 1x2
        B = np.array([[1], [2], [3]])  # 3x1
        
        with pytest.raises(ValueError, match="Matrix dimensions incompatible"):
            self.matrix_ops.matrix_multiply(A, B)


class TestAttentionMechanisms:
    """Test attention mechanism calculations."""
    
    def setup_method(self):
        self.attention = AttentionMechanisms()
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention mechanism."""
        # Simple test case
        seq_len, d_k, d_v = 3, 4, 5
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)
        
        result = self.attention.scaled_dot_product_attention(Q, K, V)
        
        # Check output shape
        assert result.output.shape == (seq_len, d_v)
        assert result.attention_weights.shape == (seq_len, seq_len)
        
        # Check attention weights sum to 1
        attention_sums = np.sum(result.attention_weights, axis=1)
        np.testing.assert_array_almost_equal(attention_sums, np.ones(seq_len), decimal=6)
        
        # Check computation steps
        assert len(result.computation_steps) >= 4  # At least QK^T, scale, softmax, output
        
        # Check properties
        assert result.properties['operation'] == 'scaled_dot_product_attention'
        assert result.properties['dimensions']['d_k'] == d_k
        assert result.properties['dimensions']['d_v'] == d_v
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        seq_len, d_model, num_heads = 4, 8, 2
        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)
        
        result = self.attention.multi_head_attention(Q, K, V, num_heads, d_model)
        
        # Check output shape
        assert result.output.shape == (seq_len, d_model)
        assert result.attention_weights.shape == (num_heads, seq_len, seq_len)
        
        # Check computation steps
        assert len(result.computation_steps) == 4  # Projection, reshape, attention, output
        
        # Check properties
        assert result.properties['operation'] == 'multi_head_attention'
        assert result.properties['num_heads'] == num_heads
        assert result.properties['d_model'] == d_model
    
    def test_attention_with_mask(self):
        """Test attention with masking."""
        seq_len, d_k = 3, 4
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        
        # Create causal mask (upper triangular)
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        
        result = self.attention.scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # Check that attention weights respect the mask
        # Upper triangular part should be near zero
        upper_triangular = np.triu(result.attention_weights, k=1)
        assert np.all(upper_triangular < 1e-6)


class TestOptimizers:
    """Test optimization algorithms."""
    
    def setup_method(self):
        # Simple quadratic loss function: f(x) = (x - 2)^2
        self.target = 2.0
        self.loss_fn = lambda x: np.sum((x - self.target) ** 2)
        self.grad_fn = lambda x: 2 * (x - self.target)
    
    def test_gradient_descent(self):
        """Test gradient descent optimizer."""
        optimizer = GradientDescentOptimizer(learning_rate=0.1)
        initial_params = np.array([0.0])
        
        result = optimizer.optimize(
            initial_params, self.loss_fn, self.grad_fn, num_steps=50
        )
        
        # Check convergence to target
        assert abs(result.final_parameters[0] - self.target) < 0.1
        
        # Check optimization steps
        assert len(result.optimization_steps) > 0
        assert result.convergence_info['final_loss'] < result.optimization_steps[0].loss
        
        # Check computation steps
        assert len(result.computation_steps) > 0
        assert any(step.operation_name == "compute_gradients" for step in result.computation_steps)
        assert any(step.operation_name == "update_parameters" for step in result.computation_steps)
    
    def test_adam_optimizer(self):
        """Test Adam optimizer."""
        optimizer = AdamOptimizer(learning_rate=0.1)
        initial_params = np.array([0.0])
        
        result = optimizer.optimize(
            initial_params, self.loss_fn, self.grad_fn, num_steps=50
        )
        
        # Check convergence to target
        assert abs(result.final_parameters[0] - self.target) < 0.1
        
        # Check Adam-specific properties
        assert 'adam_hyperparameters' in result.convergence_info
        adam_params = result.convergence_info['adam_hyperparameters']
        assert adam_params['beta1'] == 0.9
        assert adam_params['beta2'] == 0.999
        
        # Check that momentum and velocity are tracked
        assert 'm' in result.optimization_steps[-1].additional_info
        assert 'v' in result.optimization_steps[-1].additional_info
        assert 'm_hat' in result.optimization_steps[-1].additional_info
        assert 'v_hat' in result.optimization_steps[-1].additional_info
    
    def test_optimizer_convergence_properties(self):
        """Test convergence properties of optimizers."""
        gd_optimizer = GradientDescentOptimizer(learning_rate=0.1)
        adam_optimizer = AdamOptimizer(learning_rate=0.1)
        
        initial_params = np.array([5.0])  # Start far from optimum
        
        gd_result = gd_optimizer.optimize(
            initial_params, self.loss_fn, self.grad_fn, num_steps=100
        )
        
        adam_result = adam_optimizer.optimize(
            initial_params, self.loss_fn, self.grad_fn, num_steps=100
        )
        
        # Both should converge
        assert gd_result.convergence_info['loss_reduction'] > 0
        assert adam_result.convergence_info['loss_reduction'] > 0
        
        # Both should reach similar final values
        assert abs(gd_result.final_parameters[0] - adam_result.final_parameters[0]) < 0.5


if __name__ == "__main__":
    pytest.main([__file__])