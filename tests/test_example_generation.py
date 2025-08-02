"""
Tests for the example generation system.
"""

import numpy as np
import pytest
from src.computation.example_generation import (
    ParameterGenerator, ParameterValidator, ExampleGenerator,
    ParameterType, ValidationResult
)


class TestParameterGenerator:
    """Test parameter generation functionality."""
    
    def setup_method(self):
        self.generator = ParameterGenerator(seed=42)
    
    def test_weight_matrix_generation(self):
        """Test weight matrix generation with different initializations."""
        shape = (10, 5)
        
        # Test Xavier initialization
        weights_xavier = self.generator.generate_parameter(
            ParameterType.WEIGHT_MATRIX, shape, {'initialization': 'xavier'}
        )
        assert weights_xavier.shape == shape
        
        # Check Xavier bounds
        fan_in, fan_out = shape
        expected_limit = np.sqrt(6.0 / (fan_in + fan_out))
        assert np.all(np.abs(weights_xavier) <= expected_limit * 1.1)  # Small tolerance
        
        # Test He initialization
        weights_he = self.generator.generate_parameter(
            ParameterType.WEIGHT_MATRIX, shape, {'initialization': 'he'}
        )
        assert weights_he.shape == shape
        
        # Test normal initialization
        weights_normal = self.generator.generate_parameter(
            ParameterType.WEIGHT_MATRIX, shape, 
            {'initialization': 'normal', 'std_range': (0.01, 0.02)}
        )
        assert weights_normal.shape == shape
    
    def test_bias_vector_generation(self):
        """Test bias vector generation."""
        shape = (10,)
        
        # Test zero initialization
        bias_zero = self.generator.generate_parameter(
            ParameterType.BIAS_VECTOR, shape, {'initialization': 'zero'}
        )
        assert bias_zero.shape == shape
        np.testing.assert_array_equal(bias_zero, np.zeros(shape))
        
        # Test zero_or_small initialization
        bias_small = self.generator.generate_parameter(
            ParameterType.BIAS_VECTOR, shape, {'initialization': 'zero_or_small'}
        )
        assert bias_small.shape == shape
    
    def test_learning_rate_generation(self):
        """Test learning rate generation."""
        # Test log scale generation
        lr_log = self.generator.generate_parameter(
            ParameterType.LEARNING_RATE, (1,), 
            {'min': 1e-4, 'max': 1e-1, 'log_scale': True}
        )
        assert lr_log.shape == (1,)
        assert 1e-4 <= lr_log[0] <= 1e-1
        
        # Test linear scale generation
        lr_linear = self.generator.generate_parameter(
            ParameterType.LEARNING_RATE, (1,),
            {'min': 0.01, 'max': 0.1, 'log_scale': False}
        )
        assert lr_linear.shape == (1,)
        assert 0.01 <= lr_linear[0] <= 0.1
    
    def test_attention_scores_generation(self):
        """Test attention scores generation."""
        shape = (5, 5)
        scores = self.generator.generate_parameter(
            ParameterType.ATTENTION_SCORES, shape
        )
        assert scores.shape == shape
        
        # Check that diagonal elements are enhanced
        diagonal_mean = np.mean(np.diag(scores))
        off_diagonal_mean = np.mean(scores[~np.eye(shape[0], dtype=bool)])
        assert diagonal_mean > off_diagonal_mean
    
    def test_probability_distribution_generation(self):
        """Test probability distribution generation."""
        shape = (3, 5)
        probs = self.generator.generate_parameter(
            ParameterType.PROBABILITY_DISTRIBUTION, shape
        )
        assert probs.shape == shape
        
        # Check that each row sums to 1
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=6)
        
        # Check non-negativity
        assert np.all(probs >= 0)
    
    def test_embedding_vector_generation(self):
        """Test embedding vector generation."""
        shape = (10, 64)
        
        # Test unit normalization
        embeddings_unit = self.generator.generate_parameter(
            ParameterType.EMBEDDING_VECTOR, shape, {'normalization': 'unit'}
        )
        assert embeddings_unit.shape == shape
        norms = np.linalg.norm(embeddings_unit, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(10), decimal=6)
        
        # Test bounded normalization
        embeddings_bounded = self.generator.generate_parameter(
            ParameterType.EMBEDDING_VECTOR, shape, {'normalization': 'bounded'}
        )
        assert embeddings_bounded.shape == shape
        assert np.all(embeddings_bounded >= -2.0)
        assert np.all(embeddings_bounded <= 2.0)


class TestParameterValidator:
    """Test parameter validation functionality."""
    
    def setup_method(self):
        self.validator = ParameterValidator()
    
    def test_weight_matrix_validation(self):
        """Test weight matrix validation."""
        # Valid weight matrix
        valid_weights = np.random.normal(0, 0.1, (10, 5))
        result = self.validator.validate_parameter(
            valid_weights, ParameterType.WEIGHT_MATRIX
        )
        assert result.is_valid
        assert 'max_magnitude' in result.properties
        assert 'condition_number' in result.properties
        
        # Invalid weight matrix with NaN
        invalid_weights = valid_weights.copy()
        invalid_weights[0, 0] = np.nan
        result = self.validator.validate_parameter(
            invalid_weights, ParameterType.WEIGHT_MATRIX
        )
        assert not result.is_valid
        assert any("NaN" in issue for issue in result.issues)
        
        # Weight matrix with large values
        large_weights = np.ones((5, 5)) * 15.0
        result = self.validator.validate_parameter(
            large_weights, ParameterType.WEIGHT_MATRIX
        )
        assert result.is_valid  # Still valid but should have warnings
        assert any("Large weight values" in warning for warning in result.warnings)
    
    def test_learning_rate_validation(self):
        """Test learning rate validation."""
        # Valid learning rate
        valid_lr = np.array([0.001])
        result = self.validator.validate_parameter(
            valid_lr, ParameterType.LEARNING_RATE
        )
        assert result.is_valid
        assert result.properties['value'] == 0.001
        
        # Invalid learning rate (negative)
        invalid_lr = np.array([-0.001])
        result = self.validator.validate_parameter(
            invalid_lr, ParameterType.LEARNING_RATE
        )
        assert not result.is_valid
        assert any("must be positive" in issue for issue in result.issues)
        
        # Very high learning rate
        high_lr = np.array([2.0])
        result = self.validator.validate_parameter(
            high_lr, ParameterType.LEARNING_RATE
        )
        assert result.is_valid
        assert any("Very high" in warning for warning in result.warnings)
    
    def test_probability_distribution_validation(self):
        """Test probability distribution validation."""
        # Valid probability distribution
        valid_probs = np.array([0.3, 0.5, 0.2])
        result = self.validator.validate_parameter(
            valid_probs, ParameterType.PROBABILITY_DISTRIBUTION
        )
        assert result.is_valid
        assert 'entropy' in result.properties
        
        # Invalid probability distribution (doesn't sum to 1)
        invalid_probs = np.array([0.3, 0.5, 0.3])
        result = self.validator.validate_parameter(
            invalid_probs, ParameterType.PROBABILITY_DISTRIBUTION
        )
        assert not result.is_valid
        assert any("sum to 1" in issue for issue in result.issues)
        
        # Invalid probability distribution (negative values)
        negative_probs = np.array([0.5, -0.1, 0.6])
        result = self.validator.validate_parameter(
            negative_probs, ParameterType.PROBABILITY_DISTRIBUTION
        )
        assert not result.is_valid
        assert any("non-negative" in issue for issue in result.issues)
    
    def test_attention_scores_validation(self):
        """Test attention scores validation."""
        # Valid attention scores
        valid_scores = np.random.normal(0, 2, (5, 5))
        result = self.validator.validate_parameter(
            valid_scores, ParameterType.ATTENTION_SCORES
        )
        assert result.is_valid
        
        # Extreme attention scores
        extreme_scores = np.ones((3, 3)) * 100
        result = self.validator.validate_parameter(
            extreme_scores, ParameterType.ATTENTION_SCORES
        )
        assert result.is_valid
        assert any("softmax overflow" in warning for warning in result.warnings)
    
    def test_embedding_vector_validation(self):
        """Test embedding vector validation."""
        # Valid embedding vectors
        valid_embeddings = np.random.normal(0, 1, (10, 64))
        result = self.validator.validate_parameter(
            valid_embeddings, ParameterType.EMBEDDING_VECTOR
        )
        assert result.is_valid
        assert 'norms' in result.properties
        assert 'mean_norm' in result.properties
        
        # Very large embeddings
        large_embeddings = np.ones((5, 10)) * 20
        result = self.validator.validate_parameter(
            large_embeddings, ParameterType.EMBEDDING_VECTOR
        )
        assert result.is_valid
        assert any("Very large" in warning for warning in result.warnings)


class TestExampleGenerator:
    """Test complete example generation."""
    
    def setup_method(self):
        self.generator = ExampleGenerator(seed=42)
    
    def test_attention_example_generation(self):
        """Test generation of attention mechanism example."""
        parameter_specs = {
            'Q': (ParameterType.EMBEDDING_VECTOR, (4, 8)),
            'K': (ParameterType.EMBEDDING_VECTOR, (4, 8)),
            'V': (ParameterType.EMBEDDING_VECTOR, (4, 8))
        }
        
        example = self.generator.generate_example(
            "scaled_dot_product_attention", parameter_specs
        )
        
        assert example.example_id.startswith("scaled_dot_product_attention_example")
        assert len(example.input_values) == 3
        assert 'Q' in example.input_values
        assert 'K' in example.input_values
        assert 'V' in example.input_values
        
        # Check shapes
        assert example.input_values['Q'].shape == (4, 8)
        assert example.input_values['K'].shape == (4, 8)
        assert example.input_values['V'].shape == (4, 8)
        
        # Check educational notes
        assert len(example.educational_notes) > 0
        assert any("3 parameters" in note for note in example.educational_notes)
    
    def test_matrix_multiplication_example_generation(self):
        """Test generation of matrix multiplication example."""
        parameter_specs = {
            'A': (ParameterType.WEIGHT_MATRIX, (3, 4)),
            'B': (ParameterType.WEIGHT_MATRIX, (4, 5))
        }
        
        custom_constraints = {
            'A': {'initialization': 'xavier'},
            'B': {'initialization': 'he'}
        }
        
        example = self.generator.generate_example(
            "matrix_multiplication", parameter_specs, custom_constraints
        )
        
        assert len(example.input_values) == 2
        assert example.input_values['A'].shape == (3, 4)
        assert example.input_values['B'].shape == (4, 5)
    
    def test_optimization_example_generation(self):
        """Test generation of optimization example."""
        parameter_specs = {
            'initial_params': (ParameterType.WEIGHT_MATRIX, (10,)),
            'learning_rate': (ParameterType.LEARNING_RATE, (1,))
        }
        
        example = self.generator.generate_example(
            "gradient_descent", parameter_specs
        )
        
        assert len(example.input_values) == 2
        assert example.input_values['initial_params'].shape == (10,)
        assert example.input_values['learning_rate'].shape == (1,)
        assert example.input_values['learning_rate'][0] > 0
    
    def test_invalid_parameter_retry(self):
        """Test that generator retries when parameters are invalid."""
        # Create a specification that's likely to fail validation
        parameter_specs = {
            'learning_rate': (ParameterType.LEARNING_RATE, (1,))
        }
        
        # Force negative learning rates to test retry mechanism
        custom_constraints = {
            'learning_rate': {'min': -1.0, 'max': -0.1}  # Invalid range
        }
        
        with pytest.raises(RuntimeError, match="Failed to generate valid example"):
            self.generator.generate_example(
                "invalid_test", parameter_specs, custom_constraints, max_retries=3
            )
    
    def test_complex_example_generation(self):
        """Test generation of complex example with multiple parameter types."""
        parameter_specs = {
            'weights': (ParameterType.WEIGHT_MATRIX, (64, 32)),
            'bias': (ParameterType.BIAS_VECTOR, (32,)),
            'attention_scores': (ParameterType.ATTENTION_SCORES, (8, 8)),
            'probabilities': (ParameterType.PROBABILITY_DISTRIBUTION, (5,)),
            'embeddings': (ParameterType.EMBEDDING_VECTOR, (10, 64)),
            'learning_rate': (ParameterType.LEARNING_RATE, (1,))
        }
        
        example = self.generator.generate_example(
            "complex_neural_network", parameter_specs
        )
        
        assert len(example.input_values) == 6
        
        # Verify all parameters have correct shapes
        assert example.input_values['weights'].shape == (64, 32)
        assert example.input_values['bias'].shape == (32,)
        assert example.input_values['attention_scores'].shape == (8, 8)
        assert example.input_values['probabilities'].shape == (5,)
        assert example.input_values['embeddings'].shape == (10, 64)
        assert example.input_values['learning_rate'].shape == (1,)
        
        # Verify probability distribution sums to 1
        prob_sum = np.sum(example.input_values['probabilities'])
        assert abs(prob_sum - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])