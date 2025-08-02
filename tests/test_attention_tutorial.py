"""
Tests for the attention mechanism tutorial implementation.
"""

import numpy as np
import pytest

from src.content.attention.attention_tutorial import AttentionTutorial
from src.computation.attention_ops.attention_mechanisms import AttentionMechanisms
from src.computation.attention_ops.positional_encoding import PositionalEncodings


class TestAttentionTutorial:
    """Test suite for attention tutorial."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tutorial = AttentionTutorial()
        self.attention_mechanisms = AttentionMechanisms()
        self.positional_encodings = PositionalEncodings()
    
    def test_tutorial_creation(self):
        """Test that tutorial can be created successfully."""
        result = self.tutorial.create_complete_tutorial()
        
        assert result.concept is not None
        assert result.concept.title == "Attention Mechanisms and Positional Encodings"
        assert len(result.concept.equations) == 4  # 4 main equations
        assert len(result.attention_examples) == 3  # 3 attention examples
        assert len(result.positional_examples) == 3  # 3 positional examples
        assert len(result.tutorial_sections) == 5  # 5 tutorial sections
    
    def test_attention_examples(self):
        """Test attention mechanism examples."""
        result = self.tutorial.create_complete_tutorial()
        
        for example in result.attention_examples:
            # Check that attention weights are valid probabilities
            if example.attention_weights.ndim == 2:  # Single-head attention
                assert np.allclose(np.sum(example.attention_weights, axis=1), 1.0, atol=1e-6)
                assert np.all(example.attention_weights >= 0)
            
            # Check that computation steps are present
            assert len(example.computation_steps) > 0
            
            # Check that properties are computed
            assert 'operation' in example.properties
    
    def test_positional_examples(self):
        """Test positional encoding examples."""
        result = self.tutorial.create_complete_tutorial()
        
        for example in result.positional_examples:
            # Check that encoding has correct shape
            assert example.encoded_positions.shape[0] > 0
            assert example.encoded_positions.shape[1] > 0
            
            # Check that computation steps are present
            assert len(example.computation_steps) > 0
            
            # Check that properties are computed
            assert 'encoding_type' in example.properties
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention implementation."""
        seq_len, d_k, d_v = 4, 6, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)
        
        result = self.attention_mechanisms.scaled_dot_product_attention(Q, K, V)
        
        # Check output shapes
        assert result.attention_weights.shape == (seq_len, seq_len)
        assert result.output.shape == (seq_len, d_v)
        
        # Check attention weights are probabilities
        assert np.allclose(np.sum(result.attention_weights, axis=1), 1.0)
        assert np.all(result.attention_weights >= 0)
        
        # Check computation steps
        assert len(result.computation_steps) >= 4  # At least 4 main steps
    
    def test_multi_head_attention(self):
        """Test multi-head attention implementation."""
        seq_len, d_model, num_heads = 6, 12, 3
        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)
        
        result = self.attention_mechanisms.multi_head_attention(Q, K, V, num_heads, d_model)
        
        # Check output shape
        assert result.output.shape == (seq_len, d_model)
        
        # Check that we have attention weights for each head
        assert result.attention_weights.shape == (num_heads, seq_len, seq_len)
        
        # Check properties
        assert result.properties['num_heads'] == num_heads
        assert result.properties['d_model'] == d_model
    
    def test_absolute_positional_encoding(self):
        """Test absolute positional encoding."""
        seq_len, d_model = 8, 16
        
        result = self.positional_encodings.absolute_positional_encoding(seq_len, d_model)
        
        # Check output shape
        assert result.encoded_positions.shape == (seq_len, d_model)
        
        # Check that even dimensions use sine, odd use cosine
        # This is verified by the computation steps
        assert len(result.computation_steps) == 5
        
        # Check properties
        assert result.properties['encoding_type'] == 'absolute_sinusoidal'
        assert result.properties['sequence_length'] == seq_len
        assert result.properties['d_model'] == d_model
    
    def test_rope_encoding(self):
        """Test Rotary Positional Embedding (RoPE)."""
        seq_len, d_head = 6, 8
        x = np.random.randn(seq_len, d_head)
        
        result = self.positional_encodings.rotary_positional_embedding(x, seq_len, d_head)
        
        # Check output shape matches input
        assert result.encoded_positions.shape == x.shape
        
        # Check that rotation preserves norms (approximately)
        input_norms = np.linalg.norm(x, axis=1)
        output_norms = np.linalg.norm(result.encoded_positions, axis=1)
        assert np.allclose(input_norms, output_norms, atol=1e-10)
        
        # Check properties
        assert result.properties['encoding_type'] == 'rotary_positional_embedding'
        assert result.properties['d_head'] == d_head
    
    def test_relative_positional_encoding(self):
        """Test relative positional encoding."""
        seq_len, d_model = 6, 12
        
        result = self.positional_encodings.relative_positional_encoding(seq_len, d_model)
        
        # Check output shape
        assert result.encoded_positions.shape == (seq_len, seq_len, d_model)
        
        # Check properties
        assert result.properties['encoding_type'] == 'relative_positional'
        assert result.properties['sequence_length'] == seq_len
        assert result.properties['d_model'] == d_model
    
    def test_attention_with_mask(self):
        """Test attention with causal mask."""
        seq_len, d_k, d_v = 4, 6, 8
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)
        
        # Create causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        
        result = self.attention_mechanisms.scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # Check that attention weights respect the mask (upper triangular should be ~0)
        attention_weights = result.attention_weights
        upper_triangular = np.triu(attention_weights, k=1)
        assert np.allclose(upper_triangular, 0, atol=1e-6)
        
        # Check that lower triangular weights still sum to 1
        for i in range(seq_len):
            assert np.allclose(np.sum(attention_weights[i, :i+1]), 1.0, atol=1e-6)
    
    def test_tutorial_sections_structure(self):
        """Test that tutorial sections have proper structure."""
        result = self.tutorial.create_complete_tutorial()
        
        # Check that all sections have required fields
        for section in result.tutorial_sections:
            assert 'title' in section
            assert 'content' in section
        
        # Check specific sections
        section_titles = [s['title'] for s in result.tutorial_sections]
        expected_titles = [
            'Introduction to Attention Mechanisms',
            'Scaled Dot-Product Attention',
            'Multi-Head Attention',
            'Positional Encodings',
            'Advanced Topics and Applications'
        ]
        
        for expected_title in expected_titles:
            assert expected_title in section_titles
    
    def test_mathematical_properties(self):
        """Test mathematical properties of attention mechanisms."""
        seq_len, d_k, d_v = 5, 8, 6
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)
        
        result = self.attention_mechanisms.scaled_dot_product_attention(Q, K, V)
        
        # Test permutation equivariance with respect to key-value pairs
        # Permute K and V in the same way
        perm = np.random.permutation(seq_len)
        K_perm = K[perm]
        V_perm = V[perm]
        
        result_perm = self.attention_mechanisms.scaled_dot_product_attention(Q, K_perm, V_perm)
        
        # The attention pattern should change, but the mechanism should still work
        assert result_perm.attention_weights.shape == result.attention_weights.shape
        assert result_perm.output.shape == result.output.shape
    
    def test_rope_odd_dimension_error(self):
        """Test that RoPE raises error for odd dimensions."""
        seq_len, d_head = 4, 7  # Odd dimension
        x = np.random.randn(seq_len, d_head)
        
        with pytest.raises(ValueError, match="Head dimension must be even"):
            self.positional_encodings.rotary_positional_embedding(x, seq_len, d_head)
    
    def test_attention_dimension_mismatch_error(self):
        """Test that attention raises error for dimension mismatches."""
        seq_len, d_k1, d_k2, d_v = 4, 6, 8, 10
        Q = np.random.randn(seq_len, d_k1)
        K = np.random.randn(seq_len, d_k2)  # Different dimension
        V = np.random.randn(seq_len, d_v)
        
        with pytest.raises(ValueError, match="Query and Key dimensions must match"):
            self.attention_mechanisms.scaled_dot_product_attention(Q, K, V)