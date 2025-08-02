"""
Transformer architecture components with detailed mathematical breakdowns.
Implements complete transformer block, layer normalization, feed-forward networks.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from ...core.models import (
    ColorCodedMatrix, OperationVisualization, ComputationStep, VisualizationData
)
from .attention_mechanisms import AttentionMechanisms, AttentionResult


@dataclass
class LayerNormResult:
    """Result of layer normalization computation."""
    normalized_output: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    computation_steps: List[ComputationStep]
    properties: Dict[str, any]


@dataclass
class FeedForwardResult:
    """Result of feed-forward network computation."""
    output: np.ndarray
    hidden_activations: np.ndarray
    computation_steps: List[ComputationStep]
    visualization: OperationVisualization
    properties: Dict[str, any]


@dataclass
class TransformerBlockResult:
    """Result of complete transformer block computation."""
    output: np.ndarray
    attention_result: AttentionResult
    layer_norm_results: List[LayerNormResult]
    feedforward_result: FeedForwardResult
    computation_steps: List[ComputationStep]
    visualization: OperationVisualization
    properties: Dict[str, any]


class TransformerComponents:
    """Implementation of transformer architecture components."""
    
    def __init__(self):
        self.attention_mechanisms = AttentionMechanisms()
        self.color_palette = {
            'input': '#FF6B6B',           # Red for input
            'attention': '#4ECDC4',       # Teal for attention
            'layer_norm': '#45B7D1',      # Blue for layer norm
            'feedforward': '#96CEB4',     # Green for feed-forward
            'residual': '#FFEAA7',        # Yellow for residual connections
            'output': '#DDA0DD'           # Purple for output
        }
    
    def layer_normalization(self, x: np.ndarray, gamma: Optional[np.ndarray] = None,
                          beta: Optional[np.ndarray] = None, eps: float = 1e-6) -> LayerNormResult:
        """
        Compute layer normalization: LN(x) = γ * (x - μ) / σ + β
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            gamma: Scale parameter (d_model,)
            beta: Shift parameter (d_model,)
            eps: Small constant for numerical stability
            
        Returns:
            LayerNormResult with detailed computation breakdown
        """
        if len(x.shape) == 2:
            # Add batch dimension if missing
            x = x[np.newaxis, ...]
        
        batch_size, seq_len, d_model = x.shape
        
        # Initialize parameters if not provided
        if gamma is None:
            gamma = np.ones(d_model)
        if beta is None:
            beta = np.zeros(d_model)
        
        computation_steps = []
        
        # Step 1: Compute mean along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="compute_mean",
            input_values={'x': x},
            operation_description="Compute mean along feature dimension",
            output_values={'mean': mean},
            visualization_hints={'operation': 'mean_computation'}
        )
        computation_steps.append(step_1)
        
        # Step 2: Compute variance
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="compute_variance",
            input_values={'x': x, 'mean': mean},
            operation_description="Compute variance along feature dimension",
            output_values={'variance': variance},
            visualization_hints={'operation': 'variance_computation'}
        )
        computation_steps.append(step_2)
        
        # Step 3: Normalize
        x_normalized = (x - mean) / np.sqrt(variance + eps)
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="normalize",
            input_values={'x': x, 'mean': mean, 'variance': variance, 'eps': np.array([eps])},
            operation_description="Normalize: (x - μ) / √(σ² + ε)",
            output_values={'x_normalized': x_normalized},
            visualization_hints={'operation': 'normalization'}
        )
        computation_steps.append(step_3)
        
        # Step 4: Scale and shift
        output = gamma * x_normalized + beta
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="scale_and_shift",
            input_values={'x_normalized': x_normalized, 'gamma': gamma, 'beta': beta},
            operation_description="Apply learnable scale (γ) and shift (β)",
            output_values={'output': output},
            visualization_hints={'operation': 'affine_transform'}
        )
        computation_steps.append(step_4)
        
        # Calculate properties
        properties = {
            'operation': 'layer_normalization',
            'input_shape': x.shape,
            'output_mean': np.mean(output),
            'output_std': np.std(output),
            'eps': eps,
            'parameter_count': gamma.size + beta.size
        }
        
        return LayerNormResult(
            normalized_output=output.squeeze(0) if batch_size == 1 else output,
            mean=mean.squeeze(0) if batch_size == 1 else mean,
            variance=variance.squeeze(0) if batch_size == 1 else variance,
            computation_steps=computation_steps,
            properties=properties
        )
    
    def batch_normalization(self, x: np.ndarray, gamma: Optional[np.ndarray] = None,
                          beta: Optional[np.ndarray] = None, eps: float = 1e-6) -> LayerNormResult:
        """
        Compute batch normalization for comparison with layer normalization.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            gamma: Scale parameter (d_model,)
            beta: Shift parameter (d_model,)
            eps: Small constant for numerical stability
            
        Returns:
            LayerNormResult with batch normalization computation
        """
        if len(x.shape) == 2:
            x = x[np.newaxis, ...]
        
        batch_size, seq_len, d_model = x.shape
        
        if gamma is None:
            gamma = np.ones(d_model)
        if beta is None:
            beta = np.zeros(d_model)
        
        computation_steps = []
        
        # Step 1: Compute mean across batch and sequence dimensions
        mean = np.mean(x, axis=(0, 1), keepdims=True)  # (1, 1, d_model)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="compute_batch_mean",
            input_values={'x': x},
            operation_description="Compute mean across batch and sequence dimensions",
            output_values={'mean': mean},
            visualization_hints={'operation': 'batch_mean_computation'}
        )
        computation_steps.append(step_1)
        
        # Step 2: Compute variance across batch and sequence dimensions
        variance = np.mean((x - mean) ** 2, axis=(0, 1), keepdims=True)  # (1, 1, d_model)
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="compute_batch_variance",
            input_values={'x': x, 'mean': mean},
            operation_description="Compute variance across batch and sequence dimensions",
            output_values={'variance': variance},
            visualization_hints={'operation': 'batch_variance_computation'}
        )
        computation_steps.append(step_2)
        
        # Step 3: Normalize
        x_normalized = (x - mean) / np.sqrt(variance + eps)
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="batch_normalize",
            input_values={'x': x, 'mean': mean, 'variance': variance, 'eps': np.array([eps])},
            operation_description="Batch normalize: (x - μ_batch) / √(σ²_batch + ε)",
            output_values={'x_normalized': x_normalized},
            visualization_hints={'operation': 'batch_normalization'}
        )
        computation_steps.append(step_3)
        
        # Step 4: Scale and shift
        output = gamma * x_normalized + beta
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="batch_scale_and_shift",
            input_values={'x_normalized': x_normalized, 'gamma': gamma, 'beta': beta},
            operation_description="Apply learnable scale (γ) and shift (β)",
            output_values={'output': output},
            visualization_hints={'operation': 'batch_affine_transform'}
        )
        computation_steps.append(step_4)
        
        properties = {
            'operation': 'batch_normalization',
            'input_shape': x.shape,
            'output_mean': np.mean(output),
            'output_std': np.std(output),
            'eps': eps,
            'parameter_count': gamma.size + beta.size
        }
        
        return LayerNormResult(
            normalized_output=output.squeeze(0) if batch_size == 1 else output,
            mean=mean.squeeze(0) if batch_size == 1 else mean,
            variance=variance.squeeze(0) if batch_size == 1 else variance,
            computation_steps=computation_steps,
            properties=properties
        )
    
    def feed_forward_network(self, x: np.ndarray, d_ff: int, activation: str = 'relu') -> FeedForwardResult:
        """
        Compute feed-forward network: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
        
        Args:
            x: Input tensor (seq_len, d_model)
            d_ff: Hidden dimension of feed-forward network
            activation: Activation function ('relu', 'gelu')
            
        Returns:
            FeedForwardResult with detailed computation breakdown
        """
        seq_len, d_model = x.shape
        
        # Initialize weight matrices (normally learned parameters)
        W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)  # Xavier initialization
        b1 = np.zeros(d_ff)
        W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        b2 = np.zeros(d_model)
        
        computation_steps = []
        
        # Step 1: First linear transformation
        hidden_pre_activation = x @ W1 + b1  # (seq_len, d_ff)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="first_linear",
            input_values={'x': x, 'W1': W1, 'b1': b1},
            operation_description=f"First linear layer: x @ W₁ + b₁ (expand to {d_ff} dimensions)",
            output_values={'hidden_pre_activation': hidden_pre_activation},
            visualization_hints={'operation': 'linear_expansion'}
        )
        computation_steps.append(step_1)
        
        # Step 2: Apply activation function
        if activation == 'relu':
            hidden_activations = np.maximum(0, hidden_pre_activation)
            activation_desc = "Apply ReLU activation: max(0, x)"
        elif activation == 'gelu':
            # GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            hidden_activations = 0.5 * hidden_pre_activation * (
                1 + np.tanh(np.sqrt(2/np.pi) * (hidden_pre_activation + 0.044715 * hidden_pre_activation**3))
            )
            activation_desc = "Apply GELU activation: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))"
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name=f"{activation}_activation",
            input_values={'hidden_pre_activation': hidden_pre_activation},
            operation_description=activation_desc,
            output_values={'hidden_activations': hidden_activations},
            visualization_hints={'operation': f'{activation}_activation'}
        )
        computation_steps.append(step_2)
        
        # Step 3: Second linear transformation
        output = hidden_activations @ W2 + b2  # (seq_len, d_model)
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="second_linear",
            input_values={'hidden_activations': hidden_activations, 'W2': W2, 'b2': b2},
            operation_description=f"Second linear layer: h @ W₂ + b₂ (project back to {d_model} dimensions)",
            output_values={'output': output},
            visualization_hints={'operation': 'linear_projection'}
        )
        computation_steps.append(step_3)
        
        # Create visualization
        visualization = self._create_feedforward_visualization(
            x, hidden_activations, output, W1, W2
        )
        
        # Calculate properties
        properties = {
            'operation': 'feed_forward_network',
            'input_shape': x.shape,
            'hidden_dimension': d_ff,
            'activation': activation,
            'parameter_count': W1.size + b1.size + W2.size + b2.size,
            'expansion_ratio': d_ff / d_model,
            'sparsity': np.sum(hidden_activations == 0) / hidden_activations.size if activation == 'relu' else 0
        }
        
        return FeedForwardResult(
            output=output,
            hidden_activations=hidden_activations,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def transformer_block(self, x: np.ndarray, num_heads: int, d_ff: int,
                         dropout_rate: float = 0.1) -> TransformerBlockResult:
        """
        Compute complete transformer block with self-attention and feed-forward.
        
        Architecture:
        1. Multi-head self-attention with residual connection and layer norm
        2. Feed-forward network with residual connection and layer norm
        
        Args:
            x: Input tensor (seq_len, d_model)
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout_rate: Dropout probability (for documentation, not applied)
            
        Returns:
            TransformerBlockResult with complete computation breakdown
        """
        seq_len, d_model = x.shape
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        
        computation_steps = []
        layer_norm_results = []
        
        # Step 1: Multi-head self-attention
        attention_result = self.attention_mechanisms.multi_head_attention(
            x, x, x, num_heads, d_model  # Self-attention: Q=K=V=x
        )
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="multi_head_self_attention",
            input_values={'x': x},
            operation_description=f"Multi-head self-attention with {num_heads} heads",
            output_values={'attention_output': attention_result.output},
            visualization_hints={'operation': 'self_attention', 'num_heads': num_heads}
        )
        computation_steps.append(step_1)
        
        # Step 2: First residual connection and layer normalization
        attention_residual = x + attention_result.output  # Residual connection
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="attention_residual",
            input_values={'x': x, 'attention_output': attention_result.output},
            operation_description="Add residual connection: x + attention(x)",
            output_values={'attention_residual': attention_residual},
            visualization_hints={'operation': 'residual_connection'}
        )
        computation_steps.append(step_2)
        
        # Layer normalization after attention
        ln1_result = self.layer_normalization(attention_residual)
        layer_norm_results.append(ln1_result)
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="layer_norm_1",
            input_values={'attention_residual': attention_residual},
            operation_description="Apply layer normalization after attention",
            output_values={'ln1_output': ln1_result.normalized_output},
            visualization_hints={'operation': 'layer_normalization'}
        )
        computation_steps.append(step_3)
        
        # Step 4: Feed-forward network
        ff_result = self.feed_forward_network(ln1_result.normalized_output, d_ff)
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="feed_forward",
            input_values={'ln1_output': ln1_result.normalized_output},
            operation_description=f"Feed-forward network with hidden dimension {d_ff}",
            output_values={'ff_output': ff_result.output},
            visualization_hints={'operation': 'feed_forward', 'd_ff': d_ff}
        )
        computation_steps.append(step_4)
        
        # Step 5: Second residual connection and layer normalization
        ff_residual = ln1_result.normalized_output + ff_result.output
        
        step_5 = ComputationStep(
            step_number=5,
            operation_name="ff_residual",
            input_values={'ln1_output': ln1_result.normalized_output, 'ff_output': ff_result.output},
            operation_description="Add residual connection: ln1_output + ff(ln1_output)",
            output_values={'ff_residual': ff_residual},
            visualization_hints={'operation': 'residual_connection'}
        )
        computation_steps.append(step_5)
        
        # Final layer normalization
        ln2_result = self.layer_normalization(ff_residual)
        layer_norm_results.append(ln2_result)
        
        step_6 = ComputationStep(
            step_number=6,
            operation_name="layer_norm_2",
            input_values={'ff_residual': ff_residual},
            operation_description="Apply layer normalization after feed-forward",
            output_values={'final_output': ln2_result.normalized_output},
            visualization_hints={'operation': 'layer_normalization'}
        )
        computation_steps.append(step_6)
        
        # Create visualization
        visualization = self._create_transformer_block_visualization(
            x, attention_result.output, ln1_result.normalized_output,
            ff_result.output, ln2_result.normalized_output
        )
        
        # Calculate properties
        properties = {
            'operation': 'transformer_block',
            'input_shape': x.shape,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'dropout_rate': dropout_rate,
            'total_parameters': (
                4 * d_model * d_model +  # Attention projections
                2 * d_model +           # Layer norm 1 parameters
                d_model * d_ff + d_ff + # FF layer 1
                d_ff * d_model + d_model + # FF layer 2
                2 * d_model             # Layer norm 2 parameters
            ),
            'attention_heads_info': attention_result.properties,
            'feedforward_info': ff_result.properties
        }
        
        return TransformerBlockResult(
            output=ln2_result.normalized_output,
            attention_result=attention_result,
            layer_norm_results=layer_norm_results,
            feedforward_result=ff_result,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _create_feedforward_visualization(self, x: np.ndarray, hidden: np.ndarray,
                                        output: np.ndarray, W1: np.ndarray,
                                        W2: np.ndarray) -> OperationVisualization:
        """Create visualization for feed-forward network."""
        
        input_colored = ColorCodedMatrix(
            matrix_data=x,
            color_mapping={'default': self.color_palette['input']},
            element_labels={},
            highlight_patterns=[]
        )
        
        hidden_colored = ColorCodedMatrix(
            matrix_data=hidden,
            color_mapping={'default': self.color_palette['feedforward']},
            element_labels={},
            highlight_patterns=[]
        )
        
        output_colored = ColorCodedMatrix(
            matrix_data=output,
            color_mapping={'default': self.color_palette['output']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="feed_forward_network",
            input_matrices=[input_colored],
            intermediate_steps=[hidden_colored],
            output_matrix=output_colored,
            animation_sequence=[]
        )
    
    def _create_transformer_block_visualization(self, x: np.ndarray, attention_out: np.ndarray,
                                              ln1_out: np.ndarray, ff_out: np.ndarray,
                                              final_out: np.ndarray) -> OperationVisualization:
        """Create visualization for complete transformer block."""
        
        input_matrices = [
            ColorCodedMatrix(x, {'default': self.color_palette['input']}, {}, [])
        ]
        
        intermediate_steps = [
            ColorCodedMatrix(attention_out, {'default': self.color_palette['attention']}, {}, []),
            ColorCodedMatrix(ln1_out, {'default': self.color_palette['layer_norm']}, {}, []),
            ColorCodedMatrix(ff_out, {'default': self.color_palette['feedforward']}, {}, [])
        ]
        
        output_colored = ColorCodedMatrix(
            matrix_data=final_out,
            color_mapping={'default': self.color_palette['output']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="transformer_block",
            input_matrices=input_matrices,
            intermediate_steps=intermediate_steps,
            output_matrix=output_colored,
            animation_sequence=[]
        )