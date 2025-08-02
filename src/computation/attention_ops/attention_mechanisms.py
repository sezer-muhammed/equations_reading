"""
Attention mechanism calculations with detailed mathematical breakdowns.
Implements scaled dot-product attention and multi-head attention.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import (
    ColorCodedMatrix, OperationVisualization, HighlightPattern,
    AnimationFrame, ComputationStep, VisualizationData
)
from ..matrix_ops.core_operations import MatrixOperationResult


@dataclass
class AttentionResult:
    """Result of attention computation with detailed breakdown."""
    attention_weights: np.ndarray
    output: np.ndarray
    computation_steps: List[ComputationStep]
    visualization: OperationVisualization
    properties: Dict[str, any]


class AttentionMechanisms:
    """Implementation of attention mechanisms with visualization."""
    
    def __init__(self):
        self.color_palette = {
            'query': '#FF6B6B',        # Red for queries
            'key': '#4ECDC4',          # Teal for keys
            'value': '#45B7D1',        # Blue for values
            'attention': '#96CEB4',     # Green for attention weights
            'output': '#FFEAA7',       # Yellow for output
            'softmax': '#DDA0DD'       # Purple for softmax
        }
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, 
                                   V: np.ndarray, mask: Optional[np.ndarray] = None,
                                   temperature: Optional[float] = None) -> AttentionResult:
        """
        Compute scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        Args:
            Q: Query matrix (seq_len_q, d_k)
            K: Key matrix (seq_len_k, d_k) 
            V: Value matrix (seq_len_v, d_v)
            mask: Optional attention mask
            temperature: Optional temperature scaling (default: √d_k)
            
        Returns:
            AttentionResult with detailed computation breakdown
        """
        seq_len_q, d_k = Q.shape
        seq_len_k, d_k2 = K.shape
        seq_len_v, d_v = V.shape
        
        if d_k != d_k2:
            raise ValueError(f"Query and Key dimensions must match: {d_k} vs {d_k2}")
        if seq_len_k != seq_len_v:
            raise ValueError(f"Key and Value sequence lengths must match: {seq_len_k} vs {seq_len_v}")
        
        # Set temperature (scaling factor)
        if temperature is None:
            temperature = math.sqrt(d_k)
        
        computation_steps = []
        
        # Step 1: Compute Q @ K^T
        step_1_desc = f"Compute attention scores: Q @ K^T"
        K_T = K.T
        scores = Q @ K_T  # (seq_len_q, seq_len_k)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="query_key_product",
            input_values={'Q': Q, 'K': K, 'K_T': K_T},
            operation_description=step_1_desc,
            output_values={'scores': scores},
            visualization_hints={
                'operation': 'matrix_multiply',
                'highlight_interaction': True,
                'temperature': temperature
            }
        )
        computation_steps.append(step_1)
        
        # Step 2: Scale by temperature
        step_2_desc = f"Scale by √d_k = {temperature:.3f}"
        scaled_scores = scores / temperature
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="temperature_scaling",
            input_values={'scores': scores, 'temperature': np.array([temperature])},
            operation_description=step_2_desc,
            output_values={'scaled_scores': scaled_scores},
            visualization_hints={'operation': 'elementwise_divide'}
        )
        computation_steps.append(step_2)
        
        # Step 3: Apply mask if provided
        if mask is not None:
            masked_scores = scaled_scores + mask
            step_3 = ComputationStep(
                step_number=3,
                operation_name="apply_mask",
                input_values={'scaled_scores': scaled_scores, 'mask': mask},
                operation_description="Apply attention mask (add large negative values)",
                output_values={'masked_scores': masked_scores},
                visualization_hints={'operation': 'mask_application'}
            )
            computation_steps.append(step_3)
            scaled_scores = masked_scores
        
        # Step 4: Apply softmax
        step_num = 4 if mask is not None else 3
        step_4_desc = "Apply softmax to get attention weights"
        
        # Compute softmax with numerical stability
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        step_4 = ComputationStep(
            step_number=step_num,
            operation_name="softmax",
            input_values={'scaled_scores': scaled_scores},
            operation_description=step_4_desc,
            output_values={
                'exp_scores': exp_scores,
                'attention_weights': attention_weights
            },
            visualization_hints={
                'operation': 'softmax',
                'show_normalization': True
            }
        )
        computation_steps.append(step_4)
        
        # Step 5: Compute weighted sum with values
        step_5_desc = "Compute output: attention_weights @ V"
        output = attention_weights @ V  # (seq_len_q, d_v)
        
        step_5 = ComputationStep(
            step_number=step_num + 1,
            operation_name="attention_value_product",
            input_values={'attention_weights': attention_weights, 'V': V},
            operation_description=step_5_desc,
            output_values={'output': output},
            visualization_hints={
                'operation': 'weighted_sum',
                'show_attention_flow': True
            }
        )
        computation_steps.append(step_5)
        
        # Create visualization
        visualization = self._create_attention_visualization(
            Q, K, V, attention_weights, output, computation_steps
        )
        
        # Calculate properties
        properties = {
            'operation': 'scaled_dot_product_attention',
            'sequence_lengths': {'query': seq_len_q, 'key': seq_len_k, 'value': seq_len_v},
            'dimensions': {'d_k': d_k, 'd_v': d_v},
            'temperature': temperature,
            'attention_entropy': -np.sum(attention_weights * np.log(attention_weights + 1e-9)),
            'max_attention_weight': np.max(attention_weights),
            'attention_sparsity': np.sum(attention_weights < 0.01) / attention_weights.size
        }
        
        return AttentionResult(
            attention_weights=attention_weights,
            output=output,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def multi_head_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                           num_heads: int, d_model: int) -> AttentionResult:
        """
        Compute multi-head attention with head separation visualization.
        
        Args:
            Q, K, V: Input matrices (seq_len, d_model)
            num_heads: Number of attention heads
            d_model: Model dimension
            
        Returns:
            AttentionResult with multi-head breakdown
        """
        seq_len, d_model_input = Q.shape
        
        if d_model_input != d_model:
            raise ValueError(f"Input dimension {d_model_input} doesn't match d_model {d_model}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        
        d_k = d_model // num_heads
        computation_steps = []
        
        # Initialize projection matrices (normally learned parameters)
        W_Q = np.random.randn(d_model, d_model) * 0.1
        W_K = np.random.randn(d_model, d_model) * 0.1  
        W_V = np.random.randn(d_model, d_model) * 0.1
        W_O = np.random.randn(d_model, d_model) * 0.1
        
        # Step 1: Linear projections
        Q_proj = Q @ W_Q
        K_proj = K @ W_K
        V_proj = V @ W_V
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="linear_projections",
            input_values={'Q': Q, 'K': K, 'V': V, 'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V},
            operation_description="Apply linear projections to Q, K, V",
            output_values={'Q_proj': Q_proj, 'K_proj': K_proj, 'V_proj': V_proj},
            visualization_hints={'operation': 'linear_projection'}
        )
        computation_steps.append(step_1)
        
        # Step 2: Reshape for multi-head
        Q_heads = Q_proj.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)  # (num_heads, seq_len, d_k)
        K_heads = K_proj.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
        V_heads = V_proj.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="reshape_for_heads",
            input_values={'Q_proj': Q_proj, 'K_proj': K_proj, 'V_proj': V_proj},
            operation_description=f"Reshape and split into {num_heads} heads of dimension {d_k}",
            output_values={'Q_heads': Q_heads, 'K_heads': K_heads, 'V_heads': V_heads},
            visualization_hints={'operation': 'head_split', 'num_heads': num_heads}
        )
        computation_steps.append(step_2)
        
        # Step 3: Apply attention to each head
        head_outputs = []
        head_attentions = []
        
        for head_idx in range(num_heads):
            head_result = self.scaled_dot_product_attention(
                Q_heads[head_idx], K_heads[head_idx], V_heads[head_idx]
            )
            head_outputs.append(head_result.output)
            head_attentions.append(head_result.attention_weights)
        
        concatenated_heads = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="head_attention",
            input_values={'Q_heads': Q_heads, 'K_heads': K_heads, 'V_heads': V_heads},
            operation_description=f"Apply scaled dot-product attention to each of {num_heads} heads",
            output_values={'head_outputs': np.array(head_outputs), 'concatenated': concatenated_heads},
            visualization_hints={'operation': 'multi_head_attention', 'head_attentions': head_attentions}
        )
        computation_steps.append(step_3)
        
        # Step 4: Final linear projection
        final_output = concatenated_heads @ W_O
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="output_projection",
            input_values={'concatenated_heads': concatenated_heads, 'W_O': W_O},
            operation_description="Apply final linear projection",
            output_values={'final_output': final_output},
            visualization_hints={'operation': 'linear_projection'}
        )
        computation_steps.append(step_4)
        
        # Create visualization
        visualization = self._create_multihead_visualization(
            Q, K, V, head_attentions, final_output, num_heads
        )
        
        # Calculate properties
        properties = {
            'operation': 'multi_head_attention',
            'num_heads': num_heads,
            'd_k': d_k,
            'd_model': d_model,
            'sequence_length': seq_len,
            'total_parameters': W_Q.size + W_K.size + W_V.size + W_O.size,
            'head_attention_patterns': [np.max(att) for att in head_attentions]
        }
        
        return AttentionResult(
            attention_weights=np.array(head_attentions),
            output=final_output,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _create_attention_visualization(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                      attention_weights: np.ndarray, output: np.ndarray,
                                      computation_steps: List[ComputationStep]) -> OperationVisualization:
        """Create visualization for attention mechanism."""
        
        # Create color-coded matrices
        query_colored = ColorCodedMatrix(
            matrix_data=Q,
            color_mapping={'default': self.color_palette['query']},
            element_labels={},
            highlight_patterns=[]
        )
        
        key_colored = ColorCodedMatrix(
            matrix_data=K,
            color_mapping={'default': self.color_palette['key']},
            element_labels={},
            highlight_patterns=[]
        )
        
        value_colored = ColorCodedMatrix(
            matrix_data=V,
            color_mapping={'default': self.color_palette['value']},
            element_labels={},
            highlight_patterns=[]
        )
        
        attention_colored = ColorCodedMatrix(
            matrix_data=attention_weights,
            color_mapping={'default': self.color_palette['attention']},
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
            operation_type="scaled_dot_product_attention",
            input_matrices=[query_colored, key_colored, value_colored],
            intermediate_steps=[attention_colored],
            output_matrix=output_colored,
            animation_sequence=[]
        )
    
    def _create_multihead_visualization(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                      head_attentions: List[np.ndarray], output: np.ndarray,
                                      num_heads: int) -> OperationVisualization:
        """Create visualization for multi-head attention."""
        
        input_matrices = [
            ColorCodedMatrix(Q, {'default': self.color_palette['query']}, {}, []),
            ColorCodedMatrix(K, {'default': self.color_palette['key']}, {}, []),
            ColorCodedMatrix(V, {'default': self.color_palette['value']}, {}, [])
        ]
        
        # Create intermediate steps for each head
        intermediate_steps = []
        for i, head_att in enumerate(head_attentions):
            head_colored = ColorCodedMatrix(
                matrix_data=head_att,
                color_mapping={'default': f'head_{i}'},
                element_labels={},
                highlight_patterns=[]
            )
            intermediate_steps.append(head_colored)
        
        output_colored = ColorCodedMatrix(
            matrix_data=output,
            color_mapping={'default': self.color_palette['output']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="multi_head_attention",
            input_matrices=input_matrices,
            intermediate_steps=intermediate_steps,
            output_matrix=output_colored,
            animation_sequence=[]
        )