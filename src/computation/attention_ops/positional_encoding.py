"""
Positional encoding implementations for attention mechanisms.
Includes absolute, relative, and rotary positional embeddings (RoPE).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ...core.models import ComputationStep, OperationVisualization, ColorCodedMatrix


@dataclass
class PositionalEncodingResult:
    """Result of positional encoding computation."""
    encoded_positions: np.ndarray
    computation_steps: List[ComputationStep]
    visualization: OperationVisualization
    properties: Dict[str, any]


class PositionalEncodings:
    """Implementation of various positional encoding schemes."""
    
    def __init__(self):
        self.color_palette = {
            'sin': '#FF6B6B',      # Red for sine components
            'cos': '#4ECDC4',      # Teal for cosine components
            'position': '#45B7D1', # Blue for position indices
            'encoding': '#96CEB4', # Green for final encoding
            'frequency': '#FFEAA7' # Yellow for frequency components
        }
    
    def absolute_positional_encoding(self, seq_len: int, d_model: int) -> PositionalEncodingResult:
        """
        Compute absolute sinusoidal positional encoding.
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
            
        Returns:
            PositionalEncodingResult with detailed computation
        """
        computation_steps = []
        
        # Step 1: Create position indices
        positions = np.arange(seq_len).reshape(-1, 1)  # (seq_len, 1)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="create_positions",
            input_values={'seq_len': np.array([seq_len])},
            operation_description=f"Create position indices [0, 1, ..., {seq_len-1}]",
            output_values={'positions': positions},
            visualization_hints={'operation': 'position_creation'}
        )
        computation_steps.append(step_1)
        
        # Step 2: Create dimension indices
        dim_indices = np.arange(0, d_model, 2)  # Even dimensions: 0, 2, 4, ...
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="create_dimensions",
            input_values={'d_model': np.array([d_model])},
            operation_description=f"Create dimension indices [0, 2, 4, ..., {d_model-2}]",
            output_values={'dim_indices': dim_indices},
            visualization_hints={'operation': 'dimension_creation'}
        )
        computation_steps.append(step_2)
        
        # Step 3: Compute frequency denominators
        denominators = 10000 ** (dim_indices / d_model)  # (d_model//2,)
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="compute_frequencies",
            input_values={'dim_indices': dim_indices, 'd_model': np.array([d_model])},
            operation_description="Compute frequency denominators: 10000^(2i/d_model)",
            output_values={'denominators': denominators},
            visualization_hints={'operation': 'frequency_computation'}
        )
        computation_steps.append(step_3)
        
        # Step 4: Compute angles
        angles = positions / denominators  # Broadcasting: (seq_len, 1) / (d_model//2,) -> (seq_len, d_model//2)
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="compute_angles",
            input_values={'positions': positions, 'denominators': denominators},
            operation_description="Compute angles: pos / 10000^(2i/d_model)",
            output_values={'angles': angles},
            visualization_hints={'operation': 'angle_computation'}
        )
        computation_steps.append(step_4)
        
        # Step 5: Apply sine and cosine
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angles)  # Even positions get sine
        pe[:, 1::2] = np.cos(angles)  # Odd positions get cosine
        
        step_5 = ComputationStep(
            step_number=5,
            operation_name="apply_sincos",
            input_values={'angles': angles},
            operation_description="Apply sin to even dims, cos to odd dims",
            output_values={'positional_encoding': pe},
            visualization_hints={'operation': 'sincos_application'}
        )
        computation_steps.append(step_5)
        
        # Create visualization
        visualization = self._create_positional_visualization(
            positions, angles, pe, computation_steps
        )
        
        # Calculate properties
        properties = {
            'encoding_type': 'absolute_sinusoidal',
            'sequence_length': seq_len,
            'd_model': d_model,
            'max_frequency': np.max(1.0 / denominators),
            'min_frequency': np.min(1.0 / denominators),
            'encoding_range': (np.min(pe), np.max(pe))
        }
        
        return PositionalEncodingResult(
            encoded_positions=pe,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def rotary_positional_embedding(self, x: np.ndarray, seq_len: int, 
                                  d_head: int, base: float = 10000.0) -> PositionalEncodingResult:
        """
        Apply Rotary Position Embedding (RoPE) to input tensor.
        RoPE rotates query and key representations by position-dependent angles.
        
        Args:
            x: Input tensor (seq_len, d_head)
            seq_len: Sequence length
            d_head: Head dimension (must be even)
            base: Base for frequency computation
            
        Returns:
            PositionalEncodingResult with RoPE applied
        """
        if d_head % 2 != 0:
            raise ValueError(f"Head dimension must be even for RoPE, got {d_head}")
        
        computation_steps = []
        
        # Step 1: Create position indices
        positions = np.arange(seq_len)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="create_positions",
            input_values={'seq_len': np.array([seq_len])},
            operation_description="Create position indices for RoPE",
            output_values={'positions': positions},
            visualization_hints={'operation': 'position_creation'}
        )
        computation_steps.append(step_1)
        
        # Step 2: Compute frequency bands
        dim_pairs = np.arange(0, d_head, 2)  # 0, 2, 4, ..., d_head-2
        freqs = 1.0 / (base ** (dim_pairs / d_head))  # (d_head//2,)
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="compute_frequencies",
            input_values={'dim_pairs': dim_pairs, 'd_head': np.array([d_head]), 'base': np.array([base])},
            operation_description=f"Compute RoPE frequencies: 1 / {base}^(2i/d_head)",
            output_values={'frequencies': freqs},
            visualization_hints={'operation': 'rope_frequency_computation'}
        )
        computation_steps.append(step_2)
        
        # Step 3: Compute rotation angles
        angles = np.outer(positions, freqs)  # (seq_len, d_head//2)
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="compute_angles",
            input_values={'positions': positions, 'frequencies': freqs},
            operation_description="Compute rotation angles: pos * freq",
            output_values={'angles': angles},
            visualization_hints={'operation': 'angle_computation'}
        )
        computation_steps.append(step_3)
        
        # Step 4: Create rotation matrices (cos and sin components)
        cos_angles = np.cos(angles)  # (seq_len, d_head//2)
        sin_angles = np.sin(angles)  # (seq_len, d_head//2)
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="compute_rotation_components",
            input_values={'angles': angles},
            operation_description="Compute cos and sin components for rotation",
            output_values={'cos_angles': cos_angles, 'sin_angles': sin_angles},
            visualization_hints={'operation': 'rotation_components'}
        )
        computation_steps.append(step_4)
        
        # Step 5: Apply RoPE rotation
        # Split x into pairs: [x0, x1, x2, x3, ...] -> [(x0,x1), (x2,x3), ...]
        x_pairs = x.reshape(seq_len, d_head // 2, 2)  # (seq_len, d_head//2, 2)
        
        # Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        x_rotated = np.zeros_like(x_pairs)
        x_rotated[:, :, 0] = x_pairs[:, :, 0] * cos_angles - x_pairs[:, :, 1] * sin_angles
        x_rotated[:, :, 1] = x_pairs[:, :, 0] * sin_angles + x_pairs[:, :, 1] * cos_angles
        
        # Reshape back to original shape
        x_rope = x_rotated.reshape(seq_len, d_head)
        
        step_5 = ComputationStep(
            step_number=5,
            operation_name="apply_rope_rotation",
            input_values={'x': x, 'cos_angles': cos_angles, 'sin_angles': sin_angles},
            operation_description="Apply RoPE rotation to input tensor",
            output_values={'x_rope': x_rope},
            visualization_hints={'operation': 'rope_application'}
        )
        computation_steps.append(step_5)
        
        # Create visualization
        visualization = self._create_rope_visualization(
            x, angles, cos_angles, sin_angles, x_rope
        )
        
        # Calculate properties
        properties = {
            'encoding_type': 'rotary_positional_embedding',
            'sequence_length': seq_len,
            'd_head': d_head,
            'base': base,
            'frequency_range': (np.min(freqs), np.max(freqs)),
            'rotation_magnitude': np.mean(np.abs(angles))
        }
        
        return PositionalEncodingResult(
            encoded_positions=x_rope,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def relative_positional_encoding(self, seq_len: int, d_model: int, 
                                   max_relative_position: int = 128) -> PositionalEncodingResult:
        """
        Compute relative positional encoding matrix.
        Creates learnable embeddings for relative distances between positions.
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
            max_relative_position: Maximum relative distance to encode
            
        Returns:
            PositionalEncodingResult with relative position matrix
        """
        computation_steps = []
        
        # Step 1: Create relative position matrix
        positions = np.arange(seq_len)
        relative_positions = positions[:, None] - positions[None, :]  # (seq_len, seq_len)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="compute_relative_positions",
            input_values={'positions': positions},
            operation_description="Compute relative position matrix: pos_i - pos_j",
            output_values={'relative_positions': relative_positions},
            visualization_hints={'operation': 'relative_position_matrix'}
        )
        computation_steps.append(step_1)
        
        # Step 2: Clip relative positions to maximum range
        clipped_positions = np.clip(
            relative_positions, 
            -max_relative_position, 
            max_relative_position
        )
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="clip_positions",
            input_values={'relative_positions': relative_positions, 
                         'max_relative_position': np.array([max_relative_position])},
            operation_description=f"Clip relative positions to [{-max_relative_position}, {max_relative_position}]",
            output_values={'clipped_positions': clipped_positions},
            visualization_hints={'operation': 'position_clipping'}
        )
        computation_steps.append(step_2)
        
        # Step 3: Convert to embedding indices (shift to make non-negative)
        embedding_indices = clipped_positions + max_relative_position  # [0, 2*max_relative_position]
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="create_embedding_indices",
            input_values={'clipped_positions': clipped_positions, 
                         'max_relative_position': np.array([max_relative_position])},
            operation_description="Shift positions to create embedding indices",
            output_values={'embedding_indices': embedding_indices},
            visualization_hints={'operation': 'index_creation'}
        )
        computation_steps.append(step_3)
        
        # Step 4: Create learnable embedding matrix (normally learned parameters)
        vocab_size = 2 * max_relative_position + 1
        embedding_matrix = np.random.randn(vocab_size, d_model) * 0.1
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="create_embedding_matrix",
            input_values={'vocab_size': np.array([vocab_size]), 'd_model': np.array([d_model])},
            operation_description=f"Create learnable embedding matrix ({vocab_size}, {d_model})",
            output_values={'embedding_matrix': embedding_matrix},
            visualization_hints={'operation': 'embedding_creation'}
        )
        computation_steps.append(step_4)
        
        # Step 5: Look up embeddings
        relative_embeddings = embedding_matrix[embedding_indices]  # (seq_len, seq_len, d_model)
        
        step_5 = ComputationStep(
            step_number=5,
            operation_name="lookup_embeddings",
            input_values={'embedding_indices': embedding_indices, 'embedding_matrix': embedding_matrix},
            operation_description="Look up relative position embeddings",
            output_values={'relative_embeddings': relative_embeddings},
            visualization_hints={'operation': 'embedding_lookup'}
        )
        computation_steps.append(step_5)
        
        # Create visualization
        visualization = self._create_relative_visualization(
            relative_positions, embedding_indices, relative_embeddings
        )
        
        # Calculate properties
        properties = {
            'encoding_type': 'relative_positional',
            'sequence_length': seq_len,
            'd_model': d_model,
            'max_relative_position': max_relative_position,
            'vocab_size': vocab_size,
            'embedding_parameters': embedding_matrix.size
        }
        
        return PositionalEncodingResult(
            encoded_positions=relative_embeddings,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _create_positional_visualization(self, positions: np.ndarray, angles: np.ndarray,
                                       pe: np.ndarray, computation_steps: List[ComputationStep]) -> OperationVisualization:
        """Create visualization for absolute positional encoding."""
        
        position_colored = ColorCodedMatrix(
            matrix_data=positions,
            color_mapping={'default': self.color_palette['position']},
            element_labels={},
            highlight_patterns=[]
        )
        
        angle_colored = ColorCodedMatrix(
            matrix_data=angles,
            color_mapping={'default': self.color_palette['frequency']},
            element_labels={},
            highlight_patterns=[]
        )
        
        pe_colored = ColorCodedMatrix(
            matrix_data=pe,
            color_mapping={'default': self.color_palette['encoding']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="absolute_positional_encoding",
            input_matrices=[position_colored],
            intermediate_steps=[angle_colored],
            output_matrix=pe_colored,
            animation_sequence=[]
        )
    
    def _create_rope_visualization(self, x: np.ndarray, angles: np.ndarray,
                                 cos_angles: np.ndarray, sin_angles: np.ndarray,
                                 x_rope: np.ndarray) -> OperationVisualization:
        """Create visualization for RoPE."""
        
        input_colored = ColorCodedMatrix(
            matrix_data=x,
            color_mapping={'default': self.color_palette['position']},
            element_labels={},
            highlight_patterns=[]
        )
        
        cos_colored = ColorCodedMatrix(
            matrix_data=cos_angles,
            color_mapping={'default': self.color_palette['cos']},
            element_labels={},
            highlight_patterns=[]
        )
        
        sin_colored = ColorCodedMatrix(
            matrix_data=sin_angles,
            color_mapping={'default': self.color_palette['sin']},
            element_labels={},
            highlight_patterns=[]
        )
        
        output_colored = ColorCodedMatrix(
            matrix_data=x_rope,
            color_mapping={'default': self.color_palette['encoding']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="rotary_positional_embedding",
            input_matrices=[input_colored],
            intermediate_steps=[cos_colored, sin_colored],
            output_matrix=output_colored,
            animation_sequence=[]
        )
    
    def _create_relative_visualization(self, relative_positions: np.ndarray,
                                     embedding_indices: np.ndarray,
                                     relative_embeddings: np.ndarray) -> OperationVisualization:
        """Create visualization for relative positional encoding."""
        
        rel_pos_colored = ColorCodedMatrix(
            matrix_data=relative_positions,
            color_mapping={'default': self.color_palette['position']},
            element_labels={},
            highlight_patterns=[]
        )
        
        indices_colored = ColorCodedMatrix(
            matrix_data=embedding_indices,
            color_mapping={'default': self.color_palette['frequency']},
            element_labels={},
            highlight_patterns=[]
        )
        
        # For visualization, show a 2D slice of the 3D embedding tensor
        embeddings_2d = relative_embeddings[:, :, 0]  # Take first dimension
        embeddings_colored = ColorCodedMatrix(
            matrix_data=embeddings_2d,
            color_mapping={'default': self.color_palette['encoding']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="relative_positional_encoding",
            input_matrices=[rel_pos_colored],
            intermediate_steps=[indices_colored],
            output_matrix=embeddings_colored,
            animation_sequence=[]
        )