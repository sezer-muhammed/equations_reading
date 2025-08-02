"""
Operation visualization components for step-by-step mathematical breakdowns.
Provides detailed visual representations of matrix operations, attention mechanisms, and gradient flow.
"""

from .matrix_operations import (
    MatrixOperationVisualizer,
    OperationStep,
    create_sample_matrix_multiplication,
    create_sample_attention_visualization
)

__all__ = [
    'MatrixOperationVisualizer',
    'OperationStep', 
    'create_sample_matrix_multiplication',
    'create_sample_attention_visualization'
]