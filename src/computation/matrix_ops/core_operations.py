"""
Core matrix operations with visualization data extraction.
Implements fundamental linear algebra operations needed for AI mathematics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...core.models import (
    ColorCodedMatrix, OperationVisualization, HighlightPattern,
    AnimationFrame, ComputationStep, VisualizationData
)


@dataclass
class MatrixOperationResult:
    """Result of a matrix operation with visualization data."""
    result: np.ndarray
    computation_steps: List[ComputationStep]
    visualization: OperationVisualization
    properties: Dict[str, any]


class CoreMatrixOperations:
    """Core matrix operations with visualization data extraction."""
    
    def __init__(self):
        self.color_palette = {
            'input_a': '#FF6B6B',      # Red for first input
            'input_b': '#4ECDC4',      # Teal for second input  
            'output': '#45B7D1',       # Blue for output
            'intermediate': '#96CEB4',  # Green for intermediate
            'highlight': '#FFEAA7'     # Yellow for highlights
        }
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray, 
                       step_by_step: bool = True) -> MatrixOperationResult:
        """
        Perform matrix multiplication with detailed visualization data.
        
        Args:
            A: First matrix (m x n)
            B: Second matrix (n x p)
            step_by_step: Whether to generate step-by-step visualization
            
        Returns:
            MatrixOperationResult with computation steps and visualization
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape} x {B.shape}")
        
        m, n = A.shape
        n2, p = B.shape
        result = np.zeros((m, p))
        
        computation_steps = []
        animation_frames = []
        
        # Create color-coded input matrices
        input_a_colored = ColorCodedMatrix(
            matrix_data=A,
            color_mapping={'default': self.color_palette['input_a']},
            element_labels={},
            highlight_patterns=[]
        )
        
        input_b_colored = ColorCodedMatrix(
            matrix_data=B,
            color_mapping={'default': self.color_palette['input_b']},
            element_labels={},
            highlight_patterns=[]
        )
        
        # Perform multiplication with step tracking
        for i in range(m):
            for j in range(p):
                # Calculate dot product for position (i,j)
                dot_product = 0
                intermediate_values = {}
                
                for k in range(n):
                    product = A[i, k] * B[k, j]
                    dot_product += product
                    intermediate_values[f'a_{i}{k}_b_{k}{j}'] = product
                
                result[i, j] = dot_product
                
                if step_by_step:
                    # Create computation step
                    step = ComputationStep(
                        step_number=i * p + j + 1,
                        operation_name=f"dot_product_{i}_{j}",
                        input_values={
                            'row_a': A[i, :],
                            'col_b': B[:, j]
                        },
                        operation_description=f"Computing element ({i},{j}): row {i} of A · column {j} of B",
                        output_values={'result': np.array([dot_product])},
                        visualization_hints={
                            'highlight_row': i,
                            'highlight_col': j,
                            'intermediate_products': intermediate_values
                        }
                    )
                    computation_steps.append(step)
                    
                    # Create animation frame
                    current_result = result.copy()
                    current_result[i+1:, :] = np.nan  # Hide future results
                    current_result[:, j+1:] = np.nan
                    
                    highlights = [
                        HighlightPattern(
                            pattern_type="row",
                            indices=[(i, k) for k in range(n)],
                            color=self.color_palette['highlight'],
                            description=f"Row {i} of matrix A"
                        ),
                        HighlightPattern(
                            pattern_type="column", 
                            indices=[(k, j) for k in range(n)],
                            color=self.color_palette['highlight'],
                            description=f"Column {j} of matrix B"
                        )
                    ]
                    
                    frame = AnimationFrame(
                        frame_number=len(animation_frames),
                        matrix_state=current_result,
                        highlights=highlights,
                        description=f"Computing element ({i},{j})",
                        duration_ms=1500
                    )
                    animation_frames.append(frame)
        
        # Create output matrix visualization
        output_colored = ColorCodedMatrix(
            matrix_data=result,
            color_mapping={'default': self.color_palette['output']},
            element_labels={},
            highlight_patterns=[]
        )
        
        # Create operation visualization
        visualization = OperationVisualization(
            operation_type="matrix_multiply",
            input_matrices=[input_a_colored, input_b_colored],
            intermediate_steps=[],
            output_matrix=output_colored,
            animation_sequence=animation_frames
        )
        
        # Calculate properties
        properties = {
            'operation': 'matrix_multiplication',
            'input_shapes': [A.shape, B.shape],
            'output_shape': result.shape,
            'total_operations': m * n * p,
            'computational_complexity': f"O({m} × {n} × {p})"
        }
        
        return MatrixOperationResult(
            result=result,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def matrix_transpose(self, A: np.ndarray) -> MatrixOperationResult:
        """Compute matrix transpose with visualization."""
        result = A.T
        
        # Create visualization showing the transpose operation
        input_colored = ColorCodedMatrix(
            matrix_data=A,
            color_mapping={'default': self.color_palette['input_a']},
            element_labels={},
            highlight_patterns=[]
        )
        
        output_colored = ColorCodedMatrix(
            matrix_data=result,
            color_mapping={'default': self.color_palette['output']},
            element_labels={},
            highlight_patterns=[]
        )
        
        # Single computation step for transpose
        step = ComputationStep(
            step_number=1,
            operation_name="transpose",
            input_values={'matrix': A},
            operation_description="Transpose: swap rows and columns",
            output_values={'result': result},
            visualization_hints={'operation': 'transpose'}
        )
        
        visualization = OperationVisualization(
            operation_type="transpose",
            input_matrices=[input_colored],
            intermediate_steps=[],
            output_matrix=output_colored,
            animation_sequence=[]
        )
        
        properties = {
            'operation': 'transpose',
            'input_shape': A.shape,
            'output_shape': result.shape,
            'preserves_determinant': A.shape[0] == A.shape[1]
        }
        
        return MatrixOperationResult(
            result=result,
            computation_steps=[step],
            visualization=visualization,
            properties=properties
        )
    
    def matrix_elementwise_ops(self, A: np.ndarray, B: np.ndarray, 
                              operation: str) -> MatrixOperationResult:
        """Perform element-wise operations (add, subtract, multiply, divide)."""
        if A.shape != B.shape:
            raise ValueError(f"Matrix shapes must match: {A.shape} vs {B.shape}")
        
        ops = {
            'add': np.add,
            'subtract': np.subtract, 
            'multiply': np.multiply,
            'divide': np.divide
        }
        
        if operation not in ops:
            raise ValueError(f"Unknown operation: {operation}")
        
        result = ops[operation](A, B)
        
        # Create computation step
        step = ComputationStep(
            step_number=1,
            operation_name=f"elementwise_{operation}",
            input_values={'matrix_a': A, 'matrix_b': B},
            operation_description=f"Element-wise {operation}",
            output_values={'result': result},
            visualization_hints={'operation': f'elementwise_{operation}'}
        )
        
        # Create visualizations
        input_a_colored = ColorCodedMatrix(
            matrix_data=A,
            color_mapping={'default': self.color_palette['input_a']},
            element_labels={},
            highlight_patterns=[]
        )
        
        input_b_colored = ColorCodedMatrix(
            matrix_data=B,
            color_mapping={'default': self.color_palette['input_b']},
            element_labels={},
            highlight_patterns=[]
        )
        
        output_colored = ColorCodedMatrix(
            matrix_data=result,
            color_mapping={'default': self.color_palette['output']},
            element_labels={},
            highlight_patterns=[]
        )
        
        visualization = OperationVisualization(
            operation_type=f"elementwise_{operation}",
            input_matrices=[input_a_colored, input_b_colored],
            intermediate_steps=[],
            output_matrix=output_colored,
            animation_sequence=[]
        )
        
        properties = {
            'operation': f'elementwise_{operation}',
            'input_shapes': [A.shape, B.shape],
            'output_shape': result.shape,
            'preserves_shape': True
        }
        
        return MatrixOperationResult(
            result=result,
            computation_steps=[step],
            visualization=visualization,
            properties=properties
        )