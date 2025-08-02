"""
Step-by-step matrix operation visualizations.
Provides detailed visual breakdowns of mathematical operations.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

from src.core.models import ColorCodedMatrix, HighlightPattern, OperationVisualization, AnimationFrame
from src.visualization.matrix_viz import ColorCodedMatrixVisualizer, ConceptType, get_concept_colors


@dataclass
class OperationStep:
    """Represents a single step in a matrix operation."""
    step_number: int
    description: str
    input_matrices: List[ColorCodedMatrix]
    output_matrix: ColorCodedMatrix
    highlights: List[HighlightPattern]
    mathematical_expression: str

class MatrixOperationVisualizer:
    """Visualizes step-by-step matrix operations with detailed breakdowns."""
    
    def __init__(self):
        self.matrix_viz = ColorCodedMatrixVisualizer()
        self.concept_colors = get_concept_colors(ConceptType.MATRIX_OPERATIONS)
    
    def visualize_matrix_multiplication(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray,
        show_steps: bool = True
    ) -> OperationVisualization:
        """
        Create step-by-step visualization of matrix multiplication.
        
        Args:
            matrix_a: First matrix (m x n)
            matrix_b: Second matrix (n x p)
            show_steps: Whether to show intermediate computation steps
            
        Returns:
            OperationVisualization with complete step breakdown
        """
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {matrix_a.shape} x {matrix_b.shape}")
        
        # Create color-coded input matrices
        input_a = ColorCodedMatrix(
            matrix_data=matrix_a,
            color_mapping={"default": self.concept_colors.get_color("input_a")},
            element_labels=self._create_matrix_labels(matrix_a, "A")
        )
        
        input_b = ColorCodedMatrix(
            matrix_data=matrix_b,
            color_mapping={"default": self.concept_colors.get_color("input_b")},
            element_labels=self._create_matrix_labels(matrix_b, "B")
        )
        
        # Compute result
        result = np.dot(matrix_a, matrix_b)
        output_matrix = ColorCodedMatrix(
            matrix_data=result,
            color_mapping={"default": self.concept_colors.get_color("output")},
            element_labels=self._create_matrix_labels(result, "C")
        )
        
        # Create intermediate steps if requested
        intermediate_steps = []
        animation_frames = []
        
        if show_steps:
            intermediate_steps, animation_frames = self._create_multiplication_steps(
                matrix_a, matrix_b, result
            )
        
        return OperationVisualization(
            operation_type="matrix_multiply",
            input_matrices=[input_a, input_b],
            intermediate_steps=intermediate_steps,
            output_matrix=output_matrix,
            animation_sequence=animation_frames
        )
    
    def visualize_attention_computation(
        self,
        query: np.ndarray,
        key: np.ndarray, 
        value: np.ndarray,
        scale_factor: Optional[float] = None
    ) -> OperationVisualization:
        """
        Visualize scaled dot-product attention computation.
        
        Args:
            query: Query matrix (seq_len x d_k)
            key: Key matrix (seq_len x d_k)  
            value: Value matrix (seq_len x d_v)
            scale_factor: Scaling factor (default: 1/sqrt(d_k))
            
        Returns:
            OperationVisualization showing attention computation steps
        """
        if scale_factor is None:
            scale_factor = 1.0 / np.sqrt(query.shape[-1])
        
        # Step 1: Q * K^T
        key_transpose = key.T
        qk_scores = np.dot(query, key_transpose)
        
        # Step 2: Scale
        scaled_scores = qk_scores * scale_factor
        
        # Step 3: Softmax
        attention_weights = self._softmax(scaled_scores)
        
        # Step 4: Apply to values
        output = np.dot(attention_weights, value)
        
        # Create color-coded matrices for each step
        query_matrix = ColorCodedMatrix(
            matrix_data=query,
            color_mapping={"default": self.concept_colors.get_color("query")},
            element_labels=self._create_matrix_labels(query, "Q")
        )
        
        key_matrix = ColorCodedMatrix(
            matrix_data=key,
            color_mapping={"default": self.concept_colors.get_color("key")},
            element_labels=self._create_matrix_labels(key, "K")
        )
        
        value_matrix = ColorCodedMatrix(
            matrix_data=value,
            color_mapping={"default": self.concept_colors.get_color("value")},
            element_labels=self._create_matrix_labels(value, "V")
        )
        
        # Intermediate steps
        qk_matrix = ColorCodedMatrix(
            matrix_data=qk_scores,
            color_mapping={"default": self.concept_colors.get_color("intermediate")},
            element_labels=self._create_matrix_labels(qk_scores, "QK^T")
        )
        
        scaled_matrix = ColorCodedMatrix(
            matrix_data=scaled_scores,
            color_mapping={"default": self.concept_colors.get_color("intermediate")},
            element_labels=self._create_matrix_labels(scaled_scores, "Scaled")
        )
        
        attention_matrix = ColorCodedMatrix(
            matrix_data=attention_weights,
            color_mapping={"default": self.concept_colors.get_color("attention")},
            element_labels=self._create_matrix_labels(attention_weights, "Attn")
        )
        
        output_matrix = ColorCodedMatrix(
            matrix_data=output,
            color_mapping={"default": self.concept_colors.get_color("output")},
            element_labels=self._create_matrix_labels(output, "Out")
        )
        
        return OperationVisualization(
            operation_type="attention",
            input_matrices=[query_matrix, key_matrix, value_matrix],
            intermediate_steps=[qk_matrix, scaled_matrix, attention_matrix],
            output_matrix=output_matrix,
            animation_sequence=self._create_attention_animation(
                query, key, value, attention_weights, output
            )
        )
    
    def visualize_gradient_flow(
        self,
        forward_matrices: List[np.ndarray],
        gradients: List[np.ndarray],
        layer_names: List[str]
    ) -> OperationVisualization:
        """
        Visualize gradient flow through network layers.
        
        Args:
            forward_matrices: Forward pass activations
            gradients: Corresponding gradients
            layer_names: Names for each layer
            
        Returns:
            OperationVisualization showing gradient backpropagation
        """
        if len(forward_matrices) != len(gradients) or len(forward_matrices) != len(layer_names):
            raise ValueError("All input lists must have the same length")
        
        # Create forward pass matrices
        forward_colored = []
        for i, (matrix, name) in enumerate(zip(forward_matrices, layer_names)):
            colored_matrix = ColorCodedMatrix(
                matrix_data=matrix,
                color_mapping={"default": self.concept_colors.get_color("forward")},
                element_labels=self._create_matrix_labels(matrix, f"F_{name}")
            )
            forward_colored.append(colored_matrix)
        
        # Create gradient matrices
        gradient_colored = []
        for i, (grad, name) in enumerate(zip(gradients, layer_names)):
            colored_grad = ColorCodedMatrix(
                matrix_data=grad,
                color_mapping={"default": self._get_gradient_color(grad)},
                element_labels=self._create_matrix_labels(grad, f"∇{name}")
            )
            gradient_colored.append(colored_grad)
        
        # Create combined visualization using the last gradient as representative
        combined_matrix = gradients[-1]  # Use the final layer gradient as output
        output_matrix = ColorCodedMatrix(
            matrix_data=combined_matrix,
            color_mapping={"default": self.concept_colors.get_color("gradient_flow")},
            element_labels=self._create_matrix_labels(combined_matrix, "∇Total")
        )
        
        return OperationVisualization(
            operation_type="gradient_flow",
            input_matrices=forward_colored,
            intermediate_steps=gradient_colored,
            output_matrix=output_matrix,
            animation_sequence=self._create_gradient_animation(forward_matrices, gradients)
        )
    
    def create_operation_plot(
        self, 
        operation_viz: OperationVisualization,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Create a comprehensive plot showing the operation visualization.
        
        Args:
            operation_viz: The operation visualization to plot
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure with the complete visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'{operation_viz.operation_type.replace("_", " ").title()} Visualization', 
                    fontsize=16, fontweight='bold')
        
        # Simple matrix visualization using imshow
        def plot_matrix_simple(matrix: ColorCodedMatrix, ax: plt.Axes, title: str):
            """Simple matrix plotting using imshow."""
            im = ax.imshow(matrix.matrix_data, cmap='viridis', aspect='equal')
            ax.set_title(title)
            
            # Add value annotations
            for i in range(matrix.matrix_data.shape[0]):
                for j in range(matrix.matrix_data.shape[1]):
                    text = ax.text(j, i, f'{matrix.matrix_data[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=8)
            
            return im
        
        # Plot input matrices
        for i, input_matrix in enumerate(operation_viz.input_matrices[:2]):
            if i < 2:
                plot_matrix_simple(input_matrix, axes[0, i], f'Input Matrix {i+1}')
        
        # Plot intermediate steps
        if operation_viz.intermediate_steps:
            for i, step_matrix in enumerate(operation_viz.intermediate_steps[:2]):
                if i < 2:
                    plot_matrix_simple(step_matrix, axes[1, i], f'Step {i+1}')
        
        # Plot output
        plot_matrix_simple(operation_viz.output_matrix, axes[0, 2], 'Output Matrix')
        
        # Add operation description
        if len(operation_viz.intermediate_steps) > 2:
            axes[1, 2].text(0.1, 0.5, 'Additional steps available\nin animation sequence', 
                          transform=axes[1, 2].transAxes, fontsize=12,
                          verticalalignment='center')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _create_matrix_labels(self, matrix: np.ndarray, prefix: str) -> Dict[Tuple[int, int], str]:
        """Create element labels for matrix visualization."""
        labels = {}
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                labels[(i, j)] = f"{prefix}[{i},{j}]"
        return labels
    
    def _create_multiplication_steps(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray, 
        result: np.ndarray
    ) -> Tuple[List[ColorCodedMatrix], List[AnimationFrame]]:
        """Create step-by-step multiplication visualization."""
        intermediate_steps = []
        animation_frames = []
        
        m, n = matrix_a.shape
        n2, p = matrix_b.shape
        
        # Create partial result matrices for each step
        for i in range(m):
            for j in range(p):
                # Highlight current computation
                partial_result = np.zeros_like(result)
                partial_result[i, j] = result[i, j]
                
                # Create highlights for current row and column
                row_highlight = HighlightPattern(
                    pattern_type="row",
                    indices=[(i, k) for k in range(n)],
                    color=self.concept_colors.get_color("highlight_row"),
                    description=f"Row {i} of matrix A"
                )
                
                col_highlight = HighlightPattern(
                    pattern_type="column", 
                    indices=[(k, j) for k in range(n)],
                    color=self.concept_colors.get_color("highlight_col"),
                    description=f"Column {j} of matrix B"
                )
                
                step_matrix = ColorCodedMatrix(
                    matrix_data=partial_result,
                    color_mapping={"default": self.concept_colors.get_color("partial_result")},
                    element_labels=self._create_matrix_labels(partial_result, "C"),
                    highlight_patterns=[row_highlight, col_highlight]
                )
                
                intermediate_steps.append(step_matrix)
                
                # Create animation frame
                frame = AnimationFrame(
                    frame_number=i * p + j,
                    matrix_state=partial_result.copy(),
                    highlights=[row_highlight, col_highlight],
                    description=f"Computing C[{i},{j}] = Σ A[{i},k] × B[k,{j}]",
                    duration_ms=1500
                )
                animation_frames.append(frame)
        
        return intermediate_steps, animation_frames
    
    def _create_attention_animation(
        self,
        query: np.ndarray,
        key: np.ndarray, 
        value: np.ndarray,
        attention_weights: np.ndarray,
        output: np.ndarray
    ) -> List[AnimationFrame]:
        """Create animation sequence for attention computation."""
        frames = []
        
        # Frame 1: Show Q and K
        frames.append(AnimationFrame(
            frame_number=0,
            matrix_state=query,
            highlights=[],
            description="Query (Q) and Key (K) matrices",
            duration_ms=2000
        ))
        
        # Frame 2: Show Q × K^T computation
        qk_scores = np.dot(query, key.T)
        frames.append(AnimationFrame(
            frame_number=1,
            matrix_state=qk_scores,
            highlights=[],
            description="Computing Q × K^T (attention scores)",
            duration_ms=2000
        ))
        
        # Frame 3: Show attention weights after softmax
        frames.append(AnimationFrame(
            frame_number=2,
            matrix_state=attention_weights,
            highlights=[],
            description="Attention weights after softmax normalization",
            duration_ms=2000
        ))
        
        # Frame 4: Show final output
        frames.append(AnimationFrame(
            frame_number=3,
            matrix_state=output,
            highlights=[],
            description="Final output: Attention × Value",
            duration_ms=2000
        ))
        
        return frames
    
    def _create_gradient_animation(
        self,
        forward_matrices: List[np.ndarray],
        gradients: List[np.ndarray]
    ) -> List[AnimationFrame]:
        """Create animation for gradient backpropagation."""
        frames = []
        
        # Forward pass frames
        for i, matrix in enumerate(forward_matrices):
            frames.append(AnimationFrame(
                frame_number=i,
                matrix_state=matrix,
                highlights=[],
                description=f"Forward pass - Layer {i+1}",
                duration_ms=1500
            ))
        
        # Backward pass frames
        for i, grad in enumerate(reversed(gradients)):
            frames.append(AnimationFrame(
                frame_number=len(forward_matrices) + i,
                matrix_state=grad,
                highlights=[],
                description=f"Gradient backprop - Layer {len(gradients)-i}",
                duration_ms=1500
            ))
        
        return frames
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _get_gradient_color(self, gradient: np.ndarray) -> str:
        """Get color based on gradient magnitude."""
        magnitude = np.mean(np.abs(gradient))
        if magnitude > 1.0:
            return self.concept_colors.get_color("high_gradient")
        elif magnitude > 0.1:
            return self.concept_colors.get_color("medium_gradient") 
        else:
            return self.concept_colors.get_color("low_gradient")


def create_sample_matrix_multiplication() -> OperationVisualization:
    """Create a sample matrix multiplication visualization for testing."""
    visualizer = MatrixOperationVisualizer()
    
    # Create sample matrices
    matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_b = np.array([[7, 8], [9, 10], [11, 12]])
    
    return visualizer.visualize_matrix_multiplication(matrix_a, matrix_b, show_steps=True)


def create_sample_attention_visualization() -> OperationVisualization:
    """Create a sample attention computation visualization for testing."""
    visualizer = MatrixOperationVisualizer()
    
    # Create sample Q, K, V matrices
    seq_len, d_k, d_v = 4, 3, 3
    query = np.random.randn(seq_len, d_k) * 0.5
    key = np.random.randn(seq_len, d_k) * 0.5  
    value = np.random.randn(seq_len, d_v) * 0.5
    
    return visualizer.visualize_attention_computation(query, key, value)