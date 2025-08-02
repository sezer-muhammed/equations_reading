"""
Matrix operation highlighting and animation sequences.
Provides step-by-step visual demonstrations of matrix operations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass

from ...core.models import (
    ColorCodedMatrix, HighlightPattern, AnimationFrame, 
    OperationVisualization
)
from .color_coded_matrix import ColorCodedMatrixVisualizer, MatrixVisualizationStyle


@dataclass
class AnimationConfig:
    """Configuration for matrix animations."""
    frame_duration_ms: int = 1000
    transition_duration_ms: int = 500
    loop: bool = True
    save_frames: bool = False
    output_format: str = "gif"


class MatrixAnimationEngine:
    """Creates animated visualizations of matrix operations."""
    
    def __init__(
        self, 
        visualizer: Optional[ColorCodedMatrixVisualizer] = None,
        config: Optional[AnimationConfig] = None
    ):
        self.visualizer = visualizer or ColorCodedMatrixVisualizer()
        self.config = config or AnimationConfig()
        self.current_animation = None
    
    def create_operation_animation(
        self, 
        operation_viz: OperationVisualization
    ) -> animation.FuncAnimation:
        """Create animation for a complete matrix operation."""
        
        # Prepare all frames
        all_matrices = (
            operation_viz.input_matrices + 
            operation_viz.intermediate_steps + 
            [operation_viz.output_matrix]
        )
        
        # Create figure with subplots for multiple matrices if needed
        fig, axes = self._setup_animation_figure(all_matrices)
        
        def animate_frame(frame_num: int):
            """Animation function for each frame."""
            # Clear all axes
            for ax in axes.flat:
                ax.clear()
            
            # Determine which matrices to show at this frame
            matrices_to_show = self._get_frame_matrices(
                frame_num, operation_viz
            )
            
            # Draw matrices on appropriate axes
            for idx, matrix in enumerate(matrices_to_show):
                if idx < len(axes.flat):
                    self._draw_matrix_on_axis(
                        axes.flat[idx], matrix, frame_num
                    )
        
        # Create animation
        total_frames = len(operation_viz.animation_sequence) or len(all_matrices)
        self.current_animation = animation.FuncAnimation(
            fig, animate_frame, frames=total_frames,
            interval=self.config.frame_duration_ms,
            repeat=self.config.loop,
            blit=False
        )
        
        return self.current_animation
    
    def create_matrix_multiplication_animation(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray,
        title: str = "Matrix Multiplication"
    ) -> animation.FuncAnimation:
        """Create step-by-step matrix multiplication animation."""
        
        # Validate dimensions
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError("Matrix dimensions incompatible for multiplication")
        
        result = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
        
        # Create animation frames for each element calculation
        frames = []
        for i in range(matrix_a.shape[0]):
            for j in range(matrix_b.shape[1]):
                # Calculate element (i,j)
                element_value = np.dot(matrix_a[i, :], matrix_b[:, j])
                result[i, j] = element_value
                
                # Create frame showing current calculation
                frame = self._create_multiplication_frame(
                    matrix_a, matrix_b, result.copy(), i, j, element_value
                )
                frames.append(frame)
        
        # Setup figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        
        def animate_multiplication(frame_num: int):
            """Animation function for matrix multiplication."""
            if frame_num >= len(frames):
                return
            
            frame_data = frames[frame_num]
            
            # Clear axes
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            
            # Draw matrices with highlights
            self._draw_multiplication_frame(ax1, ax2, ax3, frame_data)
        
        self.current_animation = animation.FuncAnimation(
            fig, animate_multiplication, frames=len(frames),
            interval=self.config.frame_duration_ms,
            repeat=self.config.loop,
            blit=False
        )
        
        return self.current_animation
    
    def create_attention_animation(
        self, 
        query: np.ndarray, 
        key: np.ndarray, 
        value: np.ndarray,
        title: str = "Attention Mechanism"
    ) -> animation.FuncAnimation:
        """Create attention mechanism visualization with Q, K, V highlighting."""
        
        # Calculate attention components
        scores = np.matmul(query, key.T)
        attention_weights = self._softmax(scores)
        output = np.matmul(attention_weights, value)
        
        # Create frames for each step
        frames = [
            {"step": "query_key", "data": {"Q": query, "K": key, "scores": scores}},
            {"step": "softmax", "data": {"scores": scores, "weights": attention_weights}},
            {"step": "weighted_sum", "data": {"weights": attention_weights, "V": value, "output": output}}
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        def animate_attention(frame_num: int):
            """Animation function for attention mechanism."""
            if frame_num >= len(frames):
                return
            
            frame_data = frames[frame_num]
            
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            self._draw_attention_frame(axes, frame_data)
        
        self.current_animation = animation.FuncAnimation(
            fig, animate_attention, frames=len(frames),
            interval=self.config.frame_duration_ms * 2,  # Slower for attention
            repeat=self.config.loop,
            blit=False
        )
        
        return self.current_animation
    
    def _setup_animation_figure(
        self, 
        matrices: List[ColorCodedMatrix]
    ) -> tuple:
        """Setup figure and axes for animation."""
        num_matrices = len(matrices)
        
        if num_matrices <= 2:
            fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 6))
        elif num_matrices <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        else:
            # For more matrices, use a grid layout
            rows = int(np.ceil(np.sqrt(num_matrices)))
            cols = int(np.ceil(num_matrices / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        
        # Ensure axes is always a flat array
        if not hasattr(axes, 'flat'):
            axes = np.array([axes])
        
        return fig, axes
    
    def _get_frame_matrices(
        self, 
        frame_num: int, 
        operation_viz: OperationVisualization
    ) -> List[ColorCodedMatrix]:
        """Get matrices to display for a specific frame."""
        if operation_viz.animation_sequence:
            # Use predefined animation sequence
            if frame_num < len(operation_viz.animation_sequence):
                frame = operation_viz.animation_sequence[frame_num]
                # Convert frame data to matrices (implementation depends on frame structure)
                return self._frame_to_matrices(frame, operation_viz)
        
        # Default: show progression through operation steps
        all_matrices = (
            operation_viz.input_matrices + 
            operation_viz.intermediate_steps + 
            [operation_viz.output_matrix]
        )
        
        if frame_num < len(all_matrices):
            return all_matrices[:frame_num + 1]
        else:
            return all_matrices
    
    def _draw_matrix_on_axis(
        self, 
        ax: plt.Axes, 
        matrix: ColorCodedMatrix, 
        frame_num: int
    ) -> None:
        """Draw a single matrix on the given axis."""
        # Use the visualizer to draw the matrix
        # This is a simplified version - in practice, we'd need to adapt
        # the visualizer to work with existing axes
        rows, cols = matrix.matrix_data.shape
        
        # Create color mapping
        colors = self.visualizer._create_color_mapping(matrix, "matrix")
        
        # Draw cells
        self.visualizer._draw_matrix_cells(ax, matrix, colors)
        
        # Add values and highlights
        if self.visualizer.style.show_values:
            self.visualizer._add_value_labels(ax, matrix)
        
        self.visualizer._apply_highlights(ax, matrix)
        self.visualizer._configure_axes(ax, matrix, f"Frame {frame_num}")
    
    def _create_multiplication_frame(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray, 
        result: np.ndarray,
        row: int, 
        col: int, 
        value: float
    ) -> Dict[str, Any]:
        """Create frame data for matrix multiplication step."""
        return {
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "result": result,
            "current_row": row,
            "current_col": col,
            "current_value": value,
            "highlight_row": row,
            "highlight_col": col
        }
    
    def _draw_multiplication_frame(
        self, 
        ax1: plt.Axes, 
        ax2: plt.Axes, 
        ax3: plt.Axes, 
        frame_data: Dict[str, Any]
    ) -> None:
        """Draw matrix multiplication frame."""
        matrix_a = frame_data["matrix_a"]
        matrix_b = frame_data["matrix_b"]
        result = frame_data["result"]
        row = frame_data["current_row"]
        col = frame_data["current_col"]
        
        # Create ColorCodedMatrix objects with highlights
        a_highlights = [HighlightPattern("row", [(row,)], "red", "Current row")]
        b_highlights = [HighlightPattern("column", [(col,)], "blue", "Current column")]
        r_highlights = [HighlightPattern("element", [(row, col)], "green", "Current result")]
        
        matrix_a_viz = ColorCodedMatrix(matrix_a, {}, highlight_patterns=a_highlights)
        matrix_b_viz = ColorCodedMatrix(matrix_b, {}, highlight_patterns=b_highlights)
        result_viz = ColorCodedMatrix(result, {}, highlight_patterns=r_highlights)
        
        # Draw on axes (simplified - would need full implementation)
        ax1.set_title("Matrix A")
        ax2.set_title("Matrix B")
        ax3.set_title("Result")
    
    def _draw_attention_frame(
        self, 
        axes: np.ndarray, 
        frame_data: Dict[str, Any]
    ) -> None:
        """Draw attention mechanism frame."""
        step = frame_data["step"]
        data = frame_data["data"]
        
        if step == "query_key":
            # Show Q, K matrices and their product
            pass
        elif step == "softmax":
            # Show scores and attention weights
            pass
        elif step == "weighted_sum":
            # Show final weighted sum calculation
            pass
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _frame_to_matrices(
        self, 
        frame: AnimationFrame, 
        operation_viz: OperationVisualization
    ) -> List[ColorCodedMatrix]:
        """Convert animation frame to list of matrices."""
        # This would convert the frame data back to ColorCodedMatrix objects
        # Implementation depends on how frames are structured
        return operation_viz.input_matrices  # Placeholder
    
    def save_animation(
        self, 
        filename: str, 
        writer: str = "pillow"
    ) -> None:
        """Save the current animation to file."""
        if self.current_animation is None:
            raise ValueError("No animation to save")
        
        self.current_animation.save(filename, writer=writer)


def create_sample_multiplication_animation() -> animation.FuncAnimation:
    """Create a sample matrix multiplication animation for testing."""
    engine = MatrixAnimationEngine()
    
    # Create sample matrices
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    return engine.create_matrix_multiplication_animation(a, b)