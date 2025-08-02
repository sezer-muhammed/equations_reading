"""
Color-coded matrix visualizer for mathematical concepts.
Implements element-wise color coding with consistent schemes across related concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ...core.models import ColorCodedMatrix, HighlightPattern, AnimationFrame
from ...core.config import get_color_scheme, get_config


@dataclass
class MatrixVisualizationStyle:
    """Style configuration for matrix visualization."""
    cell_size: float = 1.0
    border_width: float = 0.1
    font_size: int = 10
    show_values: bool = True
    show_grid: bool = True
    colorbar: bool = True
    title_size: int = 12


class ColorCodedMatrixVisualizer:
    """Creates color-coded visualizations of matrices with consistent styling."""
    
    def __init__(self, style: Optional[MatrixVisualizationStyle] = None):
        self.style = style or MatrixVisualizationStyle()
        self.config = get_config()
        self.color_schemes = {
            'matrix': get_color_scheme('matrix'),
            'operation': get_color_scheme('operation'),
            'default': get_color_scheme('default')
        }
    
    def create_matrix_visualization(
        self, 
        matrix: ColorCodedMatrix,
        title: str = "",
        scheme: str = "matrix"
    ) -> plt.Figure:
        """Create a color-coded visualization of a matrix."""
        fig, ax = plt.subplots(
            figsize=self.config.visualization.default_figure_size,
            dpi=self.config.visualization.default_dpi
        )
        
        # Get matrix dimensions
        rows, cols = matrix.matrix_data.shape
        
        # Create color mapping
        colors = self._create_color_mapping(matrix, scheme)
        
        # Draw matrix cells
        self._draw_matrix_cells(ax, matrix, colors)
        
        # Add value labels if enabled
        if self.style.show_values:
            self._add_value_labels(ax, matrix)
        
        # Add element labels if provided
        if matrix.element_labels:
            self._add_element_labels(ax, matrix)
        
        # Apply highlight patterns
        self._apply_highlights(ax, matrix)
        
        # Configure axes and styling
        self._configure_axes(ax, matrix, title)
        
        # Add colorbar if enabled
        if self.style.colorbar and scheme in ['matrix']:
            self._add_colorbar(fig, ax, matrix, colors)
        
        plt.tight_layout()
        return fig
    
    def _create_color_mapping(
        self, 
        matrix: ColorCodedMatrix, 
        scheme: str
    ) -> np.ndarray:
        """Create color mapping for matrix elements."""
        data = matrix.matrix_data
        
        if scheme == "matrix":
            # Use gradient coloring based on values
            if 'gradient' in self.color_schemes['matrix']:
                gradient_colors = self.color_schemes['matrix']['gradient']
                cmap = LinearSegmentedColormap.from_list(
                    "custom", gradient_colors
                )
            else:
                cmap = plt.cm.viridis
            
            # Normalize values for color mapping
            norm = Normalize(vmin=data.min(), vmax=data.max())
            colors = cmap(norm(data))
            
        elif scheme == "operation":
            # Use discrete colors for different operation types
            colors = np.full(data.shape + (4,), 0.8)  # Default light gray
            
            # Apply custom color mapping if provided
            for key, color_code in matrix.color_mapping.items():
                if key in self.color_schemes['operation']:
                    # Apply color based on matrix regions or conditions
                    pass
        
        else:
            # Default grayscale based on values
            norm = Normalize(vmin=data.min(), vmax=data.max())
            colors = plt.cm.gray(norm(data))
        
        return colors
    
    def _draw_matrix_cells(
        self, 
        ax: plt.Axes, 
        matrix: ColorCodedMatrix, 
        colors: np.ndarray
    ) -> None:
        """Draw individual matrix cells with colors."""
        rows, cols = matrix.matrix_data.shape
        
        for i in range(rows):
            for j in range(cols):
                # Create rectangle for each cell
                rect = patches.Rectangle(
                    (j, rows - i - 1), 1, 1,
                    facecolor=colors[i, j] if colors.ndim > 2 else colors[i, j],
                    edgecolor='black' if self.style.show_grid else 'none',
                    linewidth=self.style.border_width
                )
                ax.add_patch(rect)
    
    def _add_value_labels(
        self, 
        ax: plt.Axes, 
        matrix: ColorCodedMatrix
    ) -> None:
        """Add numerical values as text labels on matrix cells."""
        rows, cols = matrix.matrix_data.shape
        data = matrix.matrix_data
        
        for i in range(rows):
            for j in range(cols):
                value = data[i, j]
                # Format value based on magnitude
                if abs(value) < 0.01:
                    text = f"{value:.2e}"
                elif abs(value) < 1:
                    text = f"{value:.3f}"
                else:
                    text = f"{value:.2f}"
                
                ax.text(
                    j + 0.5, rows - i - 0.5, text,
                    ha='center', va='center',
                    fontsize=self.style.font_size,
                    color='white' if self._is_dark_color(i, j, matrix) else 'black'
                )
    
    def _add_element_labels(
        self, 
        ax: plt.Axes, 
        matrix: ColorCodedMatrix
    ) -> None:
        """Add custom element labels to matrix cells."""
        rows = matrix.matrix_data.shape[0]
        
        for (i, j), label in matrix.element_labels.items():
            ax.text(
                j + 0.5, rows - i - 0.5, label,
                ha='center', va='center',
                fontsize=self.style.font_size + 2,
                fontweight='bold',
                color='red'
            )
    
    def _apply_highlights(
        self, 
        ax: plt.Axes, 
        matrix: ColorCodedMatrix
    ) -> None:
        """Apply highlight patterns to matrix visualization."""
        rows = matrix.matrix_data.shape[0]
        
        for pattern in matrix.highlight_patterns:
            if pattern.pattern_type == "row":
                for row_idx in pattern.indices:
                    if isinstance(row_idx, tuple):
                        row_idx = row_idx[0]
                    rect = patches.Rectangle(
                        (0, rows - row_idx - 1), matrix.matrix_data.shape[1], 1,
                        facecolor='none',
                        edgecolor=pattern.color,
                        linewidth=3
                    )
                    ax.add_patch(rect)
            
            elif pattern.pattern_type == "column":
                for col_idx in pattern.indices:
                    if isinstance(col_idx, tuple):
                        col_idx = col_idx[0]
                    rect = patches.Rectangle(
                        (col_idx, 0), 1, rows,
                        facecolor='none',
                        edgecolor=pattern.color,
                        linewidth=3
                    )
                    ax.add_patch(rect)
            
            elif pattern.pattern_type == "element":
                for idx in pattern.indices:
                    i, j = idx
                    rect = patches.Rectangle(
                        (j, rows - i - 1), 1, 1,
                        facecolor='none',
                        edgecolor=pattern.color,
                        linewidth=3
                    )
                    ax.add_patch(rect)
    
    def _configure_axes(
        self, 
        ax: plt.Axes, 
        matrix: ColorCodedMatrix, 
        title: str
    ) -> None:
        """Configure axes appearance and labels."""
        rows, cols = matrix.matrix_data.shape
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(cols) + 0.5)
        ax.set_yticks(np.arange(rows) + 0.5)
        ax.set_xticklabels(range(cols))
        ax.set_yticklabels(range(rows - 1, -1, -1))
        
        # Add title
        if title:
            ax.set_title(title, fontsize=self.style.title_size, pad=20)
        
        # Remove tick marks
        ax.tick_params(length=0)
    
    def _add_colorbar(
        self, 
        fig: plt.Figure, 
        ax: plt.Axes, 
        matrix: ColorCodedMatrix, 
        colors: np.ndarray
    ) -> None:
        """Add colorbar to show value mapping."""
        data = matrix.matrix_data
        norm = Normalize(vmin=data.min(), vmax=data.max())
        
        if 'gradient' in self.color_schemes['matrix']:
            gradient_colors = self.color_schemes['matrix']['gradient']
            cmap = LinearSegmentedColormap.from_list("custom", gradient_colors)
        else:
            cmap = plt.cm.viridis
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Value', rotation=270, labelpad=15)
    
    def _is_dark_color(
        self, 
        i: int, 
        j: int, 
        matrix: ColorCodedMatrix
    ) -> bool:
        """Determine if a cell has a dark background color."""
        # Simple heuristic: if value is in lower half of range, consider dark
        data = matrix.matrix_data
        normalized_value = (data[i, j] - data.min()) / (data.max() - data.min())
        return normalized_value < 0.5


def create_sample_matrix() -> ColorCodedMatrix:
    """Create a sample matrix for testing visualization."""
    data = np.random.randn(4, 4)
    
    highlights = [
        HighlightPattern(
            pattern_type="row",
            indices=[(0,)],
            color="red",
            description="First row highlight"
        ),
        HighlightPattern(
            pattern_type="element",
            indices=[(1, 1), (2, 2)],
            color="blue",
            description="Diagonal elements"
        )
    ]
    
    return ColorCodedMatrix(
        matrix_data=data,
        color_mapping={"default": "viridis"},
        element_labels={(0, 0): "A", (3, 3): "B"},
        highlight_patterns=highlights
    )