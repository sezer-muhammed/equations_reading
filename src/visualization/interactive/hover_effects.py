"""
Hover effects for variable highlighting in mathematical visualizations.
Implements interactive highlighting that connects equations to visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass

from ...core.models import ColorCodedMatrix, HighlightPattern, VariableDefinition


@dataclass
class HoverRegion:
    """Defines a hoverable region in a visualization."""
    region_id: str
    bounds: Tuple[float, float, float, float]  # (x, y, width, height)
    variable_name: str
    description: str
    highlight_color: str = "#FFE66D"
    related_elements: List[str] = None


@dataclass
class HighlightStyle:
    """Style configuration for hover highlights."""
    border_color: str = "#FFE66D"
    border_width: float = 3.0
    fill_color: str = "#FFE66D"
    fill_alpha: float = 0.3
    text_color: str = "#000000"
    text_size: int = 10


class HoverEffectManager:
    """Manages hover effects for mathematical visualizations."""
    
    def __init__(
        self,
        figure: plt.Figure,
        axes: plt.Axes,
        style: Optional[HighlightStyle] = None
    ):
        self.figure = figure
        self.axes = axes
        self.style = style or HighlightStyle()
        
        # Hover state management
        self.hover_regions = {}
        self.active_highlights = {}
        self.hover_callbacks = {}
        self.tooltip_text = None
        
        # Connect mouse events
        self.figure.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.figure.canvas.mpl_connect('button_press_event', self._on_mouse_click)
        
        # Variable cross-reference mapping
        self.variable_mappings = {}
    
    def add_hover_region(
        self,
        region: HoverRegion,
        hover_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Add a hoverable region to the visualization."""
        self.hover_regions[region.region_id] = region
        
        if hover_callback:
            self.hover_callbacks[region.region_id] = hover_callback
    
    def add_variable_mapping(
        self,
        variable_name: str,
        matrix_elements: List[Tuple[int, int]],
        equation_regions: List[str]
    ) -> None:
        """Map variables to matrix elements and equation regions."""
        self.variable_mappings[variable_name] = {
            'matrix_elements': matrix_elements,
            'equation_regions': equation_regions
        }
    
    def create_matrix_hover_regions(
        self,
        matrix: ColorCodedMatrix,
        variable_definitions: Dict[str, VariableDefinition]
    ) -> None:
        """Create hover regions for matrix elements based on variable definitions."""
        rows, cols = matrix.matrix_data.shape
        
        # Create hover region for each matrix element
        for i in range(rows):
            for j in range(cols):
                region_id = f"matrix_element_{i}_{j}"
                
                # Determine variable name for this element
                var_name = self._get_element_variable_name(i, j, variable_definitions)
                
                region = HoverRegion(
                    region_id=region_id,
                    bounds=(j, rows - i - 1, 1, 1),  # Matrix coordinates
                    variable_name=var_name,
                    description=f"Element ({i},{j}): {var_name}",
                    related_elements=[f"eq_{var_name}"]
                )
                
                self.add_hover_region(region, self._create_matrix_hover_callback(i, j))
    
    def create_equation_hover_regions(
        self,
        equation_text_objects: List[Text],
        variable_definitions: Dict[str, VariableDefinition]
    ) -> None:
        """Create hover regions for equation variables."""
        for text_obj in equation_text_objects:
            # Extract variable names from text
            variables = self._extract_variables_from_text(text_obj.get_text())
            
            for var_name in variables:
                if var_name in variable_definitions:
                    region_id = f"eq_{var_name}"
                    
                    # Get text bounding box
                    bbox = text_obj.get_window_extent()
                    bounds = self._convert_bbox_to_data_coords(bbox)
                    
                    region = HoverRegion(
                        region_id=region_id,
                        bounds=bounds,
                        variable_name=var_name,
                        description=variable_definitions[var_name].description,
                        related_elements=[f"matrix_element_{i}_{j}" 
                                        for i, j in self._get_variable_matrix_elements(var_name)]
                    )
                    
                    self.add_hover_region(region, self._create_equation_hover_callback(var_name))
    
    def _on_mouse_move(self, event) -> None:
        """Handle mouse movement for hover effects."""
        if event.inaxes != self.axes:
            self._clear_all_highlights()
            return
        
        # Check which region the mouse is over
        current_region = self._get_region_at_position(event.xdata, event.ydata)
        
        if current_region:
            self._activate_hover(current_region)
        else:
            self._clear_all_highlights()
    
    def _on_mouse_click(self, event) -> None:
        """Handle mouse clicks for persistent highlighting."""
        if event.inaxes != self.axes:
            return
        
        current_region = self._get_region_at_position(event.xdata, event.ydata)
        
        if current_region:
            # Toggle persistent highlight
            region_id = current_region.region_id
            if region_id in self.active_highlights:
                self._remove_highlight(region_id)
            else:
                self._add_persistent_highlight(current_region)
    
    def _get_region_at_position(self, x: float, y: float) -> Optional[HoverRegion]:
        """Find the hover region at the given position."""
        if x is None or y is None:
            return None
        
        for region in self.hover_regions.values():
            bounds = region.bounds
            if (bounds[0] <= x <= bounds[0] + bounds[2] and
                bounds[1] <= y <= bounds[1] + bounds[3]):
                return region
        
        return None
    
    def _activate_hover(self, region: HoverRegion) -> None:
        """Activate hover effects for a region."""
        # Clear previous highlights
        self._clear_temporary_highlights()
        
        # Add highlight for current region
        self._add_highlight(region, temporary=True)
        
        # Highlight related elements
        if region.related_elements:
            for related_id in region.related_elements:
                if related_id in self.hover_regions:
                    related_region = self.hover_regions[related_id]
                    self._add_highlight(related_region, temporary=True)
        
        # Show tooltip
        self._show_tooltip(region)
        
        # Call custom callback if provided
        if region.region_id in self.hover_callbacks:
            self.hover_callbacks[region.region_id](region.variable_name)
        
        # Refresh display
        self.figure.canvas.draw_idle()
    
    def _add_highlight(self, region: HoverRegion, temporary: bool = False) -> None:
        """Add visual highlight to a region."""
        bounds = region.bounds
        
        # Create highlight rectangle
        highlight_rect = Rectangle(
            (bounds[0], bounds[1]), bounds[2], bounds[3],
            facecolor=self.style.fill_color,
            edgecolor=self.style.border_color,
            linewidth=self.style.border_width,
            alpha=self.style.fill_alpha,
            zorder=10
        )
        
        self.axes.add_patch(highlight_rect)
        
        # Store highlight reference
        highlight_key = f"{'temp_' if temporary else ''}{region.region_id}"
        self.active_highlights[highlight_key] = highlight_rect
    
    def _add_persistent_highlight(self, region: HoverRegion) -> None:
        """Add persistent highlight that remains until clicked again."""
        self._add_highlight(region, temporary=False)
        self.figure.canvas.draw_idle()
    
    def _remove_highlight(self, region_id: str) -> None:
        """Remove highlight from a specific region."""
        if region_id in self.active_highlights:
            highlight = self.active_highlights[region_id]
            highlight.remove()
            del self.active_highlights[region_id]
            self.figure.canvas.draw_idle()
    
    def _clear_temporary_highlights(self) -> None:
        """Clear all temporary highlights."""
        temp_keys = [key for key in self.active_highlights.keys() if key.startswith('temp_')]
        
        for key in temp_keys:
            highlight = self.active_highlights[key]
            highlight.remove()
            del self.active_highlights[key]
    
    def _clear_all_highlights(self) -> None:
        """Clear all highlights."""
        for highlight in self.active_highlights.values():
            highlight.remove()
        
        self.active_highlights.clear()
        self._hide_tooltip()
        self.figure.canvas.draw_idle()
    
    def _show_tooltip(self, region: HoverRegion) -> None:
        """Show tooltip with variable information."""
        self._hide_tooltip()  # Clear previous tooltip
        
        # Create tooltip text
        tooltip_text = f"{region.variable_name}: {region.description}"
        
        # Position tooltip near the region
        bounds = region.bounds
        x_pos = bounds[0] + bounds[2] / 2
        y_pos = bounds[1] + bounds[3] + 0.1
        
        self.tooltip_text = self.axes.text(
            x_pos, y_pos, tooltip_text,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            fontsize=self.style.text_size,
            ha='center',
            zorder=20
        )
    
    def _hide_tooltip(self) -> None:
        """Hide the current tooltip."""
        if self.tooltip_text:
            self.tooltip_text.remove()
            self.tooltip_text = None
    
    def _get_element_variable_name(
        self,
        row: int,
        col: int,
        variable_definitions: Dict[str, VariableDefinition]
    ) -> str:
        """Determine variable name for a matrix element."""
        # Simple heuristic - in practice, this would be more sophisticated
        for var_name, var_def in variable_definitions.items():
            if var_def.data_type == "matrix":
                return var_name
        
        return f"element_{row}_{col}"
    
    def _extract_variables_from_text(self, text: str) -> List[str]:
        """Extract variable names from equation text."""
        # Simple regex-based extraction - in practice, would use LaTeX parsing
        import re
        
        # Find single letters that could be variables
        variables = re.findall(r'\b[a-zA-Z]\b', text)
        
        # Filter out common non-variable letters
        non_variables = {'a', 'e', 'i', 'o', 'u', 'x', 'y', 'z'}  # Basic filter
        
        return [var for var in variables if var not in non_variables]
    
    def _convert_bbox_to_data_coords(self, bbox) -> Tuple[float, float, float, float]:
        """Convert text bounding box to data coordinates."""
        # Transform from display coordinates to data coordinates
        transform = self.axes.transData.inverted()
        
        # Get bbox in display coordinates
        x0, y0 = bbox.x0, bbox.y0
        x1, y1 = bbox.x1, bbox.y1
        
        # Transform to data coordinates
        data_coords = transform.transform([(x0, y0), (x1, y1)])
        
        x_min, y_min = data_coords[0]
        x_max, y_max = data_coords[1]
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _get_variable_matrix_elements(self, var_name: str) -> List[Tuple[int, int]]:
        """Get matrix elements associated with a variable."""
        if var_name in self.variable_mappings:
            return self.variable_mappings[var_name]['matrix_elements']
        
        return []
    
    def _create_matrix_hover_callback(self, row: int, col: int) -> Callable[[str], None]:
        """Create callback for matrix element hover."""
        def callback(var_name: str):
            print(f"Hovering over matrix element ({row}, {col}): {var_name}")
            # Custom logic for matrix element highlighting
        
        return callback
    
    def _create_equation_hover_callback(self, var_name: str) -> Callable[[str], None]:
        """Create callback for equation variable hover."""
        def callback(variable: str):
            print(f"Hovering over equation variable: {variable}")
            # Custom logic for equation variable highlighting
        
        return callback


class VariableHighlighter:
    """Specialized highlighter for mathematical variables across visualizations."""
    
    def __init__(self, hover_manager: HoverEffectManager):
        self.hover_manager = hover_manager
        self.variable_colors = {}
        self.cross_references = {}
    
    def setup_variable_highlighting(
        self,
        variables: Dict[str, VariableDefinition],
        matrix: ColorCodedMatrix
    ) -> None:
        """Setup comprehensive variable highlighting across equation and matrix."""
        
        # Assign consistent colors to variables
        self._assign_variable_colors(variables)
        
        # Create cross-references between equation variables and matrix elements
        self._create_cross_references(variables, matrix)
        
        # Setup hover regions
        self.hover_manager.create_matrix_hover_regions(matrix, variables)
    
    def _assign_variable_colors(self, variables: Dict[str, VariableDefinition]) -> None:
        """Assign consistent colors to variables."""
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
        
        for i, (var_name, var_def) in enumerate(variables.items()):
            color = var_def.color_code or colors[i % len(colors)]
            self.variable_colors[var_name] = color
    
    def _create_cross_references(
        self,
        variables: Dict[str, VariableDefinition],
        matrix: ColorCodedMatrix
    ) -> None:
        """Create cross-references between variables and matrix elements."""
        rows, cols = matrix.matrix_data.shape
        
        for var_name, var_def in variables.items():
            if var_def.data_type == "matrix":
                # Map entire matrix to this variable
                elements = [(i, j) for i in range(rows) for j in range(cols)]
                self.cross_references[var_name] = elements
            elif var_def.data_type == "vector":
                # Map row or column to this variable
                if var_def.shape and len(var_def.shape) == 1:
                    if var_def.shape[0] == rows:
                        # Column vector
                        elements = [(i, 0) for i in range(rows)]
                    else:
                        # Row vector
                        elements = [(0, j) for j in range(cols)]
                    self.cross_references[var_name] = elements
            elif var_def.data_type == "scalar":
                # Map to specific element (e.g., diagonal)
                elements = [(min(i, rows-1), min(i, cols-1)) for i in range(min(rows, cols))]
                self.cross_references[var_name] = elements


def create_attention_hover_demo() -> Tuple[plt.Figure, HoverEffectManager]:
    """Create demonstration of hover effects for attention mechanism."""
    
    # Create sample attention matrices
    query = np.random.randn(4, 3)
    key = np.random.randn(4, 3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create hover effect manager
    hover_manager = HoverEffectManager(fig, ax)
    
    # Create sample matrix visualization
    matrix = ColorCodedMatrix(
        matrix_data=query,
        color_mapping={"query": "#FF6B6B"},
        element_labels={(0, 0): "Q₀₀", (1, 1): "Q₁₁"}
    )
    
    # Define variables
    variables = {
        "Q": VariableDefinition(
            name="Q", description="Query matrix", data_type="matrix",
            shape=(4, 3), color_code="#FF6B6B"
        ),
        "K": VariableDefinition(
            name="K", description="Key matrix", data_type="matrix", 
            shape=(4, 3), color_code="#4ECDC4"
        )
    }
    
    # Setup hover regions
    hover_manager.create_matrix_hover_regions(matrix, variables)
    
    # Draw matrix
    im = ax.imshow(query, cmap='viridis', aspect='auto')
    ax.set_title("Hover over matrix elements to see variable highlighting")
    
    return fig, hover_manager