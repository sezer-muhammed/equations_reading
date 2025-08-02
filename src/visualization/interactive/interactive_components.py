"""
Main interactive visualization components that integrate parameter widgets,
hover effects, and animation controls for comprehensive mathematical demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass

from ...core.models import (
    VariableDefinition, ColorCodedMatrix, OperationVisualization,
    NumericalExample, MathematicalConcept
)
from .parameter_widgets import MathematicalParameterWidget, ParameterConfig
from .hover_effects import HoverEffectManager, VariableHighlighter
from .animation_controls import StepByStepAnimationController, AnimationControlConfig


@dataclass
class InteractiveVisualizationConfig:
    """Configuration for interactive mathematical visualizations."""
    enable_parameter_widgets: bool = True
    enable_hover_effects: bool = True
    enable_animation_controls: bool = True
    figure_size: Tuple[float, float] = (14, 10)
    widget_panel_width: float = 0.25
    control_panel_height: float = 0.15


class InteractiveMathVisualization:
    """Comprehensive interactive visualization for mathematical concepts."""
    
    def __init__(
        self,
        concept: MathematicalConcept,
        config: Optional[InteractiveVisualizationConfig] = None
    ):
        self.concept = concept
        self.config = config or InteractiveVisualizationConfig()
        
        # Create figure with appropriate layout
        self.figure = plt.figure(figsize=self.config.figure_size)
        self._setup_layout()
        
        # Interactive components
        self.parameter_widget = None
        self.hover_manager = None
        self.animation_controller = None
        self.variable_highlighter = None
        
        # Current state
        self.current_example = None
        self.current_visualization = None
        
        # Initialize components
        self._initialize_components()
    
    def _setup_layout(self) -> None:
        """Setup figure layout with space for widgets and controls."""
        # Main visualization area
        if self.config.enable_parameter_widgets:
            # Leave space for parameter widgets on the right
            main_width = 1.0 - self.config.widget_panel_width - 0.05
            self.main_ax = plt.axes((0.05, 0.2, main_width, 0.75))
        else:
            self.main_ax = plt.axes((0.05, 0.2, 0.9, 0.75))
        
        # Title area
        self.title_ax = plt.axes((0.05, 0.95, 0.9, 0.04))
        self.title_ax.axis('off')
        self.title_text = self.title_ax.text(
            0.5, 0.5, self.concept.title,
            ha='center', va='center',
            fontsize=16, fontweight='bold'
        )
    
    def _initialize_components(self) -> None:
        """Initialize all interactive components."""
        
        # Get first example and visualization for initialization
        if self.concept.examples:
            self.current_example = self.concept.examples[0]
        
        if self.concept.visualizations:
            self.current_visualization = self.concept.visualizations[0]
        
        # Initialize parameter widgets
        if self.config.enable_parameter_widgets and self.concept.equations:
            self._initialize_parameter_widgets()
        
        # Initialize hover effects
        if self.config.enable_hover_effects:
            self._initialize_hover_effects()
        
        # Initialize animation controls
        if self.config.enable_animation_controls:
            self._initialize_animation_controls()
    
    def _initialize_parameter_widgets(self) -> None:
        """Initialize parameter manipulation widgets."""
        if not self.concept.equations:
            return
        
        # Use variables from first equation
        equation = self.concept.equations[0]
        
        def update_visualization(params: Dict[str, float]) -> None:
            """Update visualization with new parameter values."""
            self._update_with_parameters(params)
        
        self.parameter_widget = MathematicalParameterWidget(
            equation.variables,
            update_visualization,
            self.config.figure_size
        )
        
        # Replace main axis with parameter widget's main axis
        self.main_ax.remove()
        self.main_ax = self.parameter_widget.get_main_axis()
    
    def _initialize_hover_effects(self) -> None:
        """Initialize hover effects for variable highlighting."""
        self.hover_manager = HoverEffectManager(self.figure, self.main_ax)
        
        if self.concept.equations:
            equation = self.concept.equations[0]
            self.variable_highlighter = VariableHighlighter(self.hover_manager)
    
    def _initialize_animation_controls(self) -> None:
        """Initialize animation controls for step-by-step demonstrations."""
        # This will be setup when an operation visualization is loaded
        pass
    
    def load_example(self, example_index: int = 0) -> None:
        """Load a specific numerical example."""
        if example_index < len(self.concept.examples):
            self.current_example = self.concept.examples[example_index]
            self._render_current_state()
    
    def load_visualization(self, viz_index: int = 0) -> None:
        """Load a specific visualization."""
        if viz_index < len(self.concept.visualizations):
            self.current_visualization = self.concept.visualizations[viz_index]
            self._render_current_state()
    
    def create_operation_animation(self, operation_viz: OperationVisualization) -> None:
        """Create animated demonstration of a mathematical operation."""
        if not self.config.enable_animation_controls:
            return
        
        def render_frame(frame_num: int) -> None:
            """Render a specific animation frame."""
            self._render_operation_frame(operation_viz, frame_num)
        
        # Create animation controller
        anim_config = AnimationControlConfig(
            control_panel_height=self.config.control_panel_height
        )
        
        self.animation_controller = StepByStepAnimationController(
            self.figure, operation_viz, render_frame, anim_config
        )
        
        # Initial render
        render_frame(0)
    
    def _update_with_parameters(self, params: Dict[str, float]) -> None:
        """Update visualization with new parameter values."""
        if not self.current_example:
            return
        
        # Update example values with new parameters
        updated_example = self._apply_parameters_to_example(
            self.current_example, params
        )
        
        # Re-render with updated values
        self._render_example(updated_example)
    
    def _apply_parameters_to_example(
        self, 
        example: NumericalExample, 
        params: Dict[str, float]
    ) -> NumericalExample:
        """Apply parameter changes to a numerical example."""
        # Create copy of example with updated input values
        updated_inputs = example.input_values.copy()
        
        for param_name, param_value in params.items():
            if param_name in updated_inputs:
                # Update scalar parameters
                if np.isscalar(updated_inputs[param_name]):
                    updated_inputs[param_name] = param_value
                else:
                    # For arrays, scale by parameter value
                    original_shape = updated_inputs[param_name].shape
                    updated_inputs[param_name] = (
                        updated_inputs[param_name] * param_value / 
                        np.mean(updated_inputs[param_name])
                    )
        
        # Recompute outputs (simplified - would need actual computation)
        updated_outputs = self._recompute_example_outputs(updated_inputs)
        
        # Create updated example
        updated_example = NumericalExample(
            example_id=example.example_id + "_updated",
            description=f"{example.description} (updated)",
            input_values=updated_inputs,
            computation_steps=example.computation_steps,  # Would need to recompute
            output_values=updated_outputs,
            visualization_data=example.visualization_data
        )
        
        return updated_example
    
    def _recompute_example_outputs(
        self, 
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Recompute example outputs based on updated inputs."""
        # Simplified computation - in practice, would use actual mathematical operations
        outputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                # Simple transformation for demonstration
                outputs[f"{key}_result"] = value * 2
            else:
                outputs[f"{key}_result"] = value * 2
        
        return outputs
    
    def _render_current_state(self) -> None:
        """Render the current example and visualization state."""
        self.main_ax.clear()
        
        if self.current_example:
            self._render_example(self.current_example)
        elif self.current_visualization:
            self._render_visualization(self.current_visualization)
        
        self.figure.canvas.draw_idle()
    
    def _render_example(self, example: NumericalExample) -> None:
        """Render a numerical example."""
        # Find matrix data to visualize
        matrix_data = None
        matrix_name = ""
        
        for name, value in example.output_values.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                matrix_data = value
                matrix_name = name
                break
        
        if matrix_data is None:
            # Try input values
            for name, value in example.input_values.items():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    matrix_data = value
                    matrix_name = name
                    break
        
        if matrix_data is not None:
            # Create color-coded matrix
            matrix = ColorCodedMatrix(
                matrix_data=matrix_data,
                color_mapping={"default": "viridis"}
            )
            
            # Render matrix
            im = self.main_ax.imshow(matrix_data, cmap='viridis', aspect='auto')
            self.main_ax.set_title(f"{matrix_name}: {example.description}")
            
            # Add colorbar
            plt.colorbar(im, ax=self.main_ax, shrink=0.8)
            
            # Setup hover effects if enabled
            if self.hover_manager and self.concept.equations:
                equation = self.concept.equations[0]
                self.hover_manager.create_matrix_hover_regions(matrix, equation.variables)
        
        else:
            # Render as text if no matrix data
            self.main_ax.text(
                0.5, 0.5, f"Example: {example.description}",
                ha='center', va='center',
                transform=self.main_ax.transAxes,
                fontsize=14
            )
            self.main_ax.set_xlim(0, 1)
            self.main_ax.set_ylim(0, 1)
    
    def _render_visualization(self, visualization) -> None:
        """Render a visualization."""
        self.main_ax.text(
            0.5, 0.5, f"Visualization: {visualization.title}",
            ha='center', va='center',
            transform=self.main_ax.transAxes,
            fontsize=14
        )
        self.main_ax.set_xlim(0, 1)
        self.main_ax.set_ylim(0, 1)
    
    def _render_operation_frame(
        self, 
        operation_viz: OperationVisualization, 
        frame_num: int
    ) -> None:
        """Render a specific frame of an operation animation."""
        self.main_ax.clear()
        
        # Determine which matrices to show
        all_matrices = (
            operation_viz.input_matrices + 
            operation_viz.intermediate_steps + 
            [operation_viz.output_matrix]
        )
        
        if frame_num < len(all_matrices):
            matrix = all_matrices[frame_num]
            
            # Render matrix
            im = self.main_ax.imshow(
                matrix.matrix_data, cmap='viridis', aspect='auto'
            )
            
            # Apply highlights
            for highlight in matrix.highlight_patterns:
                self._apply_highlight_to_axis(highlight)
            
            self.main_ax.set_title(f"Step {frame_num + 1}: {operation_viz.operation_type}")
            plt.colorbar(im, ax=self.main_ax, shrink=0.8)
    
    def _apply_highlight_to_axis(self, highlight) -> None:
        """Apply a highlight pattern to the main axis."""
        # Simplified highlight rendering
        for idx in highlight.indices:
            if len(idx) == 2:
                i, j = idx
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor=highlight.color,
                    linewidth=3
                )
                self.main_ax.add_patch(rect)
    
    def show(self) -> None:
        """Display the interactive visualization."""
        # Initial render
        self._render_current_state()
        
        # Show the figure
        plt.tight_layout()
        plt.show()
    
    def get_figure(self) -> plt.Figure:
        """Get the matplotlib figure for embedding in other applications."""
        return self.figure


def create_attention_interactive_demo() -> InteractiveMathVisualization:
    """Create interactive demonstration of attention mechanism."""
    
    # Create attention mechanism concept
    from ...core.models import Equation, Explanation, Visualization, VisualizationData
    
    # Define attention variables
    variables = {
        "Q": VariableDefinition(
            name="Q", description="Query matrix", data_type="matrix",
            shape=(4, 3), color_code="#FF6B6B"
        ),
        "K": VariableDefinition(
            name="K", description="Key matrix", data_type="matrix",
            shape=(4, 3), color_code="#4ECDC4"
        ),
        "V": VariableDefinition(
            name="V", description="Value matrix", data_type="matrix",
            shape=(4, 3), color_code="#45B7D1"
        ),
        "d_k": VariableDefinition(
            name="d_k", description="Key dimension", data_type="scalar",
            constraints="positive", color_code="#96CEB4"
        )
    }
    
    # Create attention equation
    attention_eq = Equation(
        equation_id="scaled_dot_product_attention",
        latex_expression=r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
        variables=variables
    )
    
    # Create sample numerical example
    Q = np.random.randn(4, 3)
    K = np.random.randn(4, 3)
    V = np.random.randn(4, 3)
    
    attention_example = NumericalExample(
        example_id="attention_demo",
        description="Scaled dot-product attention calculation",
        input_values={"Q": Q, "K": K, "V": V, "d_k": np.array(3.0)},
        computation_steps=[],
        output_values={"attention_output": np.random.randn(4, 3)}
    )
    
    # Create concept
    concept = MathematicalConcept(
        concept_id="attention_mechanism",
        title="Interactive Attention Mechanism Demonstration",
        prerequisites=["linear_algebra", "matrix_operations"],
        equations=[attention_eq],
        explanations=[],
        examples=[attention_example],
        visualizations=[],
        difficulty_level=3
    )
    
    # Create interactive visualization
    return InteractiveMathVisualization(concept)