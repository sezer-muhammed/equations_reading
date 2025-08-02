"""
Animation controls for step-by-step mathematical demonstrations.
Provides play/pause/step controls for educational visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from typing import List, Dict, Callable, Optional, Any, Tuple
from dataclasses import dataclass

from ...core.models import AnimationFrame, OperationVisualization, ColorCodedMatrix, HighlightPattern


@dataclass
class AnimationControlConfig:
    """Configuration for animation controls."""
    show_play_pause: bool = True
    show_step_controls: bool = True
    show_speed_control: bool = True
    show_frame_slider: bool = True
    show_reset_button: bool = True
    control_panel_height: float = 0.15


class AnimationController:
    """Controls for interactive mathematical animations."""
    
    def __init__(
        self,
        figure: plt.Figure,
        animation_frames: List[AnimationFrame],
        render_callback: Callable[[int], None],
        config: Optional[AnimationControlConfig] = None
    ):
        self.figure = figure
        self.frames = animation_frames
        self.render_callback = render_callback
        self.config = config or AnimationControlConfig()
        
        # Animation state
        self.current_frame = 0
        self.is_playing = False
        self.animation_speed = 1.0
        self.animation_obj = None
        
        # Control widgets
        self.controls = {}
        self.control_axes = {}
        
        # Setup control panel
        self._create_control_panel()
        self._setup_controls()
    
    def _create_control_panel(self) -> None:
        """Create the control panel at the bottom of the figure."""
        # Adjust main plot area to make room for controls
        self.figure.subplots_adjust(bottom=self.config.control_panel_height + 0.05)
        
        # Calculate control positions
        panel_bottom = 0.02
        panel_height = self.config.control_panel_height
        
        self.control_positions = self._calculate_control_positions(
            panel_bottom, panel_height
        )
    
    def _calculate_control_positions(
        self, 
        bottom: float, 
        height: float
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Calculate positions for control widgets."""
        positions = {}
        
        # Control layout: [Play/Pause] [Step Back] [Step Forward] [Speed] [Frame Slider] [Reset]
        button_width = 0.08
        button_height = height * 0.6
        slider_width = 0.3
        spacing = 0.02
        
        current_x = 0.05
        button_y = bottom + (height - button_height) / 2
        
        if self.config.show_play_pause:
            positions['play_pause'] = (current_x, button_y, button_width, button_height)
            current_x += button_width + spacing
        
        if self.config.show_step_controls:
            positions['step_back'] = (current_x, button_y, button_width, button_height)
            current_x += button_width + spacing
            
            positions['step_forward'] = (current_x, button_y, button_width, button_height)
            current_x += button_width + spacing
        
        if self.config.show_speed_control:
            positions['speed'] = (current_x, button_y, button_width * 1.5, button_height)
            current_x += button_width * 1.5 + spacing
        
        if self.config.show_frame_slider:
            positions['frame_slider'] = (current_x, button_y, slider_width, button_height)
            current_x += slider_width + spacing
        
        if self.config.show_reset_button:
            positions['reset'] = (current_x, button_y, button_width, button_height)
        
        return positions
    
    def _setup_controls(self) -> None:
        """Create and setup all control widgets."""
        
        if self.config.show_play_pause:
            self._create_play_pause_button()
        
        if self.config.show_step_controls:
            self._create_step_buttons()
        
        if self.config.show_speed_control:
            self._create_speed_control()
        
        if self.config.show_frame_slider:
            self._create_frame_slider()
        
        if self.config.show_reset_button:
            self._create_reset_button()
    
    def _create_play_pause_button(self) -> None:
        """Create play/pause button."""
        pos = self.control_positions['play_pause']
        ax = plt.axes(pos)
        
        button = Button(ax, 'Play')
        button.on_clicked(self._on_play_pause_click)
        
        self.controls['play_pause'] = button
        self.control_axes['play_pause'] = ax
    
    def _create_step_buttons(self) -> None:
        """Create step forward/backward buttons."""
        # Step backward button
        pos = self.control_positions['step_back']
        ax_back = plt.axes(pos)
        button_back = Button(ax_back, '◀')
        button_back.on_clicked(self._on_step_back_click)
        
        self.controls['step_back'] = button_back
        self.control_axes['step_back'] = ax_back
        
        # Step forward button
        pos = self.control_positions['step_forward']
        ax_forward = plt.axes(pos)
        button_forward = Button(ax_forward, '▶')
        button_forward.on_clicked(self._on_step_forward_click)
        
        self.controls['step_forward'] = button_forward
        self.control_axes['step_forward'] = ax_forward
    
    def _create_speed_control(self) -> None:
        """Create animation speed control slider."""
        pos = self.control_positions['speed']
        ax = plt.axes(pos)
        
        slider = Slider(
            ax, 'Speed', 0.1, 3.0, 
            valinit=self.animation_speed,
            valfmt='%.1fx'
        )
        slider.on_changed(self._on_speed_change)
        
        self.controls['speed'] = slider
        self.control_axes['speed'] = ax
    
    def _create_frame_slider(self) -> None:
        """Create frame position slider."""
        pos = self.control_positions['frame_slider']
        ax = plt.axes(pos)
        
        max_frame = max(0, len(self.frames) - 1)
        slider = Slider(
            ax, 'Frame', 0, max_frame,
            valinit=self.current_frame,
            valfmt='%d',
            valstep=1
        )
        slider.on_changed(self._on_frame_change)
        
        self.controls['frame_slider'] = slider
        self.control_axes['frame_slider'] = ax
    
    def _create_reset_button(self) -> None:
        """Create reset button."""
        pos = self.control_positions['reset']
        ax = plt.axes(pos)
        
        button = Button(ax, 'Reset')
        button.on_clicked(self._on_reset_click)
        
        self.controls['reset'] = button
        self.control_axes['reset'] = ax
    
    def _on_play_pause_click(self, event) -> None:
        """Handle play/pause button click."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def _on_step_back_click(self, event) -> None:
        """Handle step backward button click."""
        self.step_backward()
    
    def _on_step_forward_click(self, event) -> None:
        """Handle step forward button click."""
        self.step_forward()
    
    def _on_speed_change(self, value: float) -> None:
        """Handle speed slider change."""
        self.animation_speed = value
        
        # Update animation interval if playing
        if self.animation_obj and self.is_playing:
            self.animation_obj.event_source.interval = self._get_frame_interval()
    
    def _on_frame_change(self, value: float) -> None:
        """Handle frame slider change."""
        new_frame = int(value)
        if new_frame != self.current_frame:
            self.goto_frame(new_frame)
    
    def _on_reset_click(self, event) -> None:
        """Handle reset button click."""
        self.reset()
    
    def play(self) -> None:
        """Start animation playback."""
        if not self.is_playing:
            self.is_playing = True
            self.controls['play_pause'].label.set_text('Pause')
            
            # Create animation if not exists
            if not self.animation_obj:
                self._create_animation()
            
            # Resume animation
            self.animation_obj.resume()
            self.figure.canvas.draw_idle()
    
    def pause(self) -> None:
        """Pause animation playback."""
        if self.is_playing:
            self.is_playing = False
            self.controls['play_pause'].label.set_text('Play')
            
            if self.animation_obj:
                self.animation_obj.pause()
            
            self.figure.canvas.draw_idle()
    
    def step_forward(self) -> None:
        """Step to next frame."""
        if self.current_frame < len(self.frames) - 1:
            self.goto_frame(self.current_frame + 1)
    
    def step_backward(self) -> None:
        """Step to previous frame."""
        if self.current_frame > 0:
            self.goto_frame(self.current_frame - 1)
    
    def goto_frame(self, frame_number: int) -> None:
        """Go to specific frame."""
        frame_number = max(0, min(frame_number, len(self.frames) - 1))
        
        if frame_number != self.current_frame:
            self.current_frame = frame_number
            
            # Update frame slider
            if 'frame_slider' in self.controls:
                self.controls['frame_slider'].set_val(frame_number)
            
            # Render frame
            self.render_callback(frame_number)
            self.figure.canvas.draw_idle()
    
    def reset(self) -> None:
        """Reset animation to beginning."""
        self.pause()
        self.goto_frame(0)
    
    def _create_animation(self) -> None:
        """Create the matplotlib animation object."""
        def animate_func(frame_num: int) -> None:
            if frame_num < len(self.frames):
                self.current_frame = frame_num
                
                # Update frame slider
                if 'frame_slider' in self.controls:
                    self.controls['frame_slider'].set_val(frame_num)
                
                # Render frame
                self.render_callback(frame_num)
        
        self.animation_obj = animation.FuncAnimation(
            self.figure, animate_func,
            frames=len(self.frames),
            interval=self._get_frame_interval(),
            repeat=True,
            blit=False
        )
    
    def _get_frame_interval(self) -> int:
        """Get frame interval in milliseconds based on speed."""
        base_interval = 1000  # 1 second per frame
        return int(base_interval / self.animation_speed)
    
    def get_current_frame(self) -> int:
        """Get current frame number."""
        return self.current_frame
    
    def get_total_frames(self) -> int:
        """Get total number of frames."""
        return len(self.frames)
    
    def is_animation_playing(self) -> bool:
        """Check if animation is currently playing."""
        return self.is_playing


class StepByStepAnimationController(AnimationController):
    """Specialized controller for step-by-step mathematical demonstrations."""
    
    def __init__(
        self,
        figure: plt.Figure,
        operation_visualization: OperationVisualization,
        render_callback: Callable[[int], None],
        config: Optional[AnimationControlConfig] = None
    ):
        # Convert operation visualization to animation frames
        frames = self._create_frames_from_operation(operation_visualization)
        
        super().__init__(figure, frames, render_callback, config)
        
        self.operation_viz = operation_visualization
        self.step_descriptions = self._create_step_descriptions()
        
        # Add step description display
        self._create_step_description_display()
    
    def _create_frames_from_operation(
        self, 
        operation_viz: OperationVisualization
    ) -> List[AnimationFrame]:
        """Convert operation visualization to animation frames."""
        frames = []
        
        # Create frames for each step in the operation
        all_matrices = (
            operation_viz.input_matrices + 
            operation_viz.intermediate_steps + 
            [operation_viz.output_matrix]
        )
        
        for i, matrix in enumerate(all_matrices):
            frame = AnimationFrame(
                frame_number=i,
                matrix_state=matrix.matrix_data,
                highlights=matrix.highlight_patterns,
                description=f"Step {i + 1}: {self._get_step_description(i, operation_viz)}",
                duration_ms=1500
            )
            frames.append(frame)
        
        return frames
    
    def _get_step_description(
        self, 
        step: int, 
        operation_viz: OperationVisualization
    ) -> str:
        """Get description for a specific step."""
        operation_type = operation_viz.operation_type
        
        if operation_type == "matrix_multiply":
            if step == 0:
                return "Input matrices A and B"
            elif step < len(operation_viz.intermediate_steps) + 1:
                return f"Computing element-wise products (step {step})"
            else:
                return "Final result matrix C = A × B"
        
        elif operation_type == "attention":
            if step == 0:
                return "Query (Q) and Key (K) matrices"
            elif step == 1:
                return "Computing attention scores: Q × K^T"
            elif step == 2:
                return "Applying softmax to get attention weights"
            elif step == 3:
                return "Computing weighted sum with Values (V)"
            else:
                return "Final attention output"
        
        else:
            return f"Step {step + 1}"
    
    def _create_step_descriptions(self) -> List[str]:
        """Create detailed descriptions for each step."""
        descriptions = []
        
        for frame in self.frames:
            descriptions.append(frame.description)
        
        return descriptions
    
    def _create_step_description_display(self) -> None:
        """Create text display for step descriptions."""
        # Add text area above controls
        text_ax = plt.axes((0.1, self.config.control_panel_height + 0.06, 0.8, 0.04))
        text_ax.set_xlim(0, 1)
        text_ax.set_ylim(0, 1)
        text_ax.axis('off')
        
        # Create text object
        self.description_text = text_ax.text(
            0.5, 0.5, self.step_descriptions[0] if self.step_descriptions else "",
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
        )
        
        self.control_axes['description'] = text_ax
    
    def goto_frame(self, frame_number: int) -> None:
        """Override to update step description."""
        super().goto_frame(frame_number)
        
        # Update step description
        if hasattr(self, 'description_text') and frame_number < len(self.step_descriptions):
            self.description_text.set_text(self.step_descriptions[frame_number])


def create_matrix_multiplication_demo() -> Tuple[plt.Figure, StepByStepAnimationController]:
    """Create demonstration of step-by-step matrix multiplication."""
    
    # Create sample matrices
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    result = np.matmul(matrix_a, matrix_b)
    
    # Create operation visualization
    input_a = ColorCodedMatrix(matrix_a, {"input": "#3498DB"})
    input_b = ColorCodedMatrix(matrix_b, {"input": "#E74C3C"})
    output = ColorCodedMatrix(result, {"output": "#27AE60"})
    
    # Create intermediate steps (simplified)
    intermediate = ColorCodedMatrix(
        np.zeros_like(result), 
        {"intermediate": "#F39C12"},
        highlight_patterns=[
            HighlightPattern("element", [(0, 0)], "yellow", "Computing element (0,0)")
        ]
    )
    
    operation_viz = OperationVisualization(
        operation_type="matrix_multiply",
        input_matrices=[input_a, input_b],
        intermediate_steps=[intermediate],
        output_matrix=output
    )
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    main_ax = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=3)
    
    def render_frame(frame_num: int) -> None:
        """Render a specific frame of the animation."""
        main_ax.clear()
        
        if frame_num < len(operation_viz.input_matrices):
            # Show input matrices
            matrix = operation_viz.input_matrices[frame_num]
            main_ax.imshow(matrix.matrix_data, cmap='viridis')
            main_ax.set_title(f"Input Matrix {frame_num + 1}")
        
        elif frame_num < len(operation_viz.input_matrices) + len(operation_viz.intermediate_steps):
            # Show intermediate steps
            idx = frame_num - len(operation_viz.input_matrices)
            matrix = operation_viz.intermediate_steps[idx]
            main_ax.imshow(matrix.matrix_data, cmap='viridis')
            main_ax.set_title("Intermediate Calculation")
        
        else:
            # Show final result
            main_ax.imshow(operation_viz.output_matrix.matrix_data, cmap='viridis')
            main_ax.set_title("Final Result")
    
    # Create animation controller
    controller = StepByStepAnimationController(fig, operation_viz, render_frame)
    
    # Initial render
    render_frame(0)
    
    return fig, controller