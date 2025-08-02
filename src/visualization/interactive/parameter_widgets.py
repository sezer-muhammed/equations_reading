"""
Parameter manipulation widgets for real-time mathematical visualization updates.
Implements interactive controls for equation parameters with live feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass

from ...core.models import VariableDefinition, NumericalExample, VisualizationData


@dataclass
class ParameterConfig:
    """Configuration for a parameter widget."""
    name: str
    display_name: str
    min_value: float
    max_value: float
    initial_value: float
    step_size: float = 0.01
    widget_type: str = "slider"  # "slider", "button", "checkbox", "radio"
    options: Optional[List[str]] = None  # For radio/checkbox widgets


class InteractiveParameterWidget:
    """Creates interactive widgets for manipulating mathematical parameters."""
    
    def __init__(
        self,
        parameters: Dict[str, ParameterConfig],
        update_callback: Callable[[Dict[str, float]], None],
        figure_size: Tuple[float, float] = (12, 8)
    ):
        self.parameters = parameters
        self.update_callback = update_callback
        self.current_values = {name: config.initial_value 
                             for name, config in parameters.items()}
        
        # Create figure with space for widgets
        self.fig = plt.figure(figsize=figure_size)
        self.main_ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        
        # Widget storage
        self.widgets = {}
        self.widget_axes = {}
        
        self._create_widgets()
        self._setup_callbacks()
    
    def _create_widgets(self) -> None:
        """Create all parameter widgets."""
        widget_positions = self._calculate_widget_positions()
        
        for i, (param_name, config) in enumerate(self.parameters.items()):
            if i >= len(widget_positions):
                break
                
            pos = widget_positions[i]
            
            if config.widget_type == "slider":
                self._create_slider_widget(param_name, config, pos)
            elif config.widget_type == "button":
                self._create_button_widget(param_name, config, pos)
            elif config.widget_type == "checkbox":
                self._create_checkbox_widget(param_name, config, pos)
            elif config.widget_type == "radio":
                self._create_radio_widget(param_name, config, pos)
    
    def _calculate_widget_positions(self) -> List[Tuple[float, float, float, float]]:
        """Calculate positions for widget placement."""
        positions = []
        
        # Right side panel for widgets
        widget_width = 0.2
        widget_height = 0.03
        start_x = 0.78
        start_y = 0.85
        spacing = 0.08
        
        for i in range(len(self.parameters)):
            y_pos = start_y - i * spacing
            positions.append((start_x, y_pos, widget_width, widget_height))
        
        return positions
    
    def _create_slider_widget(
        self, 
        param_name: str, 
        config: ParameterConfig,
        position: Tuple[float, float, float, float]
    ) -> None:
        """Create a slider widget for continuous parameters."""
        ax = plt.axes(position)
        self.widget_axes[param_name] = ax
        
        slider = Slider(
            ax, config.display_name,
            config.min_value, config.max_value,
            valinit=config.initial_value,
            valfmt='%.3f'
        )
        
        self.widgets[param_name] = slider
    
    def _create_button_widget(
        self, 
        param_name: str, 
        config: ParameterConfig,
        position: Tuple[float, float, float, float]
    ) -> None:
        """Create button widgets for discrete actions."""
        ax = plt.axes(position)
        self.widget_axes[param_name] = ax
        
        button = Button(ax, config.display_name)
        self.widgets[param_name] = button
    
    def _create_checkbox_widget(
        self, 
        param_name: str, 
        config: ParameterConfig,
        position: Tuple[float, float, float, float]
    ) -> None:
        """Create checkbox widgets for boolean parameters."""
        ax = plt.axes(position)
        self.widget_axes[param_name] = ax
        
        options = config.options or [config.display_name]
        checkbox = CheckButtons(ax, options, [config.initial_value > 0])
        self.widgets[param_name] = checkbox
    
    def _create_radio_widget(
        self, 
        param_name: str, 
        config: ParameterConfig,
        position: Tuple[float, float, float, float]
    ) -> None:
        """Create radio button widgets for categorical parameters."""
        ax = plt.axes(position)
        self.widget_axes[param_name] = ax
        
        options = config.options or ["Option 1", "Option 2"]
        radio = RadioButtons(ax, options)
        self.widgets[param_name] = radio
    
    def _setup_callbacks(self) -> None:
        """Setup callbacks for all widgets."""
        for param_name, widget in self.widgets.items():
            config = self.parameters[param_name]
            
            if config.widget_type == "slider":
                widget.on_changed(lambda val, name=param_name: self._on_slider_change(name, val))
            elif config.widget_type == "button":
                widget.on_clicked(lambda event, name=param_name: self._on_button_click(name))
            elif config.widget_type == "checkbox":
                widget.on_clicked(lambda label, name=param_name: self._on_checkbox_change(name, label))
            elif config.widget_type == "radio":
                widget.on_clicked(lambda label, name=param_name: self._on_radio_change(name, label))
    
    def _on_slider_change(self, param_name: str, value: float) -> None:
        """Handle slider value changes."""
        self.current_values[param_name] = value
        self.update_callback(self.current_values.copy())
    
    def _on_button_click(self, param_name: str) -> None:
        """Handle button clicks."""
        # Toggle or increment value
        config = self.parameters[param_name]
        current = self.current_values[param_name]
        
        if current >= config.max_value:
            self.current_values[param_name] = config.min_value
        else:
            self.current_values[param_name] = min(
                current + config.step_size, config.max_value
            )
        
        self.update_callback(self.current_values.copy())
    
    def _on_checkbox_change(self, param_name: str, label: str) -> None:
        """Handle checkbox changes."""
        # Toggle boolean value
        self.current_values[param_name] = 1.0 - self.current_values[param_name]
        self.update_callback(self.current_values.copy())
    
    def _on_radio_change(self, param_name: str, label: str) -> None:
        """Handle radio button changes."""
        config = self.parameters[param_name]
        if config.options and label in config.options:
            self.current_values[param_name] = float(config.options.index(label))
        
        self.update_callback(self.current_values.copy())
    
    def get_main_axis(self) -> plt.Axes:
        """Get the main plotting axis for visualizations."""
        return self.main_ax
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current parameter values."""
        return self.current_values.copy()
    
    def update_parameter(self, param_name: str, value: float) -> None:
        """Programmatically update a parameter value."""
        if param_name in self.current_values:
            self.current_values[param_name] = value
            
            # Update widget display
            if param_name in self.widgets:
                widget = self.widgets[param_name]
                config = self.parameters[param_name]
                
                if config.widget_type == "slider":
                    widget.set_val(value)
    
    def reset_parameters(self) -> None:
        """Reset all parameters to initial values."""
        for param_name, config in self.parameters.items():
            self.update_parameter(param_name, config.initial_value)
        
        self.update_callback(self.current_values.copy())


class MathematicalParameterWidget(InteractiveParameterWidget):
    """Specialized widget for mathematical equation parameters."""
    
    def __init__(
        self,
        equation_variables: Dict[str, VariableDefinition],
        update_callback: Callable[[Dict[str, float]], None],
        figure_size: Tuple[float, float] = (12, 8)
    ):
        # Convert VariableDefinition to ParameterConfig
        parameters = self._convert_variables_to_parameters(equation_variables)
        super().__init__(parameters, update_callback, figure_size)
        
        self.equation_variables = equation_variables
    
    def _convert_variables_to_parameters(
        self, 
        variables: Dict[str, VariableDefinition]
    ) -> Dict[str, ParameterConfig]:
        """Convert equation variables to parameter configurations."""
        parameters = {}
        
        for name, var_def in variables.items():
            # Determine parameter range based on variable type and constraints
            min_val, max_val, initial_val = self._get_parameter_range(var_def)
            
            parameters[name] = ParameterConfig(
                name=name,
                display_name=var_def.name,
                min_value=min_val,
                max_value=max_val,
                initial_value=initial_val,
                step_size=0.01 if var_def.data_type == "scalar" else 0.1
            )
        
        return parameters
    
    def _get_parameter_range(
        self, 
        var_def: VariableDefinition
    ) -> Tuple[float, float, float]:
        """Determine appropriate parameter range for a variable."""
        if var_def.constraints:
            if "positive" in var_def.constraints.lower():
                return 0.01, 10.0, 1.0
            elif "normalized" in var_def.constraints.lower():
                return 0.0, 1.0, 0.5
            elif "probability" in var_def.constraints.lower():
                return 0.0, 1.0, 0.5
        
        # Default ranges based on data type
        if var_def.data_type == "scalar":
            return -5.0, 5.0, 1.0
        elif var_def.data_type in ["vector", "matrix"]:
            return -2.0, 2.0, 1.0
        else:
            return -1.0, 1.0, 0.0


def create_attention_parameter_widget() -> MathematicalParameterWidget:
    """Create parameter widget for attention mechanism demonstration."""
    
    # Define attention mechanism variables
    variables = {
        "d_model": VariableDefinition(
            name="d_model", 
            description="Model dimension",
            data_type="scalar",
            constraints="positive"
        ),
        "temperature": VariableDefinition(
            name="temperature", 
            description="Attention temperature",
            data_type="scalar", 
            constraints="positive"
        ),
        "dropout_rate": VariableDefinition(
            name="dropout_rate",
            description="Dropout probability", 
            data_type="scalar",
            constraints="probability"
        )
    }
    
    def update_attention_visualization(params: Dict[str, float]) -> None:
        """Update attention visualization with new parameters."""
        print(f"Updating attention with parameters: {params}")
        # Implementation would update the actual visualization
    
    return MathematicalParameterWidget(variables, update_attention_visualization)