"""
Interactive visualization components for mathematical demonstrations.
Provides parameter widgets, hover effects, and animation controls.
"""

from .parameter_widgets import (
    InteractiveParameterWidget,
    MathematicalParameterWidget,
    ParameterConfig,
    create_attention_parameter_widget
)

from .hover_effects import (
    HoverEffectManager,
    VariableHighlighter,
    HoverRegion,
    HighlightStyle,
    create_attention_hover_demo
)

from .animation_controls import (
    AnimationController,
    StepByStepAnimationController,
    AnimationControlConfig,
    create_matrix_multiplication_demo
)

from .interactive_components import (
    InteractiveMathVisualization,
    InteractiveVisualizationConfig,
    create_attention_interactive_demo
)

__all__ = [
    # Parameter widgets
    'InteractiveParameterWidget',
    'MathematicalParameterWidget', 
    'ParameterConfig',
    'create_attention_parameter_widget',
    
    # Hover effects
    'HoverEffectManager',
    'VariableHighlighter',
    'HoverRegion',
    'HighlightStyle',
    'create_attention_hover_demo',
    
    # Animation controls
    'AnimationController',
    'StepByStepAnimationController',
    'AnimationControlConfig',
    'create_matrix_multiplication_demo',
    
    # Main interactive components
    'InteractiveMathVisualization',
    'InteractiveVisualizationConfig',
    'create_attention_interactive_demo'
]