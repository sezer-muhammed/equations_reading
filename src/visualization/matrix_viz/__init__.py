"""
Matrix visualization components for color-coded mathematical displays.
Provides consistent visual representations across related concepts.
"""

from .color_coded_matrix import (
    ColorCodedMatrixVisualizer,
    MatrixVisualizationStyle,
    create_sample_matrix
)
from .matrix_animations import (
    MatrixAnimationEngine,
    AnimationConfig,
    create_sample_multiplication_animation
)
from .color_schemes import (
    ColorSchemeManager,
    ConceptColorScheme,
    ConceptType,
    ColorMapping,
    get_concept_colors,
    get_consistent_color,
    color_manager
)

__all__ = [
    # Core visualizer
    'ColorCodedMatrixVisualizer',
    'MatrixVisualizationStyle',
    'create_sample_matrix',
    
    # Animation engine
    'MatrixAnimationEngine', 
    'AnimationConfig',
    'create_sample_multiplication_animation',
    
    # Color schemes
    'ColorSchemeManager',
    'ConceptColorScheme',
    'ConceptType',
    'ColorMapping',
    'get_concept_colors',
    'get_consistent_color',
    'color_manager'
]