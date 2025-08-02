"""
Core data models for mathematical concepts, equations, and examples.
Based on the design document specifications for the AI Math Tutorial system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


@dataclass
class VariableDefinition:
    """Definition of a mathematical variable with its properties."""
    name: str
    description: str
    data_type: str  # "scalar", "vector", "matrix", "tensor"
    shape: Optional[Tuple[int, ...]] = None
    constraints: Optional[str] = None  # e.g., "positive", "normalized"
    color_code: Optional[str] = None  # For visualization consistency


@dataclass
class DerivationStep:
    """A single step in a mathematical derivation."""
    step_number: int
    latex_expression: str
    explanation: str
    mathematical_justification: str
    intermediate_values: Optional[Dict[str, Any]] = None


@dataclass
class ComputationStep:
    """A computational step with input, operation, and output."""
    step_number: int
    operation_name: str
    input_values: Dict[str, np.ndarray]
    operation_description: str
    output_values: Dict[str, np.ndarray]
    visualization_hints: Optional[Dict[str, Any]] = None


@dataclass
class HighlightPattern:
    """Pattern for highlighting elements in visualizations."""
    pattern_type: str  # "row", "column", "element", "block"
    indices: List[Tuple[int, ...]]
    color: str
    description: str


@dataclass
class AnimationFrame:
    """Single frame in an animation sequence."""
    frame_number: int
    matrix_state: np.ndarray
    highlights: List[HighlightPattern]
    description: str
    duration_ms: int = 1000


@dataclass
class VisualizationData:
    """Data structure for visualization components."""
    visualization_type: str
    data: Dict[str, Any]
    color_mappings: Dict[str, str]
    interactive_elements: List[str] = field(default_factory=list)


@dataclass
class ColorCodedMatrix:
    """Matrix with color coding for visualization."""
    matrix_data: np.ndarray
    color_mapping: Dict[str, str]
    element_labels: Optional[Dict[Tuple[int, int], str]] = None
    highlight_patterns: List[HighlightPattern] = field(default_factory=list)


@dataclass
class OperationVisualization:
    """Visualization of a mathematical operation."""
    operation_type: str  # "matrix_multiply", "attention", "convolution"
    input_matrices: List[ColorCodedMatrix]
    intermediate_steps: List[ColorCodedMatrix]
    output_matrix: ColorCodedMatrix
    animation_sequence: List[AnimationFrame] = field(default_factory=list)


@dataclass
class Equation:
    """Mathematical equation with all associated metadata."""
    equation_id: str
    latex_expression: str
    variables: Dict[str, VariableDefinition]
    derivation_steps: List[DerivationStep] = field(default_factory=list)
    mathematical_properties: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    complexity_level: int = 1  # 1-5 scale


@dataclass
class NumericalExample:
    """Concrete numerical example with computation steps."""
    example_id: str
    description: str
    input_values: Dict[str, np.ndarray]
    computation_steps: List[ComputationStep]
    output_values: Dict[str, np.ndarray]
    visualization_data: Optional[VisualizationData] = None
    educational_notes: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    """Textual explanation of a mathematical concept."""
    explanation_type: str  # "intuitive", "formal", "historical", "practical"
    content: str
    mathematical_level: int  # 1-5 scale
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class Visualization:
    """Visual representation of a mathematical concept."""
    visualization_id: str
    visualization_type: str
    title: str
    description: str
    data: VisualizationData
    interactive: bool = False
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class MathematicalConcept:
    """Core entity representing a mathematical concept in the tutorial."""
    concept_id: str
    title: str
    prerequisites: List[str]
    equations: List[Equation]
    explanations: List[Explanation]
    examples: List[NumericalExample]
    visualizations: List[Visualization]
    difficulty_level: int  # 1-5 scale
    learning_objectives: List[str] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)


@dataclass
class TutorialChapter:
    """A chapter in the tutorial containing multiple concepts."""
    chapter_id: str
    title: str
    concepts: List[MathematicalConcept]
    introduction: str
    summary: str
    chapter_number: int
    estimated_time_minutes: int = 60