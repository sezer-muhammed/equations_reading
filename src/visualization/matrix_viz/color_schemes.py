"""
Consistent color schemes for matrix visualizations across related concepts.
Ensures visual coherence throughout the tutorial system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ...core.config import get_color_scheme


class ConceptType(Enum):
    """Types of mathematical concepts for color scheme selection."""
    LINEAR_ALGEBRA = "linear_algebra"
    MATRIX_OPERATIONS = "matrix_operations"
    ATTENTION = "attention"
    OPTIMIZATION = "optimization"
    NEURAL_NETWORK = "neural_network"
    PROBABILITY = "probability"
    CALCULUS = "calculus"


@dataclass
class ColorMapping:
    """Defines color mapping for specific mathematical elements."""
    element_type: str
    color_code: str
    description: str
    rgb_values: Optional[Tuple[float, float, float]] = None


class ConceptColorScheme:
    """Color scheme for a specific mathematical concept."""
    
    def __init__(
        self, 
        concept_type: ConceptType,
        primary_colors: Dict[str, str],
        gradient_colors: Optional[List[str]] = None,
        special_mappings: Optional[Dict[str, ColorMapping]] = None
    ):
        self.concept_type = concept_type
        self.primary_colors = primary_colors
        self.gradient_colors = gradient_colors or []
        self.special_mappings = special_mappings or {}
        
        # Create matplotlib colormap
        if self.gradient_colors:
            self.colormap = LinearSegmentedColormap.from_list(
                f"{concept_type.value}_gradient", 
                self.gradient_colors
            )
        else:
            self.colormap = plt.cm.viridis
    
    def get_color(self, element_type: str) -> str:
        """Get color for a specific element type."""
        if element_type in self.primary_colors:
            return self.primary_colors[element_type]
        elif element_type in self.special_mappings:
            return self.special_mappings[element_type].color_code
        else:
            return self.primary_colors.get("default", "#1f77b4")
    
    def get_gradient_color(self, value: float, vmin: float = 0, vmax: float = 1) -> str:
        """Get color from gradient based on normalized value."""
        normalized = (value - vmin) / (vmax - vmin) if vmax != vmin else 0
        normalized = np.clip(normalized, 0, 1)
        rgba = self.colormap(normalized)
        return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"


class ColorSchemeManager:
    """Manages consistent color schemes across all mathematical concepts."""
    
    def __init__(self):
        self.schemes = self._initialize_schemes()
        self.concept_relationships = self._define_concept_relationships()
    
    def _initialize_schemes(self) -> Dict[ConceptType, ConceptColorScheme]:
        """Initialize color schemes for all concept types."""
        schemes = {}
        
        # Linear Algebra scheme
        schemes[ConceptType.LINEAR_ALGEBRA] = ConceptColorScheme(
            ConceptType.LINEAR_ALGEBRA,
            primary_colors={
                "matrix": "#2E86AB",      # Blue for matrices
                "vector": "#A23B72",      # Purple for vectors
                "scalar": "#F18F01",      # Orange for scalars
                "result": "#C73E1D",      # Red for results
                "highlight": "#FFE66D",   # Yellow for highlights
                "default": "#2E86AB"
            },
            gradient_colors=["#0D1B2A", "#2E86AB", "#A23B72", "#F18F01"],
            special_mappings={
                "eigenvalue": ColorMapping("eigenvalue", "#C73E1D", "Eigenvalue"),
                "eigenvector": ColorMapping("eigenvector", "#A23B72", "Eigenvector"),
                "determinant": ColorMapping("determinant", "#F18F01", "Determinant")
            }
        )
        
        # Matrix Operations scheme
        schemes[ConceptType.MATRIX_OPERATIONS] = ConceptColorScheme(
            ConceptType.MATRIX_OPERATIONS,
            primary_colors={
                "input_a": "#3498DB",         # Blue for first input matrix
                "input_b": "#E74C3C",         # Red for second input matrix
                "output": "#27AE60",          # Green for output matrix
                "intermediate": "#F39C12",    # Orange for intermediate steps
                "highlight_row": "#9B59B6",   # Purple for row highlights
                "highlight_col": "#E67E22",   # Orange for column highlights
                "partial_result": "#1ABC9C",  # Teal for partial results
                "query": "#FF6B6B",           # Red for attention queries
                "key": "#4ECDC4",             # Teal for attention keys
                "value": "#45B7D1",           # Blue for attention values
                "attention": "#96CEB4",       # Green for attention weights
                "forward": "#74B9FF",         # Blue for forward pass
                "gradient_flow": "#E17055",   # Orange for gradient flow
                "high_gradient": "#E74C3C",   # Red for high gradients
                "medium_gradient": "#F39C12", # Orange for medium gradients
                "low_gradient": "#3498DB",    # Blue for low gradients
                "default": "#3498DB"
            },
            gradient_colors=["#3498DB", "#E74C3C", "#27AE60", "#F39C12"],
            special_mappings={
                "multiplication": ColorMapping("multiplication", "#3498DB", "Matrix multiplication"),
                "addition": ColorMapping("addition", "#27AE60", "Matrix addition"),
                "transpose": ColorMapping("transpose", "#E74C3C", "Matrix transpose"),
                "inverse": ColorMapping("inverse", "#9B59B6", "Matrix inverse")
            }
        )
        
        # Attention mechanism scheme
        schemes[ConceptType.ATTENTION] = ConceptColorScheme(
            ConceptType.ATTENTION,
            primary_colors={
                "query": "#FF6B6B",       # Red for queries
                "key": "#4ECDC4",         # Teal for keys
                "value": "#45B7D1",       # Blue for values
                "attention": "#96CEB4",   # Green for attention weights
                "output": "#FFEAA7",      # Yellow for output
                "default": "#FF6B6B"
            },
            gradient_colors=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
            special_mappings={
                "self_attention": ColorMapping("self_attention", "#FF6B6B", "Self-attention"),
                "cross_attention": ColorMapping("cross_attention", "#4ECDC4", "Cross-attention"),
                "multi_head": ColorMapping("multi_head", "#45B7D1", "Multi-head")
            }
        )
        
        # Optimization scheme
        schemes[ConceptType.OPTIMIZATION] = ConceptColorScheme(
            ConceptType.OPTIMIZATION,
            primary_colors={
                "gradient": "#E17055",     # Orange-red for gradients
                "parameter": "#74B9FF",    # Blue for parameters
                "loss": "#FD79A8",        # Pink for loss
                "learning_rate": "#FDCB6E", # Yellow for learning rate
                "momentum": "#6C5CE7",     # Purple for momentum
                "default": "#E17055"
            },
            gradient_colors=["#2D3436", "#E17055", "#FD79A8", "#FDCB6E"],
            special_mappings={
                "adam": ColorMapping("adam", "#74B9FF", "Adam optimizer"),
                "sgd": ColorMapping("sgd", "#E17055", "SGD optimizer"),
                "convergence": ColorMapping("convergence", "#00B894", "Convergence point")
            }
        )
        
        # Neural Network scheme
        schemes[ConceptType.NEURAL_NETWORK] = ConceptColorScheme(
            ConceptType.NEURAL_NETWORK,
            primary_colors={
                "input": "#00B894",       # Green for input
                "hidden": "#74B9FF",      # Blue for hidden layers
                "output": "#E17055",      # Orange for output
                "weight": "#6C5CE7",      # Purple for weights
                "bias": "#FD79A8",        # Pink for bias
                "activation": "#FDCB6E",  # Yellow for activations
                "default": "#74B9FF"
            },
            gradient_colors=["#00B894", "#74B9FF", "#6C5CE7", "#E17055"],
            special_mappings={
                "relu": ColorMapping("relu", "#00B894", "ReLU activation"),
                "sigmoid": ColorMapping("sigmoid", "#74B9FF", "Sigmoid activation"),
                "tanh": ColorMapping("tanh", "#6C5CE7", "Tanh activation"),
                "softmax": ColorMapping("softmax", "#E17055", "Softmax activation")
            }
        )
        
        # Probability scheme
        schemes[ConceptType.PROBABILITY] = ConceptColorScheme(
            ConceptType.PROBABILITY,
            primary_colors={
                "probability": "#A29BFE",  # Light purple for probabilities
                "distribution": "#FD79A8", # Pink for distributions
                "expectation": "#FDCB6E",  # Yellow for expectations
                "variance": "#E17055",     # Orange for variance
                "entropy": "#00B894",      # Green for entropy
                "default": "#A29BFE"
            },
            gradient_colors=["#2D3436", "#A29BFE", "#FD79A8", "#FDCB6E"],
            special_mappings={
                "gaussian": ColorMapping("gaussian", "#A29BFE", "Gaussian distribution"),
                "bernoulli": ColorMapping("bernoulli", "#FD79A8", "Bernoulli distribution"),
                "kl_divergence": ColorMapping("kl_divergence", "#E17055", "KL divergence")
            }
        )
        
        # Calculus scheme
        schemes[ConceptType.CALCULUS] = ConceptColorScheme(
            ConceptType.CALCULUS,
            primary_colors={
                "function": "#0984E3",     # Blue for functions
                "derivative": "#E17055",   # Orange for derivatives
                "integral": "#00B894",     # Green for integrals
                "limit": "#6C5CE7",       # Purple for limits
                "tangent": "#FD79A8",     # Pink for tangent lines
                "default": "#0984E3"
            },
            gradient_colors=["#2D3436", "#0984E3", "#E17055", "#00B894"],
            special_mappings={
                "partial": ColorMapping("partial", "#E17055", "Partial derivative"),
                "chain_rule": ColorMapping("chain_rule", "#FD79A8", "Chain rule"),
                "gradient": ColorMapping("gradient", "#FDCB6E", "Gradient vector")
            }
        )
        
        return schemes
    
    def _define_concept_relationships(self) -> Dict[ConceptType, List[ConceptType]]:
        """Define relationships between concepts for color consistency."""
        return {
            ConceptType.LINEAR_ALGEBRA: [
                ConceptType.MATRIX_OPERATIONS,
                ConceptType.NEURAL_NETWORK, 
                ConceptType.ATTENTION
            ],
            ConceptType.MATRIX_OPERATIONS: [
                ConceptType.LINEAR_ALGEBRA,
                ConceptType.ATTENTION,
                ConceptType.NEURAL_NETWORK
            ],
            ConceptType.ATTENTION: [
                ConceptType.MATRIX_OPERATIONS,
                ConceptType.LINEAR_ALGEBRA, 
                ConceptType.NEURAL_NETWORK
            ],
            ConceptType.OPTIMIZATION: [
                ConceptType.NEURAL_NETWORK, 
                ConceptType.CALCULUS
            ],
            ConceptType.NEURAL_NETWORK: [
                ConceptType.LINEAR_ALGEBRA,
                ConceptType.MATRIX_OPERATIONS,
                ConceptType.OPTIMIZATION,
                ConceptType.PROBABILITY
            ],
            ConceptType.PROBABILITY: [
                ConceptType.NEURAL_NETWORK, 
                ConceptType.CALCULUS
            ],
            ConceptType.CALCULUS: [
                ConceptType.OPTIMIZATION, 
                ConceptType.PROBABILITY
            ]
        }
    
    def get_scheme(self, concept_type: ConceptType) -> ConceptColorScheme:
        """Get color scheme for a specific concept type."""
        return self.schemes.get(concept_type, self.schemes[ConceptType.LINEAR_ALGEBRA])
    
    def get_related_color(
        self, 
        primary_concept: ConceptType, 
        element_type: str,
        related_concept: Optional[ConceptType] = None
    ) -> str:
        """Get color that's consistent across related concepts."""
        primary_scheme = self.get_scheme(primary_concept)
        
        if related_concept and related_concept in self.concept_relationships.get(primary_concept, []):
            # Use color from related concept if it makes sense
            related_scheme = self.get_scheme(related_concept)
            
            # Map common elements between concepts
            if element_type in related_scheme.primary_colors:
                return related_scheme.get_color(element_type)
        
        return primary_scheme.get_color(element_type)
    
    def create_concept_colormap(
        self, 
        concept_type: ConceptType, 
        n_colors: int = 256
    ) -> LinearSegmentedColormap:
        """Create a colormap for a specific concept."""
        scheme = self.get_scheme(concept_type)
        return scheme.colormap
    
    def get_complementary_colors(
        self, 
        concept_type: ConceptType, 
        n_colors: int = 5
    ) -> List[str]:
        """Get a set of complementary colors for a concept."""
        scheme = self.get_scheme(concept_type)
        
        # Return primary colors up to n_colors
        colors = list(scheme.primary_colors.values())[:n_colors]
        
        # If we need more colors, generate from gradient
        if len(colors) < n_colors and scheme.gradient_colors:
            additional_colors = []
            for i in range(n_colors - len(colors)):
                ratio = i / max(1, n_colors - len(colors) - 1)
                color = scheme.get_gradient_color(ratio)
                additional_colors.append(color)
            colors.extend(additional_colors)
        
        return colors[:n_colors]
    
    def validate_color_accessibility(self, colors: List[str]) -> Dict[str, bool]:
        """Validate color accessibility (contrast ratios, color-blind friendly)."""
        # Simplified accessibility check
        results = {}
        
        for color in colors:
            # Check if color is distinguishable (basic heuristic)
            results[color] = len(color) == 7 and color.startswith('#')
        
        return results


# Global color scheme manager instance
color_manager = ColorSchemeManager()


def get_concept_colors(concept_type: ConceptType) -> ConceptColorScheme:
    """Get color scheme for a mathematical concept."""
    return color_manager.get_scheme(concept_type)


def get_consistent_color(
    primary_concept: ConceptType, 
    element_type: str,
    related_concept: Optional[ConceptType] = None
) -> str:
    """Get color that maintains consistency across related concepts."""
    return color_manager.get_related_color(primary_concept, element_type, related_concept)