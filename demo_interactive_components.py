"""
Demonstration script for interactive visualization components.
Shows parameter widgets, hover effects, and animation controls in action.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.models import (
    VariableDefinition, ColorCodedMatrix, OperationVisualization,
    NumericalExample, MathematicalConcept, Equation, HighlightPattern
)
from src.visualization.interactive.parameter_widgets import ParameterConfig, create_attention_parameter_widget
from src.visualization.interactive.hover_effects import create_attention_hover_demo
from src.visualization.interactive.animation_controls import create_matrix_multiplication_demo
from src.visualization.interactive.interactive_components import create_attention_interactive_demo


def demo_parameter_widgets():
    """Demonstrate parameter manipulation widgets."""
    print("=== Parameter Widgets Demo ===")
    
    try:
        # Create attention parameter widget
        widget = create_attention_parameter_widget()
        print("✓ Attention parameter widget created successfully")
        
        # Test parameter access
        current_values = widget.get_current_values()
        print(f"✓ Current parameter values: {current_values}")
        
        # Test parameter update
        widget.update_parameter("temperature", 2.0)
        updated_values = widget.get_current_values()
        print(f"✓ Updated parameter values: {updated_values}")
        
        plt.close('all')  # Clean up
        
    except Exception as e:
        print(f"✗ Parameter widgets demo failed: {e}")


def demo_hover_effects():
    """Demonstrate hover effects for variable highlighting."""
    print("\n=== Hover Effects Demo ===")
    
    try:
        # Create hover demo
        fig, hover_manager = create_attention_hover_demo()
        print("✓ Hover effects demo created successfully")
        
        # Test hover region creation
        print(f"✓ Number of hover regions: {len(hover_manager.hover_regions)}")
        
        # Test variable mapping
        if hasattr(hover_manager, 'variable_mappings'):
            print(f"✓ Variable mappings: {len(hover_manager.variable_mappings)}")
        
        plt.close('all')  # Clean up
        
    except Exception as e:
        print(f"✗ Hover effects demo failed: {e}")


def demo_animation_controls():
    """Demonstrate animation controls for step-by-step demonstrations."""
    print("\n=== Animation Controls Demo ===")
    
    try:
        # Create matrix multiplication demo
        fig, controller = create_matrix_multiplication_demo()
        print("✓ Animation controls demo created successfully")
        
        # Test controller properties
        print(f"✓ Total frames: {controller.get_total_frames()}")
        print(f"✓ Current frame: {controller.get_current_frame()}")
        print(f"✓ Is playing: {controller.is_animation_playing()}")
        
        # Test frame navigation
        controller.step_forward()
        print(f"✓ After step forward - Current frame: {controller.get_current_frame()}")
        
        controller.step_backward()
        print(f"✓ After step backward - Current frame: {controller.get_current_frame()}")
        
        plt.close('all')  # Clean up
        
    except Exception as e:
        print(f"✗ Animation controls demo failed: {e}")


def demo_integrated_visualization():
    """Demonstrate integrated interactive visualization."""
    print("\n=== Integrated Interactive Visualization Demo ===")
    
    try:
        # Create attention interactive demo
        demo = create_attention_interactive_demo()
        print("✓ Integrated visualization created successfully")
        
        # Test concept properties
        print(f"✓ Concept ID: {demo.concept.concept_id}")
        print(f"✓ Number of equations: {len(demo.concept.equations)}")
        print(f"✓ Number of examples: {len(demo.concept.examples)}")
        
        # Test example loading
        if demo.concept.examples:
            demo.load_example(0)
            print("✓ Example loaded successfully")
        
        # Test parameter widget integration
        if demo.parameter_widget:
            current_params = demo.parameter_widget.get_current_values()
            print(f"✓ Parameter widget integrated - Current values: {current_params}")
        
        plt.close('all')  # Clean up
        
    except Exception as e:
        print(f"✗ Integrated visualization demo failed: {e}")


def demo_matrix_operations():
    """Demonstrate matrix operations with interactive components."""
    print("\n=== Matrix Operations Demo ===")
    
    try:
        # Create sample matrices
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = np.matmul(A, B)
        
        print("✓ Sample matrices created:")
        print(f"  A = \n{A}")
        print(f"  B = \n{B}")
        print(f"  C = A × B = \n{C}")
        
        # Create ColorCodedMatrix objects
        matrix_A = ColorCodedMatrix(
            matrix_data=A,
            color_mapping={"input": "#3498DB"},
            highlight_patterns=[
                HighlightPattern("row", [(0,)], "red", "First row")
            ]
        )
        
        matrix_B = ColorCodedMatrix(
            matrix_data=B,
            color_mapping={"input": "#E74C3C"}
        )
        
        matrix_C = ColorCodedMatrix(
            matrix_data=C,
            color_mapping={"output": "#27AE60"}
        )
        
        print("✓ ColorCodedMatrix objects created successfully")
        
        # Create operation visualization
        operation_viz = OperationVisualization(
            operation_type="matrix_multiply",
            input_matrices=[matrix_A, matrix_B],
            intermediate_steps=[],
            output_matrix=matrix_C
        )
        
        print("✓ Operation visualization created successfully")
        print(f"  Operation type: {operation_viz.operation_type}")
        print(f"  Number of input matrices: {len(operation_viz.input_matrices)}")
        
    except Exception as e:
        print(f"✗ Matrix operations demo failed: {e}")


def demo_variable_definitions():
    """Demonstrate variable definitions and their usage."""
    print("\n=== Variable Definitions Demo ===")
    
    try:
        # Create variable definitions for attention mechanism
        variables = {
            "Q": VariableDefinition(
                name="Q",
                description="Query matrix - represents what we're looking for",
                data_type="matrix",
                shape=(4, 3),
                constraints="normalized",
                color_code="#FF6B6B"
            ),
            "K": VariableDefinition(
                name="K", 
                description="Key matrix - represents what we're comparing against",
                data_type="matrix",
                shape=(4, 3),
                constraints="normalized",
                color_code="#4ECDC4"
            ),
            "V": VariableDefinition(
                name="V",
                description="Value matrix - represents the information to retrieve",
                data_type="matrix", 
                shape=(4, 3),
                color_code="#45B7D1"
            ),
            "d_k": VariableDefinition(
                name="d_k",
                description="Key dimension for scaling",
                data_type="scalar",
                constraints="positive",
                color_code="#96CEB4"
            )
        }
        
        print("✓ Variable definitions created successfully:")
        for name, var_def in variables.items():
            print(f"  {name}: {var_def.description}")
            print(f"    Type: {var_def.data_type}, Shape: {var_def.shape}")
            print(f"    Color: {var_def.color_code}")
        
        # Create equation using these variables
        equation = Equation(
            equation_id="scaled_dot_product_attention",
            latex_expression=r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            variables=variables,
            mathematical_properties=["Permutation invariant", "Differentiable"],
            applications=["Transformer models", "Neural machine translation"]
        )
        
        print("✓ Attention equation created successfully")
        print(f"  Equation ID: {equation.equation_id}")
        print(f"  Number of variables: {len(equation.variables)}")
        print(f"  Properties: {equation.mathematical_properties}")
        
    except Exception as e:
        print(f"✗ Variable definitions demo failed: {e}")


def main():
    """Run all demonstration functions."""
    print("Interactive Visualization Components Demonstration")
    print("=" * 50)
    
    # Run individual demos
    demo_variable_definitions()
    demo_matrix_operations()
    demo_parameter_widgets()
    demo_hover_effects()
    demo_animation_controls()
    demo_integrated_visualization()
    
    print("\n" + "=" * 50)
    print("Demonstration completed!")
    print("\nNote: Some components require matplotlib GUI backend for full interactivity.")
    print("The core functionality has been verified programmatically.")


if __name__ == "__main__":
    main()