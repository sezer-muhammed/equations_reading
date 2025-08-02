"""
Tests for core data models.
"""

import pytest
import numpy as np
from src.core.models import (
    MathematicalConcept,
    Equation,
    VariableDefinition,
    NumericalExample,
    ComputationStep,
    ColorCodedMatrix
)


def test_variable_definition_creation():
    """Test creating a variable definition."""
    var_def = VariableDefinition(
        name="x",
        description="Input vector",
        data_type="vector",
        shape=(10,),
        constraints="normalized",
        color_code="#1f77b4"
    )
    
    assert var_def.name == "x"
    assert var_def.data_type == "vector"
    assert var_def.shape == (10,)


def test_equation_creation():
    """Test creating an equation with variables."""
    var_x = VariableDefinition("x", "Input", "vector")
    var_w = VariableDefinition("W", "Weight matrix", "matrix")
    
    equation = Equation(
        equation_id="linear_transform",
        latex_expression=r"y = Wx + b",
        variables={"x": var_x, "W": var_w},
        mathematical_properties=["linear", "affine"],
        complexity_level=2
    )
    
    assert equation.equation_id == "linear_transform"
    assert len(equation.variables) == 2
    assert "linear" in equation.mathematical_properties


def test_numerical_example_creation():
    """Test creating a numerical example."""
    input_data = {"x": np.array([1, 2, 3])}
    output_data = {"y": np.array([2, 4, 6])}
    
    comp_step = ComputationStep(
        step_number=1,
        operation_name="multiply",
        input_values=input_data,
        operation_description="Multiply by 2",
        output_values=output_data
    )
    
    example = NumericalExample(
        example_id="simple_multiply",
        description="Simple multiplication example",
        input_values=input_data,
        computation_steps=[comp_step],
        output_values=output_data
    )
    
    assert example.example_id == "simple_multiply"
    assert len(example.computation_steps) == 1
    assert np.array_equal(example.input_values["x"], np.array([1, 2, 3]))


def test_color_coded_matrix():
    """Test creating a color-coded matrix."""
    matrix_data = np.array([[1, 2], [3, 4]])
    color_mapping = {"positive": "#2ca02c", "negative": "#d62728"}
    
    colored_matrix = ColorCodedMatrix(
        matrix_data=matrix_data,
        color_mapping=color_mapping
    )
    
    assert colored_matrix.matrix_data.shape == (2, 2)
    assert "positive" in colored_matrix.color_mapping


def test_mathematical_concept_creation():
    """Test creating a complete mathematical concept."""
    var_def = VariableDefinition("x", "Input", "vector")
    equation = Equation(
        equation_id="test_eq",
        latex_expression="y = x",
        variables={"x": var_def}
    )
    
    concept = MathematicalConcept(
        concept_id="identity_function",
        title="Identity Function",
        prerequisites=["basic_algebra"],
        equations=[equation],
        explanations=[],
        examples=[],
        visualizations=[],
        difficulty_level=1,
        learning_objectives=["Understand identity mapping"]
    )
    
    assert concept.concept_id == "identity_function"
    assert len(concept.equations) == 1
    assert concept.difficulty_level == 1
    assert "Understand identity mapping" in concept.learning_objectives


if __name__ == "__main__":
    pytest.main([__file__])