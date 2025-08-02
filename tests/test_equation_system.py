"""
Tests for the equation definition system.
"""

import pytest
import numpy as np
from src.content.equation_system import (
    EquationParser, VariableTracker, MathematicalPropertyChecker,
    EquationDefinitionSystem
)
from src.core.models import VariableDefinition


class TestEquationParser:
    """Test cases for LaTeX equation parsing."""
    
    def test_extract_variables_simple(self):
        parser = EquationParser()
        latex = r"y = mx + b"
        variables = parser.extract_variables(latex)
        expected = {'y', 'm', 'x', 'b'}
        assert variables == expected
    
    def test_extract_variables_with_subscripts(self):
        parser = EquationParser()
        latex = r"A_{ij} = \sum_{k=1}^n B_{ik} C_{kj}"
        variables = parser.extract_variables(latex)
        # Should extract A, B, C, k, n (base variable names)
        assert 'A' in variables
        assert 'B' in variables
        assert 'C' in variables
        assert 'k' in variables
        assert 'n' in variables
    
    def test_validate_latex_syntax_valid(self):
        parser = EquationParser()
        latex = r"\frac{1}{2} \sum_{i=1}^n x_i^2"
        is_valid, errors = parser.validate_latex_syntax(latex)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_latex_syntax_unbalanced_braces(self):
        parser = EquationParser()
        latex = r"\frac{1}{2 \sum_{i=1}^n x_i^2"  # Missing closing brace
        is_valid, errors = parser.validate_latex_syntax(latex)
        assert not is_valid
        assert "Unbalanced braces" in errors[0]
    
    def test_clean_latex_commands(self):
        parser = EquationParser()
        latex = r"\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}"
        cleaned = parser._clean_latex_commands(latex)
        # Should remove \text{softmax} and \frac
        assert r"\text{softmax}" not in cleaned
        assert r"\frac" not in cleaned


class TestVariableTracker:
    """Test cases for variable tracking system."""
    
    def test_register_variable(self):
        tracker = VariableTracker()
        var_def = VariableDefinition(
            name="x",
            description="Input vector",
            data_type="vector",
            shape=(10,)
        )
        
        tracker.register_variable("x", var_def)
        
        retrieved = tracker.get_variable("x")
        assert retrieved is not None
        assert retrieved.name == "x"
        assert retrieved.data_type == "vector"
        assert retrieved.color_code is not None
    
    def test_color_assignment(self):
        tracker = VariableTracker()
        var1 = VariableDefinition("x", "Variable 1", "scalar")
        var2 = VariableDefinition("y", "Variable 2", "scalar")
        
        tracker.register_variable("x", var1)
        tracker.register_variable("y", var2)
        
        # Should have different colors
        assert var1.color_code != var2.color_code
        assert var1.color_code in tracker.color_palette
        assert var2.color_code in tracker.color_palette
    
    def test_variable_relationships(self):
        tracker = VariableTracker()
        
        tracker.add_relationship("x", "y")
        tracker.add_relationship("y", "z")
        
        x_related = tracker.get_related_variables("x")
        y_related = tracker.get_related_variables("y")
        
        assert "y" in x_related
        assert "x" in y_related
        assert "z" in y_related
    
    def test_validate_consistency(self):
        tracker = VariableTracker()
        var_def = VariableDefinition("x", "Variable", "scalar")
        tracker.register_variable("x", var_def)
        
        # Add relationship to undefined variable
        tracker.add_relationship("x", "undefined_var")
        
        is_valid, errors = tracker.validate_variable_consistency()
        assert not is_valid
        assert len(errors) > 0
        assert "undefined_var" in errors[0]


class TestMathematicalPropertyChecker:
    """Test cases for mathematical property validation."""
    
    def test_validate_known_properties(self):
        checker = MathematicalPropertyChecker()
        from src.core.models import Equation
        
        equation = Equation(
            equation_id="test",
            latex_expression="a + b = b + a",
            variables={},
            mathematical_properties=["commutative", "symmetric"]
        )
        
        is_valid, errors = checker.validate_properties(equation)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_unknown_properties(self):
        checker = MathematicalPropertyChecker()
        from src.core.models import Equation
        
        equation = Equation(
            equation_id="test",
            latex_expression="a + b = b + a",
            variables={},
            mathematical_properties=["unknown_property"]
        )
        
        is_valid, errors = checker.validate_properties(equation)
        assert not is_valid
        assert "unknown_property" in errors[0]
    
    def test_suggest_properties(self):
        checker = MathematicalPropertyChecker()
        latex = "a + b * c = a * c + b * c"
        
        suggestions = checker.suggest_properties(latex)
        assert "distributive" in suggestions


class TestEquationDefinitionSystem:
    """Test cases for the complete equation definition system."""
    
    def test_create_valid_equation(self):
        system = EquationDefinitionSystem()
        
        variable_defs = {
            "y": {"description": "Output", "data_type": "scalar"},
            "m": {"description": "Slope", "data_type": "scalar"},
            "x": {"description": "Input", "data_type": "scalar"},
            "b": {"description": "Intercept", "data_type": "scalar"}
        }
        
        is_valid, equation, errors = system.create_equation(
            equation_id="linear",
            latex_expression="y = mx + b",
            variable_definitions=variable_defs,
            properties=["linear"],
            applications=["regression"]
        )
        
        assert is_valid
        assert len(errors) == 0
        assert equation.equation_id == "linear"
        assert len(equation.variables) == 4
        assert "linear" in equation.mathematical_properties
    
    def test_create_equation_missing_variable(self):
        system = EquationDefinitionSystem()
        
        variable_defs = {
            "y": {"description": "Output", "data_type": "scalar"},
            "x": {"description": "Input", "data_type": "scalar"}
            # Missing 'm' and 'b'
        }
        
        is_valid, equation, errors = system.create_equation(
            equation_id="linear",
            latex_expression="y = mx + b",
            variable_definitions=variable_defs
        )
        
        assert not is_valid
        assert len(errors) > 0
        assert any("not defined" in error for error in errors)
    
    def test_create_equation_invalid_latex(self):
        system = EquationDefinitionSystem()
        
        variable_defs = {
            "x": {"description": "Variable", "data_type": "scalar"}
        }
        
        is_valid, equation, errors = system.create_equation(
            equation_id="invalid",
            latex_expression="x = {unclosed brace",
            variable_definitions=variable_defs
        )
        
        assert not is_valid
        assert len(errors) > 0
        assert any("brace" in error.lower() for error in errors)
    
    def test_equation_storage_and_retrieval(self):
        system = EquationDefinitionSystem()
        
        variable_defs = {
            "x": {"description": "Input variable", "data_type": "scalar"},
            "y": {"description": "Output variable", "data_type": "scalar"}
        }
        
        is_valid, equation, errors = system.create_equation(
            equation_id="simple",
            latex_expression="y = x",
            variable_definitions=variable_defs
        )
        
        assert is_valid, f"Equation creation failed with errors: {errors}"
        
        retrieved = system.get_equation("simple")
        assert retrieved is not None
        assert retrieved.equation_id == "simple"
        
        equation_list = system.list_equations()
        assert "simple" in equation_list
    
    def test_validate_all_equations(self):
        system = EquationDefinitionSystem()
        
        # Add a valid equation
        variable_defs = {
            "x": {"description": "Variable", "data_type": "scalar"}
        }
        
        system.create_equation(
            equation_id="valid",
            latex_expression="y = x",
            variable_definitions=variable_defs
        )
        
        is_valid, all_errors = system.validate_all_equations()
        assert is_valid
        assert len(all_errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])