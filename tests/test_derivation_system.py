"""
Tests for the derivation step management system.
"""

import pytest
from src.content.derivation_system import (
    DerivationStepManager, IntermediateResultManager, MathematicalReasoningEngine,
    DerivationManager, DerivationStepType, JustificationType, DerivationContext,
    IntermediateResult
)
from src.core.models import DerivationStep


class TestDerivationStepManager:
    """Test cases for derivation step management."""
    
    def test_create_derivation_step(self):
        manager = DerivationStepManager()
        
        step = manager.create_derivation_step(
            step_number=1,
            latex_expression="y = mx + b",
            explanation="Linear equation form",
            step_type=DerivationStepType.DEFINITION,
            justification_type=JustificationType.DEFINITION
        )
        
        assert step.step_number == 1
        assert step.latex_expression == "y = mx + b"
        assert step.explanation == "Linear equation form"
        assert "definition" in step.mathematical_justification.lower()
    
    def test_create_step_with_rule(self):
        manager = DerivationStepManager()
        
        step = manager.create_derivation_step(
            step_number=2,
            latex_expression="ab + ac = a(b + c)",
            explanation="Factor out common term",
            step_type=DerivationStepType.FACTORIZATION,
            justification_type=JustificationType.ALGEBRAIC_PROPERTY,
            rule_applied="distributive"
        )
        
        assert "Distributive Property" in step.mathematical_justification
        assert step.step_number == 2
    
    def test_validate_step_basic(self):
        manager = DerivationStepManager()
        context = DerivationContext()
        
        step = DerivationStep(
            step_number=1,
            latex_expression="x + y = y + x",
            explanation="Commutative property of addition",
            mathematical_justification="Applied commutative property"
        )
        
        is_valid, errors = manager.validate_step(step, "x + y", context)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_step_empty_expression(self):
        manager = DerivationStepManager()
        context = DerivationContext()
        
        step = DerivationStep(
            step_number=1,
            latex_expression="",
            explanation="Empty step",
            mathematical_justification="None"
        )
        
        is_valid, errors = manager.validate_step(step, "x + y", context)
        assert not is_valid
        assert any("Empty LaTeX expression" in error for error in errors)
    
    def test_rules_database_initialization(self):
        manager = DerivationStepManager()
        
        assert "distributive" in manager.rules_database
        assert "commutative_addition" in manager.rules_database
        assert "power_rule" in manager.rules_database
        
        distributive_rule = manager.rules_database["distributive"]
        assert distributive_rule.name == "Distributive Property"
        assert distributive_rule.rule_type == JustificationType.ALGEBRAIC_PROPERTY


class TestIntermediateResultManager:
    """Test cases for intermediate result management."""
    
    def test_store_intermediate_result(self):
        manager = IntermediateResultManager()
        
        result = manager.store_intermediate_result(
            step_number=1,
            variable_name="x",
            value=5.0,
            computation_method="numerical",
            value_type="numerical"
        )
        
        assert result.step_number == 1
        assert result.variable_name == "x"
        assert result.value == 5.0
        assert result.computation_method == "numerical"
    
    def test_get_results_for_step(self):
        manager = IntermediateResultManager()
        
        manager.store_intermediate_result(1, "x", 5.0, "numerical")
        manager.store_intermediate_result(1, "y", 10.0, "numerical")
        manager.store_intermediate_result(2, "z", 15.0, "numerical")
        
        step1_results = manager.get_results_for_step(1)
        step2_results = manager.get_results_for_step(2)
        
        assert len(step1_results) == 2
        assert len(step2_results) == 1
        assert step1_results[0].variable_name in ["x", "y"]
        assert step2_results[0].variable_name == "z"
    
    def test_get_variable_history(self):
        manager = IntermediateResultManager()
        
        manager.store_intermediate_result(1, "x", 5.0, "numerical")
        manager.store_intermediate_result(2, "x", 10.0, "numerical")
        manager.store_intermediate_result(3, "x", 15.0, "numerical")
        manager.store_intermediate_result(2, "y", 20.0, "numerical")
        
        x_history = manager.get_variable_history("x")
        y_history = manager.get_variable_history("y")
        
        assert len(x_history) == 3
        assert len(y_history) == 1
        
        # Check ordering by step number
        assert x_history[0].step_number == 1
        assert x_history[1].step_number == 2
        assert x_history[2].step_number == 3
    
    def test_validate_result_consistency(self):
        manager = IntermediateResultManager()
        
        # Add consistent results
        manager.store_intermediate_result(1, "x", 5.0, "numerical", "numerical")
        manager.store_intermediate_result(2, "x", 10.0, "numerical", "numerical")
        
        is_consistent, errors = manager.validate_result_consistency()
        assert is_consistent
        assert len(errors) == 0
        
        # Add inconsistent result
        manager.store_intermediate_result(3, "x", "symbolic_expr", "symbolic", "symbolic")
        
        is_consistent, errors = manager.validate_result_consistency()
        assert not is_consistent
        assert len(errors) > 0
        assert "Inconsistent types" in errors[0]


class TestMathematicalReasoningEngine:
    """Test cases for mathematical reasoning engine."""
    
    def test_generate_step_explanation(self):
        engine = MathematicalReasoningEngine()
        
        explanation = engine.generate_step_explanation(
            step_type=DerivationStepType.ALGEBRAIC_MANIPULATION,
            latex_before="x + y",
            latex_after="y + x",
            rule_applied="commutative_addition"
        )
        
        assert "rearrange" in explanation.lower()
        assert "commutative_addition" in explanation
    
    def test_generate_step_explanation_fallback(self):
        engine = MathematicalReasoningEngine()
        
        explanation = engine.generate_step_explanation(
            step_type=DerivationStepType.INTEGRATION,  # Not in templates
            latex_before="f(x)",
            latex_after="∫f(x)dx"
        )
        
        assert "integration" in explanation.lower()
    
    def test_generate_intuitive_explanation(self):
        engine = MathematicalReasoningEngine()
        context = DerivationContext(mathematical_context="algebra")
        
        steps = [
            DerivationStep(1, "x + y = 5", "Given equation", "Starting point"),
            DerivationStep(2, "y = 5 - x", "Solve for y", "Algebraic manipulation"),
            DerivationStep(3, "y = -x + 5", "Rearrange", "Standard form")
        ]
        
        explanation = engine.generate_intuitive_explanation(steps, context)
        
        assert "key steps" in explanation.lower()
        assert "algebra" in explanation.lower()
        assert len(explanation.split('\n')) > 3  # Multi-line explanation
    
    def test_generate_intuitive_explanation_empty(self):
        engine = MathematicalReasoningEngine()
        context = DerivationContext()
        
        explanation = engine.generate_intuitive_explanation([], context)
        assert "No derivation steps" in explanation


class TestDerivationManager:
    """Test cases for the complete derivation management system."""
    
    def test_create_derivation(self):
        manager = DerivationManager()
        
        manager.create_derivation(
            derivation_id="linear_solve",
            starting_equation="2x + 3 = 7",
            target_equation="x = 2"
        )
        
        derivation = manager.get_derivation("linear_solve")
        assert derivation is not None
        assert len(derivation) == 1  # Initial step
        assert derivation[0].latex_expression == "2x + 3 = 7"
        assert derivation[0].step_number == 0
    
    def test_add_derivation_step(self):
        manager = DerivationManager()
        
        manager.create_derivation("test", "x + 1 = 3", "x = 2")
        
        success, errors = manager.add_derivation_step(
            derivation_id="test",
            latex_expression="x = 3 - 1",
            explanation="Subtract 1 from both sides",
            step_type=DerivationStepType.ALGEBRAIC_MANIPULATION,
            justification_type=JustificationType.ALGEBRAIC_PROPERTY
        )
        
        assert success
        assert len(errors) == 0
        
        derivation = manager.get_derivation("test")
        assert len(derivation) == 2
        assert derivation[1].latex_expression == "x = 3 - 1"
    
    def test_add_step_with_intermediate_values(self):
        manager = DerivationManager()
        
        manager.create_derivation("test", "x + 1 = 3", "x = 2")
        
        intermediate_values = {"x": 2, "result": 3}
        
        success, errors = manager.add_derivation_step(
            derivation_id="test",
            latex_expression="x = 2",
            explanation="Final result",
            step_type=DerivationStepType.SIMPLIFICATION,
            justification_type=JustificationType.ALGEBRAIC_PROPERTY,
            intermediate_values=intermediate_values
        )
        
        assert success
        
        # Check that intermediate results were stored
        step_results = manager.result_manager.get_results_for_step(1)
        assert len(step_results) == 2
        
        variable_names = [r.variable_name for r in step_results]
        assert "x" in variable_names
        assert "result" in variable_names
    
    def test_add_step_nonexistent_derivation(self):
        manager = DerivationManager()
        
        success, errors = manager.add_derivation_step(
            derivation_id="nonexistent",
            latex_expression="x = 1",
            explanation="Test",
            step_type=DerivationStepType.ASSUMPTION,
            justification_type=JustificationType.ASSUMPTION
        )
        
        assert not success
        assert len(errors) == 1
        assert "not found" in errors[0]
    
    def test_generate_complete_explanation(self):
        manager = DerivationManager()
        
        manager.create_derivation("test", "2x = 6", "x = 3")
        manager.add_derivation_step(
            "test", "x = 6/2", "Divide both sides by 2",
            DerivationStepType.ALGEBRAIC_MANIPULATION,
            JustificationType.ALGEBRAIC_PROPERTY
        )
        manager.add_derivation_step(
            "test", "x = 3", "Simplify",
            DerivationStepType.SIMPLIFICATION,
            JustificationType.ALGEBRAIC_PROPERTY
        )
        
        explanation = manager.generate_complete_explanation("test")
        
        assert explanation is not None
        assert "key steps" in explanation.lower()
        assert len(explanation.split('\n')) > 3
    
    def test_validate_derivation(self):
        manager = DerivationManager()
        
        manager.create_derivation("test", "x + 1 = 3", "x = 2")
        manager.add_derivation_step(
            "test", "x = 2", "Subtract 1 from both sides",
            DerivationStepType.ALGEBRAIC_MANIPULATION,
            JustificationType.ALGEBRAIC_PROPERTY
        )
        
        is_valid, errors = manager.validate_derivation("test")
        assert is_valid
        assert len(errors) == 0
    
    def test_export_derivation(self):
        manager = DerivationManager()
        
        context = DerivationContext(
            assumptions=["x is real"],
            mathematical_context="algebra"
        )
        
        manager.create_derivation("test", "x + 1 = 3", "x = 2", context)
        manager.add_derivation_step(
            "test", "x = 2", "Final result",
            DerivationStepType.SIMPLIFICATION,
            JustificationType.ALGEBRAIC_PROPERTY,
            intermediate_values={"x": 2}
        )
        
        export_data = manager.export_derivation("test")
        
        assert export_data is not None
        assert export_data["derivation_id"] == "test"
        assert len(export_data["steps"]) == 2
        assert export_data["context"]["mathematical_context"] == "algebra"
        assert "x is real" in export_data["context"]["assumptions"]
        assert export_data["complete_explanation"] is not None
    
    def test_export_nonexistent_derivation(self):
        manager = DerivationManager()
        
        export_data = manager.export_derivation("nonexistent")
        assert export_data is None


if __name__ == "__main__":
    pytest.main([__file__])