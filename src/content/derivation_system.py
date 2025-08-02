"""
Derivation step management system for mathematical content engine.
Handles step-by-step mathematical derivations, intermediate results, and reasoning.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import copy
from src.core.models import DerivationStep, Equation


class DerivationStepType(Enum):
    """Types of derivation steps."""
    ALGEBRAIC_MANIPULATION = "algebraic_manipulation"
    SUBSTITUTION = "substitution"
    SIMPLIFICATION = "simplification"
    EXPANSION = "expansion"
    FACTORIZATION = "factorization"
    INTEGRATION = "integration"
    DIFFERENTIATION = "differentiation"
    LIMIT = "limit"
    ASSUMPTION = "assumption"
    DEFINITION = "definition"
    THEOREM_APPLICATION = "theorem_application"


class JustificationType(Enum):
    """Types of mathematical justifications."""
    ALGEBRAIC_PROPERTY = "algebraic_property"
    CALCULUS_RULE = "calculus_rule"
    DEFINITION = "definition"
    THEOREM = "theorem"
    LEMMA = "lemma"
    AXIOM = "axiom"
    ASSUMPTION = "assumption"
    SUBSTITUTION = "substitution"


@dataclass
class DerivationRule:
    """Represents a mathematical rule that can be applied in derivations."""
    rule_id: str
    name: str
    description: str
    rule_type: JustificationType
    latex_pattern: str  # Pattern that this rule can be applied to
    prerequisites: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class IntermediateResult:
    """Stores intermediate computational results during derivation."""
    step_number: int
    variable_name: str
    value: Any  # Could be numerical, symbolic, or LaTeX expression
    value_type: str  # "numerical", "symbolic", "latex"
    computation_method: str
    precision: Optional[int] = None
    units: Optional[str] = None


@dataclass
class DerivationContext:
    """Context information for a derivation process."""
    assumptions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    variable_domains: Dict[str, str] = field(default_factory=dict)
    mathematical_context: str = "general"  # "linear_algebra", "calculus", etc.
    difficulty_level: int = 1


class DerivationStepManager:
    """Manages individual derivation steps with validation and tracking."""
    
    def __init__(self):
        self.rules_database = self._initialize_rules_database()
        self.step_validators = {
            DerivationStepType.ALGEBRAIC_MANIPULATION: self._validate_algebraic_step,
            DerivationStepType.SUBSTITUTION: self._validate_substitution_step,
            DerivationStepType.SIMPLIFICATION: self._validate_simplification_step,
        }
    
    def create_derivation_step(self,
                              step_number: int,
                              latex_expression: str,
                              explanation: str,
                              step_type: DerivationStepType,
                              justification_type: JustificationType,
                              rule_applied: Optional[str] = None,
                              intermediate_values: Optional[Dict[str, Any]] = None) -> DerivationStep:
        """Create a new derivation step with validation."""
        
        # Generate mathematical justification
        mathematical_justification = self._generate_justification(
            step_type, justification_type, rule_applied
        )
        
        step = DerivationStep(
            step_number=step_number,
            latex_expression=latex_expression,
            explanation=explanation,
            mathematical_justification=mathematical_justification,
            intermediate_values=intermediate_values or {}
        )
        
        return step
    
    def _generate_justification(self,
                               step_type: DerivationStepType,
                               justification_type: JustificationType,
                               rule_applied: Optional[str]) -> str:
        """Generate mathematical justification text for a step."""
        base_text = f"Applied {justification_type.value}"
        
        if rule_applied and rule_applied in self.rules_database:
            rule = self.rules_database[rule_applied]
            return f"{base_text}: {rule.name} - {rule.description}"
        
        return f"{base_text} for {step_type.value}"
    
    def validate_step(self,
                     step: DerivationStep,
                     previous_expression: str,
                     context: DerivationContext) -> Tuple[bool, List[str]]:
        """Validate that a derivation step is mathematically sound."""
        errors = []
        
        # Basic validation
        if not step.latex_expression.strip():
            errors.append("Empty LaTeX expression in derivation step")
        
        if not step.explanation.strip():
            errors.append("Missing explanation for derivation step")
        
        # Step-specific validation would go here
        # This is a simplified version - real implementation would need
        # symbolic math libraries for proper validation
        
        return len(errors) == 0, errors
    
    def _validate_algebraic_step(self, step: DerivationStep, 
                                previous_expr: str, context: DerivationContext) -> List[str]:
        """Validate algebraic manipulation steps."""
        # Placeholder for algebraic validation logic
        return []
    
    def _validate_substitution_step(self, step: DerivationStep,
                                   previous_expr: str, context: DerivationContext) -> List[str]:
        """Validate substitution steps."""
        # Placeholder for substitution validation logic
        return []
    
    def _validate_simplification_step(self, step: DerivationStep,
                                     previous_expr: str, context: DerivationContext) -> List[str]:
        """Validate simplification steps."""
        # Placeholder for simplification validation logic
        return []
    
    def _initialize_rules_database(self) -> Dict[str, DerivationRule]:
        """Initialize database of mathematical rules."""
        rules = {}
        
        # Basic algebraic rules
        rules["distributive"] = DerivationRule(
            rule_id="distributive",
            name="Distributive Property",
            description="a(b + c) = ab + ac",
            rule_type=JustificationType.ALGEBRAIC_PROPERTY,
            latex_pattern=r"[a-zA-Z]\([^)]+\+[^)]+\)"
        )
        
        rules["commutative_addition"] = DerivationRule(
            rule_id="commutative_addition",
            name="Commutative Property of Addition",
            description="a + b = b + a",
            rule_type=JustificationType.ALGEBRAIC_PROPERTY,
            latex_pattern=r"[a-zA-Z]\+[a-zA-Z]"
        )
        
        rules["associative_addition"] = DerivationRule(
            rule_id="associative_addition",
            name="Associative Property of Addition",
            description="(a + b) + c = a + (b + c)",
            rule_type=JustificationType.ALGEBRAIC_PROPERTY,
            latex_pattern=r"\([^)]+\+[^)]+\)\+[a-zA-Z]"
        )
        
        # Calculus rules
        rules["power_rule"] = DerivationRule(
            rule_id="power_rule",
            name="Power Rule",
            description="d/dx[x^n] = nx^(n-1)",
            rule_type=JustificationType.CALCULUS_RULE,
            latex_pattern=r"\\frac\{d\}\{dx\}.*\^"
        )
        
        rules["chain_rule"] = DerivationRule(
            rule_id="chain_rule",
            name="Chain Rule",
            description="d/dx[f(g(x))] = f'(g(x)) * g'(x)",
            rule_type=JustificationType.CALCULUS_RULE,
            latex_pattern=r"\\frac\{d\}\{dx\}"
        )
        
        return rules


class IntermediateResultManager:
    """Manages intermediate computational results during derivations."""
    
    def __init__(self):
        self.results_storage: Dict[int, List[IntermediateResult]] = {}
        self.computation_methods = {
            "numerical": self._compute_numerical,
            "symbolic": self._compute_symbolic,
            "approximation": self._compute_approximation
        }
    
    def store_intermediate_result(self,
                                 step_number: int,
                                 variable_name: str,
                                 value: Any,
                                 computation_method: str,
                                 value_type: str = "numerical") -> IntermediateResult:
        """Store an intermediate result for a derivation step."""
        
        result = IntermediateResult(
            step_number=step_number,
            variable_name=variable_name,
            value=value,
            value_type=value_type,
            computation_method=computation_method
        )
        
        if step_number not in self.results_storage:
            self.results_storage[step_number] = []
        
        self.results_storage[step_number].append(result)
        return result
    
    def get_results_for_step(self, step_number: int) -> List[IntermediateResult]:
        """Get all intermediate results for a specific step."""
        return self.results_storage.get(step_number, [])
    
    def get_variable_history(self, variable_name: str) -> List[IntermediateResult]:
        """Get the history of values for a specific variable across steps."""
        history = []
        for step_results in self.results_storage.values():
            for result in step_results:
                if result.variable_name == variable_name:
                    history.append(result)
        
        return sorted(history, key=lambda r: r.step_number)
    
    def validate_result_consistency(self) -> Tuple[bool, List[str]]:
        """Validate that intermediate results are consistent across steps."""
        errors = []
        
        # Check for type consistency
        variable_types = {}
        for step_results in self.results_storage.values():
            for result in step_results:
                var_name = result.variable_name
                if var_name in variable_types:
                    if variable_types[var_name] != result.value_type:
                        errors.append(f"Inconsistent types for variable {var_name}")
                else:
                    variable_types[var_name] = result.value_type
        
        return len(errors) == 0, errors
    
    def _compute_numerical(self, expression: str, variables: Dict[str, float]) -> float:
        """Compute numerical result (placeholder implementation)."""
        # In a real implementation, this would use a math evaluation library
        return 0.0
    
    def _compute_symbolic(self, expression: str, variables: Dict[str, str]) -> str:
        """Compute symbolic result (placeholder implementation)."""
        # In a real implementation, this would use SymPy or similar
        return expression
    
    def _compute_approximation(self, expression: str, variables: Dict[str, float]) -> float:
        """Compute approximate numerical result."""
        # Placeholder for approximation methods
        return 0.0


class MathematicalReasoningEngine:
    """Generates explanations and reasoning for mathematical steps."""
    
    def __init__(self):
        self.explanation_templates = self._initialize_explanation_templates()
        self.reasoning_patterns = self._initialize_reasoning_patterns()
    
    def generate_step_explanation(self,
                                 step_type: DerivationStepType,
                                 latex_before: str,
                                 latex_after: str,
                                 rule_applied: Optional[str] = None,
                                 context: Optional[DerivationContext] = None) -> str:
        """Generate a clear explanation for a derivation step."""
        
        template_key = f"{step_type.value}_explanation"
        if template_key in self.explanation_templates:
            template = self.explanation_templates[template_key]
            
            # Fill in template with specific information
            explanation = template.format(
                rule=rule_applied or "mathematical property",
                before=latex_before,
                after=latex_after
            )
            
            return explanation
        
        # Fallback generic explanation
        return f"Applied {step_type.value} to transform the expression"
    
    def generate_intuitive_explanation(self,
                                     derivation_steps: List[DerivationStep],
                                     context: DerivationContext) -> str:
        """Generate an intuitive explanation of the entire derivation."""
        
        if not derivation_steps:
            return "No derivation steps provided."
        
        explanation_parts = [
            "This derivation follows these key steps:",
            ""
        ]
        
        for i, step in enumerate(derivation_steps, 1):
            step_summary = self._summarize_step(step, i)
            explanation_parts.append(f"{i}. {step_summary}")
        
        explanation_parts.extend([
            "",
            f"The overall goal is to {self._infer_derivation_goal(derivation_steps)}",
            f"This derivation uses concepts from {context.mathematical_context}."
        ])
        
        return "\n".join(explanation_parts)
    
    def _summarize_step(self, step: DerivationStep, step_number: int) -> str:
        """Create a brief summary of a derivation step."""
        # Extract key information from the step
        if "=" in step.latex_expression:
            return f"Establish the relationship: {step.explanation}"
        elif "substitute" in step.explanation.lower():
            return f"Substitute values: {step.explanation}"
        elif "simplify" in step.explanation.lower():
            return f"Simplify the expression: {step.explanation}"
        else:
            return step.explanation
    
    def _infer_derivation_goal(self, steps: List[DerivationStep]) -> str:
        """Infer the overall goal of the derivation."""
        if not steps:
            return "complete the mathematical transformation"
        
        last_step = steps[-1]
        if "=" in last_step.latex_expression:
            return "establish the final equation"
        elif any(word in last_step.explanation.lower() 
                for word in ["prove", "show", "demonstrate"]):
            return "prove the mathematical relationship"
        else:
            return "derive the final mathematical expression"
    
    def _initialize_explanation_templates(self) -> Dict[str, str]:
        """Initialize templates for step explanations."""
        return {
            "algebraic_manipulation_explanation": 
                "Apply {rule} to rearrange the terms in the equation",
            "substitution_explanation": 
                "Substitute the known value using {rule}",
            "simplification_explanation": 
                "Simplify the expression by applying {rule}",
            "expansion_explanation": 
                "Expand the expression using {rule}",
            "factorization_explanation": 
                "Factor the expression using {rule}"
        }
    
    def _initialize_reasoning_patterns(self) -> Dict[str, str]:
        """Initialize common reasoning patterns."""
        return {
            "proof_by_substitution": "We substitute known values to verify the relationship",
            "algebraic_manipulation": "We rearrange terms using algebraic properties",
            "step_by_step_simplification": "We simplify the expression step by step"
        }


class DerivationManager:
    """High-level manager for complete mathematical derivations."""
    
    def __init__(self):
        self.step_manager = DerivationStepManager()
        self.result_manager = IntermediateResultManager()
        self.reasoning_engine = MathematicalReasoningEngine()
        self.derivations: Dict[str, List[DerivationStep]] = {}
        self.derivation_contexts: Dict[str, DerivationContext] = {}
    
    def create_derivation(self,
                         derivation_id: str,
                         starting_equation: str,
                         target_equation: str,
                         context: Optional[DerivationContext] = None) -> None:
        """Initialize a new derivation process."""
        
        self.derivations[derivation_id] = []
        self.derivation_contexts[derivation_id] = context or DerivationContext()
        
        # Create initial step
        initial_step = self.step_manager.create_derivation_step(
            step_number=0,
            latex_expression=starting_equation,
            explanation="Starting equation",
            step_type=DerivationStepType.ASSUMPTION,
            justification_type=JustificationType.ASSUMPTION
        )
        
        self.derivations[derivation_id].append(initial_step)
    
    def add_derivation_step(self,
                           derivation_id: str,
                           latex_expression: str,
                           explanation: str,
                           step_type: DerivationStepType,
                           justification_type: JustificationType,
                           rule_applied: Optional[str] = None,
                           intermediate_values: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Add a new step to an existing derivation."""
        
        if derivation_id not in self.derivations:
            return False, [f"Derivation {derivation_id} not found"]
        
        steps = self.derivations[derivation_id]
        step_number = len(steps)
        
        # Create the new step
        new_step = self.step_manager.create_derivation_step(
            step_number=step_number,
            latex_expression=latex_expression,
            explanation=explanation,
            step_type=step_type,
            justification_type=justification_type,
            rule_applied=rule_applied,
            intermediate_values=intermediate_values
        )
        
        # Validate the step
        previous_expr = steps[-1].latex_expression if steps else ""
        context = self.derivation_contexts[derivation_id]
        
        is_valid, errors = self.step_manager.validate_step(new_step, previous_expr, context)
        
        if is_valid:
            steps.append(new_step)
            
            # Store intermediate results if provided
            if intermediate_values:
                for var_name, value in intermediate_values.items():
                    self.result_manager.store_intermediate_result(
                        step_number=step_number,
                        variable_name=var_name,
                        value=value,
                        computation_method="manual"
                    )
        
        return is_valid, errors
    
    def get_derivation(self, derivation_id: str) -> Optional[List[DerivationStep]]:
        """Get all steps in a derivation."""
        return self.derivations.get(derivation_id)
    
    def generate_complete_explanation(self, derivation_id: str) -> Optional[str]:
        """Generate a complete explanation for the derivation."""
        if derivation_id not in self.derivations:
            return None
        
        steps = self.derivations[derivation_id]
        context = self.derivation_contexts[derivation_id]
        
        return self.reasoning_engine.generate_intuitive_explanation(steps, context)
    
    def validate_derivation(self, derivation_id: str) -> Tuple[bool, List[str]]:
        """Validate an entire derivation for mathematical correctness."""
        if derivation_id not in self.derivations:
            return False, [f"Derivation {derivation_id} not found"]
        
        steps = self.derivations[derivation_id]
        context = self.derivation_contexts[derivation_id]
        all_errors = []
        
        # Validate each step
        for i, step in enumerate(steps[1:], 1):  # Skip initial assumption
            previous_expr = steps[i-1].latex_expression
            is_valid, errors = self.step_manager.validate_step(step, previous_expr, context)
            if not is_valid:
                all_errors.extend([f"Step {i}: {error}" for error in errors])
        
        # Validate intermediate results consistency
        is_consistent, consistency_errors = self.result_manager.validate_result_consistency()
        if not is_consistent:
            all_errors.extend(consistency_errors)
        
        return len(all_errors) == 0, all_errors
    
    def export_derivation(self, derivation_id: str) -> Optional[Dict[str, Any]]:
        """Export derivation data for external use."""
        if derivation_id not in self.derivations:
            return None
        
        steps = self.derivations[derivation_id]
        context = self.derivation_contexts[derivation_id]
        
        return {
            "derivation_id": derivation_id,
            "steps": [
                {
                    "step_number": step.step_number,
                    "latex_expression": step.latex_expression,
                    "explanation": step.explanation,
                    "justification": step.mathematical_justification,
                    "intermediate_values": step.intermediate_values
                }
                for step in steps
            ],
            "context": {
                "assumptions": context.assumptions,
                "constraints": context.constraints,
                "mathematical_context": context.mathematical_context,
                "difficulty_level": context.difficulty_level
            },
            "complete_explanation": self.generate_complete_explanation(derivation_id)
        }