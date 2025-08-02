"""
Equation definition system for mathematical content engine.
Handles LaTeX equation parsing, variable tracking, and validation.
"""

import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from src.core.models import Equation, VariableDefinition, DerivationStep


@dataclass
class EquationParser:
    """Parser for LaTeX equations with variable extraction and validation."""
    
    def __init__(self):
        # Common LaTeX patterns for mathematical elements
        self.variable_pattern = r'\\?([a-zA-Z](?:_\{[^}]+\})?(?:\^[^{}\s]*|\^\{[^}]+\})?)'
        self.function_pattern = r'\\([a-zA-Z]+)\s*(?:\{[^}]*\}|\([^)]*\))?'
        self.operator_pattern = r'(\\[a-zA-Z]+|[+\-*/=<>])'
        
    def extract_variables(self, latex_expression: str) -> Set[str]:
        """Extract all variables from a LaTeX expression."""
        # Remove common LaTeX commands that aren't variables
        cleaned = self._clean_latex_commands(latex_expression)
        
        # Find all potential variables including subscripted ones
        variables = set()
        
        # Pattern for variables with subscripts/superscripts
        complex_var_pattern = r'([a-zA-Z])(?:_\{[^}]+\}|\^\{[^}]+\}|_[a-zA-Z0-9]|\^[a-zA-Z0-9])*'
        # Pattern for simple variables
        simple_var_pattern = r'\b([a-zA-Z])\b'
        
        # Extract complex variables first
        complex_matches = re.findall(complex_var_pattern, cleaned)
        for match in complex_matches:
            if not self._is_latex_function(match):
                variables.add(match)
        
        # Extract simple variables that aren't part of complex ones
        simple_matches = re.findall(simple_var_pattern, cleaned)
        for match in simple_matches:
            if not self._is_latex_function(match):
                variables.add(match)
                
        return variables
    
    def _clean_latex_commands(self, latex: str) -> str:
        """Remove LaTeX formatting commands that don't represent variables."""
        # Remove common formatting commands
        formatting_commands = [
            r'\\text\{[^}]*\}', r'\\mathrm\{[^}]*\}', r'\\mathbf\{[^}]*\}',
            r'\\left', r'\\right', r'\\begin\{[^}]*\}', r'\\end\{[^}]*\}',
            r'\\frac', r'\\sum', r'\\prod', r'\\int', r'\\partial'
        ]
        
        cleaned = latex
        for cmd in formatting_commands:
            cleaned = re.sub(cmd, '', cleaned)
            
        return cleaned
    
    def _is_latex_function(self, text: str) -> bool:
        """Check if text represents a LaTeX function rather than a variable."""
        latex_functions = {
            'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'max', 'min',
            'sum', 'prod', 'int', 'partial', 'nabla', 'infty', 'pi', 'alpha',
            'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda', 'mu', 'sigma'
        }
        return text.lower() in latex_functions
    
    def validate_latex_syntax(self, latex_expression: str) -> Tuple[bool, List[str]]:
        """Validate LaTeX syntax and return errors if any."""
        errors = []
        
        # Check for balanced braces
        if not self._check_balanced_braces(latex_expression):
            errors.append("Unbalanced braces in LaTeX expression")
        
        # Check for balanced parentheses
        if not self._check_balanced_parentheses(latex_expression):
            errors.append("Unbalanced parentheses in LaTeX expression")
        
        # Check for valid LaTeX commands
        invalid_commands = self._find_invalid_commands(latex_expression)
        if invalid_commands:
            errors.append(f"Invalid LaTeX commands: {', '.join(invalid_commands)}")
        
        return len(errors) == 0, errors
    
    def _check_balanced_braces(self, text: str) -> bool:
        """Check if braces are balanced in the text."""
        count = 0
        for char in text:
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    def _check_balanced_parentheses(self, text: str) -> bool:
        """Check if parentheses are balanced in the text."""
        count = 0
        for char in text:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    def _find_invalid_commands(self, latex: str) -> List[str]:
        """Find invalid LaTeX commands in the expression."""
        # This is a simplified check - in practice, you'd want a comprehensive list
        valid_commands = {
            'frac', 'sqrt', 'sum', 'prod', 'int', 'partial', 'nabla',
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda',
            'mu', 'sigma', 'pi', 'infty', 'left', 'right', 'begin', 'end',
            'text', 'mathrm', 'mathbf', 'mathit', 'sin', 'cos', 'tan',
            'log', 'ln', 'exp', 'max', 'min', 'cdot', 'times', 'div'
        }
        
        commands = re.findall(r'\\([a-zA-Z]+)', latex)
        invalid = [cmd for cmd in commands if cmd not in valid_commands]
        return list(set(invalid))  # Remove duplicates


@dataclass
class VariableTracker:
    """Tracks variable definitions and their relationships across equations."""
    
    def __init__(self):
        self.variables: Dict[str, VariableDefinition] = {}
        self.variable_relationships: Dict[str, List[str]] = {}
        self.color_assignments: Dict[str, str] = {}
        self.next_color_index = 0
        
        # Predefined color palette for consistent visualization
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
    
    def register_variable(self, var_name: str, definition: VariableDefinition) -> None:
        """Register a new variable with its definition."""
        self.variables[var_name] = definition
        
        # Assign color if not already assigned
        if var_name not in self.color_assignments:
            self.color_assignments[var_name] = self._assign_color()
            definition.color_code = self.color_assignments[var_name]
    
    def get_variable(self, var_name: str) -> Optional[VariableDefinition]:
        """Retrieve variable definition by name."""
        return self.variables.get(var_name)
    
    def add_relationship(self, var1: str, var2: str) -> None:
        """Add a relationship between two variables."""
        if var1 not in self.variable_relationships:
            self.variable_relationships[var1] = []
        if var2 not in self.variable_relationships:
            self.variable_relationships[var2] = []
            
        if var2 not in self.variable_relationships[var1]:
            self.variable_relationships[var1].append(var2)
        if var1 not in self.variable_relationships[var2]:
            self.variable_relationships[var2].append(var1)
    
    def get_related_variables(self, var_name: str) -> List[str]:
        """Get all variables related to the given variable."""
        return self.variable_relationships.get(var_name, [])
    
    def _assign_color(self) -> str:
        """Assign the next available color from the palette."""
        color = self.color_palette[self.next_color_index % len(self.color_palette)]
        self.next_color_index += 1
        return color
    
    def validate_variable_consistency(self) -> Tuple[bool, List[str]]:
        """Validate that all variables are consistently defined."""
        errors = []
        
        # Check for undefined variables in relationships
        for var, related in self.variable_relationships.items():
            if var not in self.variables:
                errors.append(f"Variable '{var}' referenced in relationships but not defined")
            for related_var in related:
                if related_var not in self.variables:
                    errors.append(f"Related variable '{related_var}' not defined")
        
        return len(errors) == 0, errors


@dataclass
class MathematicalPropertyChecker:
    """Validates mathematical properties of equations."""
    
    def __init__(self):
        self.known_properties = {
            'commutative', 'associative', 'distributive', 'identity',
            'inverse', 'idempotent', 'symmetric', 'antisymmetric',
            'transitive', 'reflexive', 'linear', 'bilinear', 'positive_definite'
        }
    
    def validate_properties(self, equation: Equation) -> Tuple[bool, List[str]]:
        """Validate that claimed mathematical properties are reasonable."""
        errors = []
        
        for prop in equation.mathematical_properties:
            if prop not in self.known_properties:
                errors.append(f"Unknown mathematical property: '{prop}'")
        
        # Additional semantic validation could be added here
        # For example, checking if properties are contradictory
        
        return len(errors) == 0, errors
    
    def suggest_properties(self, latex_expression: str) -> List[str]:
        """Suggest mathematical properties based on equation structure."""
        suggestions = []
        
        # Simple heuristics for property suggestion
        if '+' in latex_expression and '*' in latex_expression:
            suggestions.append('distributive')
        
        if '=' in latex_expression:
            suggestions.append('symmetric')
        
        # More sophisticated analysis could be added
        
        return suggestions


@dataclass
class EquationDefinitionSystem:
    """Main system for defining and managing mathematical equations."""
    
    def __init__(self):
        self.parser = EquationParser()
        self.variable_tracker = VariableTracker()
        self.property_checker = MathematicalPropertyChecker()
        self.equations: Dict[str, Equation] = {}
    
    def create_equation(self, 
                       equation_id: str,
                       latex_expression: str,
                       variable_definitions: Dict[str, Dict[str, Any]],
                       properties: Optional[List[str]] = None,
                       applications: Optional[List[str]] = None) -> Tuple[bool, Equation, List[str]]:
        """Create a new equation with validation."""
        errors = []
        
        # Validate LaTeX syntax
        is_valid_latex, latex_errors = self.parser.validate_latex_syntax(latex_expression)
        if not is_valid_latex:
            errors.extend(latex_errors)
        
        # Extract and validate variables
        extracted_vars = self.parser.extract_variables(latex_expression)
        
        # Create variable definitions
        variables = {}
        for var_name in extracted_vars:
            if var_name in variable_definitions:
                var_def = VariableDefinition(
                    name=var_name,
                    description=variable_definitions[var_name].get('description', ''),
                    data_type=variable_definitions[var_name].get('data_type', 'scalar'),
                    shape=variable_definitions[var_name].get('shape'),
                    constraints=variable_definitions[var_name].get('constraints'),
                )
                self.variable_tracker.register_variable(var_name, var_def)
                variables[var_name] = var_def
            else:
                errors.append(f"Variable '{var_name}' found in equation but not defined")
        
        # Create equation object
        equation = Equation(
            equation_id=equation_id,
            latex_expression=latex_expression,
            variables=variables,
            mathematical_properties=properties or [],
            applications=applications or []
        )
        
        # Validate mathematical properties
        if properties:
            is_valid_props, prop_errors = self.property_checker.validate_properties(equation)
            if not is_valid_props:
                errors.extend(prop_errors)
        
        # Store equation if valid
        if not errors:
            self.equations[equation_id] = equation
        
        return len(errors) == 0, equation, errors
    
    def get_equation(self, equation_id: str) -> Optional[Equation]:
        """Retrieve equation by ID."""
        return self.equations.get(equation_id)
    
    def list_equations(self) -> List[str]:
        """List all equation IDs."""
        return list(self.equations.keys())
    
    def validate_all_equations(self) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate all equations in the system."""
        all_errors = {}
        
        for eq_id, equation in self.equations.items():
            # Re-validate each equation
            is_valid_latex, latex_errors = self.parser.validate_latex_syntax(equation.latex_expression)
            is_valid_props, prop_errors = self.property_checker.validate_properties(equation)
            
            errors = latex_errors + prop_errors
            if errors:
                all_errors[eq_id] = errors
        
        # Validate variable consistency
        is_consistent, consistency_errors = self.variable_tracker.validate_variable_consistency()
        if not is_consistent:
            all_errors['variable_consistency'] = consistency_errors
        
        return len(all_errors) == 0, all_errors