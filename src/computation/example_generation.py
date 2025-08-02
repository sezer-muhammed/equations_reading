"""
Example generation system for AI equations with realistic parameter values.
Provides automatic validation and edge case detection for numerical stability.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

from ..core.models import NumericalExample, ComputationStep, VisualizationData


class ParameterType(Enum):
    """Types of parameters for AI equations."""
    WEIGHT_MATRIX = "weight_matrix"
    BIAS_VECTOR = "bias_vector"
    LEARNING_RATE = "learning_rate"
    ATTENTION_SCORES = "attention_scores"
    PROBABILITY_DISTRIBUTION = "probability_distribution"
    EMBEDDING_VECTOR = "embedding_vector"
    SEQUENCE_LENGTH = "sequence_length"
    BATCH_SIZE = "batch_size"
    HIDDEN_DIMENSION = "hidden_dimension"


@dataclass
class ParameterSpec:
    """Specification for generating a parameter."""
    param_type: ParameterType
    shape: Tuple[int, ...]
    constraints: Dict[str, Any]
    description: str
    typical_range: Tuple[float, float]


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    properties: Dict[str, Any]


class ParameterGenerator:
    """Generates realistic parameter values for AI equations."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        # Define typical parameter specifications
        self.parameter_specs = {
            ParameterType.WEIGHT_MATRIX: ParameterSpec(
                param_type=ParameterType.WEIGHT_MATRIX,
                shape=(),  # Will be specified per use
                constraints={'initialization': 'xavier', 'std_range': (0.01, 0.1)},
                description="Neural network weight matrix",
                typical_range=(-0.5, 0.5)
            ),
            ParameterType.BIAS_VECTOR: ParameterSpec(
                param_type=ParameterType.BIAS_VECTOR,
                shape=(),
                constraints={'initialization': 'zero_or_small'},
                description="Neural network bias vector",
                typical_range=(-0.1, 0.1)
            ),
            ParameterType.LEARNING_RATE: ParameterSpec(
                param_type=ParameterType.LEARNING_RATE,
                shape=(1,),
                constraints={'min': 1e-6, 'max': 1.0, 'log_scale': True},
                description="Optimization learning rate",
                typical_range=(1e-4, 1e-1)
            ),
            ParameterType.ATTENTION_SCORES: ParameterSpec(
                param_type=ParameterType.ATTENTION_SCORES,
                shape=(),
                constraints={'normalization': 'softmax'},
                description="Attention mechanism scores",
                typical_range=(-5.0, 5.0)
            ),
            ParameterType.PROBABILITY_DISTRIBUTION: ParameterSpec(
                param_type=ParameterType.PROBABILITY_DISTRIBUTION,
                shape=(),
                constraints={'sum_to_one': True, 'non_negative': True},
                description="Probability distribution",
                typical_range=(0.0, 1.0)
            ),
            ParameterType.EMBEDDING_VECTOR: ParameterSpec(
                param_type=ParameterType.EMBEDDING_VECTOR,
                shape=(),
                constraints={'normalization': 'unit_or_bounded'},
                description="Embedding vector",
                typical_range=(-1.0, 1.0)
            )
        }
    
    def generate_parameter(self, param_type: ParameterType, shape: Tuple[int, ...],
                          custom_constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate a realistic parameter value.
        
        Args:
            param_type: Type of parameter to generate
            shape: Shape of the parameter
            custom_constraints: Optional custom constraints
            
        Returns:
            Generated parameter array
        """
        spec = self.parameter_specs[param_type]
        constraints = spec.constraints.copy()
        if custom_constraints:
            constraints.update(custom_constraints)
        
        if param_type == ParameterType.WEIGHT_MATRIX:
            return self._generate_weight_matrix(shape, constraints)
        elif param_type == ParameterType.BIAS_VECTOR:
            return self._generate_bias_vector(shape, constraints)
        elif param_type == ParameterType.LEARNING_RATE:
            return self._generate_learning_rate(constraints)
        elif param_type == ParameterType.ATTENTION_SCORES:
            return self._generate_attention_scores(shape, constraints)
        elif param_type == ParameterType.PROBABILITY_DISTRIBUTION:
            return self._generate_probability_distribution(shape, constraints)
        elif param_type == ParameterType.EMBEDDING_VECTOR:
            return self._generate_embedding_vector(shape, constraints)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _generate_weight_matrix(self, shape: Tuple[int, ...], constraints: Dict[str, Any]) -> np.ndarray:
        """Generate weight matrix with proper initialization."""
        init_method = constraints.get('initialization', 'xavier')
        
        if init_method == 'xavier':
            # Xavier/Glorot initialization
            fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, shape)
        elif init_method == 'he':
            # He initialization
            fan_in = shape[0]
            std = np.sqrt(2.0 / fan_in)
            return np.random.normal(0, std, shape)
        elif init_method == 'normal':
            std_range = constraints.get('std_range', (0.01, 0.1))
            std = np.random.uniform(*std_range)
            return np.random.normal(0, std, shape)
        else:
            return np.random.uniform(-0.1, 0.1, shape)
    
    def _generate_bias_vector(self, shape: Tuple[int, ...], constraints: Dict[str, Any]) -> np.ndarray:
        """Generate bias vector."""
        init_method = constraints.get('initialization', 'zero_or_small')
        
        if init_method == 'zero':
            return np.zeros(shape)
        elif init_method == 'zero_or_small':
            # 70% chance of zero initialization, 30% small random
            if np.random.random() < 0.7:
                return np.zeros(shape)
            else:
                return np.random.uniform(-0.01, 0.01, shape)
        else:
            return np.random.uniform(-0.1, 0.1, shape)
    
    def _generate_learning_rate(self, constraints: Dict[str, Any]) -> np.ndarray:
        """Generate learning rate."""
        min_lr = constraints.get('min', 1e-6)
        max_lr = constraints.get('max', 1.0)
        log_scale = constraints.get('log_scale', True)
        
        if log_scale:
            # Generate on log scale for more realistic distribution
            log_min = np.log10(min_lr)
            log_max = np.log10(max_lr)
            log_lr = np.random.uniform(log_min, log_max)
            return np.array([10 ** log_lr])
        else:
            return np.array([np.random.uniform(min_lr, max_lr)])
    
    def _generate_attention_scores(self, shape: Tuple[int, ...], constraints: Dict[str, Any]) -> np.ndarray:
        """Generate attention scores before softmax."""
        # Generate scores that will produce reasonable attention patterns
        scores = np.random.normal(0, 2.0, shape)
        
        # Add some structure to make attention patterns more realistic
        if len(shape) == 2:
            # For sequence-to-sequence attention, add diagonal bias
            diagonal_strength = np.random.uniform(0.5, 2.0)
            for i in range(min(shape)):
                scores[i, i] += diagonal_strength
        
        return scores
    
    def _generate_probability_distribution(self, shape: Tuple[int, ...], constraints: Dict[str, Any]) -> np.ndarray:
        """Generate probability distribution."""
        # Generate from Dirichlet distribution for realistic probabilities
        alpha = np.ones(shape[-1]) * 2.0  # Concentration parameter
        if len(shape) == 1:
            return np.random.dirichlet(alpha)
        else:
            # For multiple distributions
            result = np.zeros(shape)
            for idx in np.ndindex(shape[:-1]):
                result[idx] = np.random.dirichlet(alpha)
            return result
    
    def _generate_embedding_vector(self, shape: Tuple[int, ...], constraints: Dict[str, Any]) -> np.ndarray:
        """Generate embedding vector."""
        normalization = constraints.get('normalization', 'unit_or_bounded')
        
        # Generate random vector
        vector = np.random.normal(0, 1, shape)
        
        if normalization == 'unit':
            # Normalize to unit length
            return vector / np.linalg.norm(vector, axis=-1, keepdims=True)
        elif normalization == 'bounded':
            # Clip to reasonable range
            return np.clip(vector, -2.0, 2.0)
        elif normalization == 'unit_or_bounded':
            # 50% chance of each
            if np.random.random() < 0.5:
                return vector / np.linalg.norm(vector, axis=-1, keepdims=True)
            else:
                return np.clip(vector, -2.0, 2.0)
        else:
            return vector


class ParameterValidator:
    """Validates generated parameters for mathematical properties and numerical stability."""
    
    def __init__(self):
        self.validation_functions = {
            ParameterType.WEIGHT_MATRIX: self._validate_weight_matrix,
            ParameterType.BIAS_VECTOR: self._validate_bias_vector,
            ParameterType.LEARNING_RATE: self._validate_learning_rate,
            ParameterType.ATTENTION_SCORES: self._validate_attention_scores,
            ParameterType.PROBABILITY_DISTRIBUTION: self._validate_probability_distribution,
            ParameterType.EMBEDDING_VECTOR: self._validate_embedding_vector
        }
    
    def validate_parameter(self, param: np.ndarray, param_type: ParameterType,
                          constraints: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a parameter for mathematical properties and numerical stability.
        
        Args:
            param: Parameter array to validate
            param_type: Type of parameter
            constraints: Optional constraints to check
            
        Returns:
            ValidationResult with validation details
        """
        if param_type not in self.validation_functions:
            return ValidationResult(
                is_valid=False,
                issues=[f"Unknown parameter type: {param_type}"],
                warnings=[],
                properties={}
            )
        
        return self.validation_functions[param_type](param, constraints or {})
    
    def _validate_weight_matrix(self, param: np.ndarray, constraints: Dict[str, Any]) -> ValidationResult:
        """Validate weight matrix."""
        issues = []
        warnings = []
        properties = {}
        
        # Check for NaN or infinite values
        if np.any(np.isnan(param)) or np.any(np.isinf(param)):
            issues.append("Contains NaN or infinite values")
        
        # Check magnitude
        max_val = np.max(np.abs(param))
        properties['max_magnitude'] = max_val
        if max_val > 10.0:
            warnings.append(f"Large weight values detected (max: {max_val:.3f})")
        
        # Check for dead neurons (all zeros in row/column)
        if len(param.shape) == 2:
            zero_rows = np.all(param == 0, axis=1)
            zero_cols = np.all(param == 0, axis=0)
            if np.any(zero_rows):
                warnings.append(f"Found {np.sum(zero_rows)} zero rows (dead neurons)")
            if np.any(zero_cols):
                warnings.append(f"Found {np.sum(zero_cols)} zero columns")
        
        # Check condition number for numerical stability
        if len(param.shape) == 2 and min(param.shape) > 1:
            try:
                cond_num = np.linalg.cond(param)
                properties['condition_number'] = cond_num
                if cond_num > 1e12:
                    warnings.append(f"High condition number: {cond_num:.2e}")
            except np.linalg.LinAlgError:
                warnings.append("Could not compute condition number")
        
        properties['mean'] = np.mean(param)
        properties['std'] = np.std(param)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            properties=properties
        )
    
    def _validate_bias_vector(self, param: np.ndarray, constraints: Dict[str, Any]) -> ValidationResult:
        """Validate bias vector."""
        issues = []
        warnings = []
        properties = {}
        
        # Check for NaN or infinite values
        if np.any(np.isnan(param)) or np.any(np.isinf(param)):
            issues.append("Contains NaN or infinite values")
        
        # Check magnitude
        max_val = np.max(np.abs(param))
        properties['max_magnitude'] = max_val
        if max_val > 5.0:
            warnings.append(f"Large bias values detected (max: {max_val:.3f})")
        
        properties['mean'] = np.mean(param)
        properties['std'] = np.std(param)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            properties=properties
        )
    
    def _validate_learning_rate(self, param: np.ndarray, constraints: Dict[str, Any]) -> ValidationResult:
        """Validate learning rate."""
        issues = []
        warnings = []
        properties = {}
        
        lr = param[0] if param.size > 0 else 0
        
        # Check for valid range
        if lr <= 0:
            issues.append("Learning rate must be positive")
        elif lr > 1.0:
            warnings.append(f"Very high learning rate: {lr:.6f}")
        elif lr < 1e-8:
            warnings.append(f"Very low learning rate: {lr:.2e}")
        
        properties['value'] = lr
        properties['log10_value'] = np.log10(lr) if lr > 0 else -np.inf
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            properties=properties
        )
    
    def _validate_attention_scores(self, param: np.ndarray, constraints: Dict[str, Any]) -> ValidationResult:
        """Validate attention scores."""
        issues = []
        warnings = []
        properties = {}
        
        # Check for NaN or infinite values
        if np.any(np.isnan(param)) or np.any(np.isinf(param)):
            issues.append("Contains NaN or infinite values")
        
        # Check for extreme values that might cause softmax overflow
        max_val = np.max(param)
        min_val = np.min(param)
        properties['max_value'] = max_val
        properties['min_value'] = min_val
        properties['range'] = max_val - min_val
        
        if max_val > 50:
            warnings.append(f"Very large attention scores (max: {max_val:.2f}) may cause softmax overflow")
        if min_val < -50:
            warnings.append(f"Very negative attention scores (min: {min_val:.2f}) may cause underflow")
        
        # Test softmax stability
        try:
            softmax_result = np.exp(param - np.max(param, axis=-1, keepdims=True))
            softmax_result = softmax_result / np.sum(softmax_result, axis=-1, keepdims=True)
            if np.any(np.isnan(softmax_result)):
                issues.append("Softmax produces NaN values")
        except:
            warnings.append("Could not test softmax stability")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            properties=properties
        )
    
    def _validate_probability_distribution(self, param: np.ndarray, constraints: Dict[str, Any]) -> ValidationResult:
        """Validate probability distribution."""
        issues = []
        warnings = []
        properties = {}
        
        # Check non-negativity
        if np.any(param < 0):
            issues.append("Probability values must be non-negative")
        
        # Check normalization
        sums = np.sum(param, axis=-1)
        properties['sums'] = sums
        if not np.allclose(sums, 1.0, atol=1e-6):
            issues.append("Probabilities do not sum to 1")
        
        # Check for extreme values
        max_prob = np.max(param)
        min_prob = np.min(param)
        properties['max_probability'] = max_prob
        properties['min_probability'] = min_prob
        
        if max_prob > 0.99:
            warnings.append("Very peaked distribution detected")
        if min_prob < 1e-10 and min_prob > 0:
            warnings.append("Very small probabilities may cause numerical issues")
        
        # Calculate entropy
        entropy = -np.sum(param * np.log(param + 1e-10), axis=-1)
        properties['entropy'] = entropy
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            properties=properties
        )
    
    def _validate_embedding_vector(self, param: np.ndarray, constraints: Dict[str, Any]) -> ValidationResult:
        """Validate embedding vector."""
        issues = []
        warnings = []
        properties = {}
        
        # Check for NaN or infinite values
        if np.any(np.isnan(param)) or np.any(np.isinf(param)):
            issues.append("Contains NaN or infinite values")
        
        # Check magnitude
        norms = np.linalg.norm(param, axis=-1)
        properties['norms'] = norms
        properties['mean_norm'] = np.mean(norms)
        
        max_norm = np.max(norms)
        if max_norm > 10.0:
            warnings.append(f"Very large embedding magnitude: {max_norm:.3f}")
        
        # Check for zero vectors
        zero_vectors = norms < 1e-10
        if np.any(zero_vectors):
            warnings.append(f"Found {np.sum(zero_vectors)} near-zero embedding vectors")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            properties=properties
        )


class ExampleGenerator:
    """Generates complete numerical examples for AI equations."""
    
    def __init__(self, seed: Optional[int] = None):
        self.param_generator = ParameterGenerator(seed)
        self.validator = ParameterValidator()
    
    def generate_example(self, equation_type: str, 
                        parameter_specs: Dict[str, Tuple[ParameterType, Tuple[int, ...]]],
                        custom_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
                        max_retries: int = 10) -> NumericalExample:
        """
        Generate a complete numerical example for an equation.
        
        Args:
            equation_type: Type of equation (e.g., "attention", "matrix_multiply")
            parameter_specs: Specifications for each parameter
            custom_constraints: Optional custom constraints per parameter
            max_retries: Maximum retries for generating valid parameters
            
        Returns:
            NumericalExample with validated parameters
        """
        constraints = custom_constraints or {}
        
        for attempt in range(max_retries):
            try:
                # Generate parameters
                input_values = {}
                validation_results = {}
                
                for param_name, (param_type, shape) in parameter_specs.items():
                    param_constraints = constraints.get(param_name, {})
                    param_value = self.param_generator.generate_parameter(
                        param_type, shape, param_constraints
                    )
                    
                    # Validate parameter
                    validation = self.validator.validate_parameter(
                        param_value, param_type, param_constraints
                    )
                    
                    if not validation.is_valid:
                        raise ValueError(f"Invalid parameter {param_name}: {validation.issues}")
                    
                    input_values[param_name] = param_value
                    validation_results[param_name] = validation
                
                # Create example
                example = NumericalExample(
                    example_id=f"{equation_type}_example_{attempt}",
                    description=f"Numerical example for {equation_type} with validated parameters",
                    input_values=input_values,
                    computation_steps=[],  # Will be filled by computation
                    output_values={},      # Will be filled by computation
                    visualization_data=None,
                    educational_notes=[
                        f"Generated with {len(parameter_specs)} parameters",
                        f"All parameters validated for numerical stability",
                        f"Attempt {attempt + 1} of {max_retries}"
                    ]
                )
                
                # Add validation info to educational notes
                for param_name, validation in validation_results.items():
                    if validation.warnings:
                        example.educational_notes.extend([
                            f"{param_name} warnings: {', '.join(validation.warnings)}"
                        ])
                
                return example
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to generate valid example after {max_retries} attempts: {e}")
                continue
        
        raise RuntimeError("Unexpected error in example generation")