"""
Utility functions for the AI Math Tutorial system.
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json


def generate_id(content: str, prefix: str = "") -> str:
    """Generate a unique ID based on content hash."""
    hash_obj = hashlib.md5(content.encode())
    hash_hex = hash_obj.hexdigest()[:8]
    return f"{prefix}_{hash_hex}" if prefix else hash_hex


def validate_latex(latex_expr: str) -> bool:
    """Basic validation of LaTeX expressions."""
    # Check for balanced braces
    brace_count = 0
    for char in latex_expr:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        if brace_count < 0:
            return False
    return brace_count == 0


def extract_variables_from_latex(latex_expr: str) -> List[str]:
    """Extract variable names from LaTeX expression."""
    # Simple regex to find single letter variables
    variables = re.findall(r'\\?([a-zA-Z])', latex_expr)
    # Filter out LaTeX commands
    latex_commands = {'sin', 'cos', 'tan', 'log', 'exp', 'sum', 'int', 'frac'}
    variables = [v for v in variables if v not in latex_commands]
    return list(set(variables))


def normalize_matrix(matrix: np.ndarray, method: str = "l2") -> np.ndarray:
    """Normalize a matrix using specified method."""
    if method == "l2":
        norm = np.linalg.norm(matrix)
        return matrix / norm if norm > 0 else matrix
    elif method == "softmax":
        exp_matrix = np.exp(matrix - np.max(matrix))
        return exp_matrix / np.sum(exp_matrix)
    elif method == "minmax":
        min_val, max_val = np.min(matrix), np.max(matrix)
        if max_val > min_val:
            return (matrix - min_val) / (max_val - min_val)
        return matrix
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_color_gradient(start_color: str, end_color: str, steps: int) -> List[str]:
    """Create a color gradient between two colors."""
    # Simple hex color interpolation
    start_rgb = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
    
    gradient = []
    for i in range(steps):
        ratio = i / (steps - 1) if steps > 1 else 0
        rgb = tuple(int(start_rgb[j] + ratio * (end_rgb[j] - start_rgb[j])) 
                   for j in range(3))
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        gradient.append(hex_color)
    
    return gradient


def validate_matrix_dimensions(matrices: List[np.ndarray], 
                             operation: str) -> bool:
    """Validate matrix dimensions for specific operations."""
    if operation == "multiply":
        for i in range(len(matrices) - 1):
            if matrices[i].shape[1] != matrices[i+1].shape[0]:
                return False
        return True
    elif operation == "add":
        first_shape = matrices[0].shape
        return all(m.shape == first_shape for m in matrices)
    elif operation == "attention":
        # For attention: Q, K, V should have compatible dimensions
        if len(matrices) >= 3:
            q, k, v = matrices[0], matrices[1], matrices[2]
            return (q.shape[-1] == k.shape[-1] and 
                   k.shape[-2] == v.shape[-2])
    return True


def format_number(num: float, precision: int = 4) -> str:
    """Format a number for display with appropriate precision."""
    if abs(num) < 1e-10:
        return "0"
    elif abs(num) >= 1000:
        return f"{num:.2e}"
    else:
        return f"{num:.{precision}f}".rstrip('0').rstrip('.')


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                epsilon: float = 1e-10) -> np.ndarray:
    """Perform safe division with epsilon to avoid division by zero."""
    return numerator / (denominator + epsilon)


def create_mesh_grid(x_range: Tuple[float, float], 
                    y_range: Tuple[float, float],
                    resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Create a mesh grid for 3D plotting."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    return np.meshgrid(x, y)


def serialize_numpy(obj: Any) -> Any:
    """Serialize numpy arrays for JSON storage."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: serialize_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(item) for item in obj]
    return obj


def deserialize_numpy(obj: Any) -> Any:
    """Deserialize JSON data back to numpy arrays."""
    if isinstance(obj, list) and len(obj) > 0:
        # Try to convert to numpy array if it looks like numeric data
        try:
            return np.array(obj)
        except (ValueError, TypeError):
            return [deserialize_numpy(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: deserialize_numpy(value) for key, value in obj.items()}
    return obj