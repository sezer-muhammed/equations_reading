#!/usr/bin/env python3
"""
Demo script for matrix operation visualizations.
Shows step-by-step matrix multiplication, attention computation, and gradient flow.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.operation_viz.matrix_operations import (
    MatrixOperationVisualizer,
    create_sample_matrix_multiplication,
    create_sample_attention_visualization
)


def demo_matrix_multiplication():
    """Demonstrate matrix multiplication visualization."""
    print("=== Matrix Multiplication Demo ===")
    
    # Create sample matrices
    matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_b = np.array([[7, 8], [9, 10], [11, 12]])
    
    print(f"Matrix A (2x3):\n{matrix_a}")
    print(f"Matrix B (3x2):\n{matrix_b}")
    print(f"Expected result (2x2):\n{np.dot(matrix_a, matrix_b)}")
    
    # Create visualizer
    visualizer = MatrixOperationVisualizer()
    
    # Generate visualization
    result = visualizer.visualize_matrix_multiplication(matrix_a, matrix_b, show_steps=True)
    
    print(f"Generated {len(result.intermediate_steps)} intermediate steps")
    print(f"Generated {len(result.animation_sequence)} animation frames")
    
    # Create plot
    fig = visualizer.create_operation_plot(result)
    plt.show()
    
    return result


def demo_attention_computation():
    """Demonstrate attention mechanism visualization."""
    print("\n=== Attention Computation Demo ===")
    
    # Create sample Q, K, V matrices
    seq_len, d_k, d_v = 3, 4, 4
    np.random.seed(42)  # For reproducible results
    
    query = np.random.randn(seq_len, d_k) * 0.5
    key = np.random.randn(seq_len, d_k) * 0.5
    value = np.random.randn(seq_len, d_v) * 0.5
    
    print(f"Query matrix ({seq_len}x{d_k}):\n{query}")
    print(f"Key matrix ({seq_len}x{d_k}):\n{key}")
    print(f"Value matrix ({seq_len}x{d_v}):\n{value}")
    
    # Create visualizer
    visualizer = MatrixOperationVisualizer()
    
    # Generate visualization
    result = visualizer.visualize_attention_computation(query, key, value)
    
    print(f"Generated {len(result.intermediate_steps)} intermediate steps")
    print(f"Generated {len(result.animation_sequence)} animation frames")
    
    # Create plot
    fig = visualizer.create_operation_plot(result)
    plt.show()
    
    return result


def demo_gradient_flow():
    """Demonstrate gradient flow visualization."""
    print("\n=== Gradient Flow Demo ===")
    
    # Create sample forward pass and gradients
    np.random.seed(123)
    forward_matrices = [
        np.random.randn(3, 4) * 0.5,
        np.random.randn(4, 3) * 0.3,
        np.random.randn(3, 2) * 0.2
    ]
    
    gradients = [
        np.random.randn(3, 4) * 0.1,
        np.random.randn(4, 3) * 0.05,
        np.random.randn(3, 2) * 0.02
    ]
    
    layer_names = ["input", "hidden", "output"]
    
    print("Forward pass matrices:")
    for i, matrix in enumerate(forward_matrices):
        print(f"  Layer {layer_names[i]} ({matrix.shape}): mean={np.mean(matrix):.3f}")
    
    print("Gradient matrices:")
    for i, grad in enumerate(gradients):
        print(f"  Layer {layer_names[i]} ({grad.shape}): mean={np.mean(np.abs(grad)):.3f}")
    
    # Create visualizer
    visualizer = MatrixOperationVisualizer()
    
    # Generate visualization
    result = visualizer.visualize_gradient_flow(forward_matrices, gradients, layer_names)
    
    print(f"Generated {len(result.intermediate_steps)} intermediate steps")
    print(f"Generated {len(result.animation_sequence)} animation frames")
    
    # Create plot
    fig = visualizer.create_operation_plot(result)
    plt.show()
    
    return result


def main():
    """Run all demos."""
    print("Matrix Operation Visualization Demo")
    print("=" * 40)
    
    # Run demos
    mult_result = demo_matrix_multiplication()
    attn_result = demo_attention_computation()
    grad_result = demo_gradient_flow()
    
    print("\n=== Demo Complete ===")
    print("All visualizations generated successfully!")
    
    # Show summary
    print(f"\nSummary:")
    print(f"- Matrix multiplication: {len(mult_result.animation_sequence)} steps")
    print(f"- Attention computation: {len(attn_result.animation_sequence)} steps")
    print(f"- Gradient flow: {len(grad_result.animation_sequence)} steps")


if __name__ == "__main__":
    main()