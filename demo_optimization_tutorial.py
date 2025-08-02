#!/usr/bin/env python3
"""
Demonstration of the optimization algorithms tutorial.
Shows gradient descent, Adam optimizer, learning rate scheduling, and loss landscapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.content.foundations.optimization_tutorial import OptimizationTutorial
from src.visualization.graph_viz.optimization_landscapes import OptimizationLandscapeVisualizer
from src.visualization.interactive.optimization_widgets import OptimizationInteractiveComponents


def main():
    """Run the optimization tutorial demonstration."""
    print("🚀 AI Math Tutorial: Optimization Algorithms")
    print("=" * 50)
    
    # Create tutorial instance
    tutorial = OptimizationTutorial()
    
    # Generate complete tutorial
    print("📚 Generating optimization tutorial content...")
    tutorial_result = tutorial.create_complete_tutorial()
    
    # Display tutorial information
    concept = tutorial_result.concept
    print(f"\n📖 Tutorial: {concept.title}")
    print(f"🎯 Difficulty Level: {concept.difficulty_level}/5")
    print(f"📋 Prerequisites: {', '.join(concept.prerequisites)}")
    
    # Show equations
    print(f"\n🧮 Mathematical Equations ({len(concept.equations)}):")
    for eq in concept.equations:
        print(f"  • {eq.equation_id}: {eq.latex_expression}")
    
    # Demonstrate gradient descent examples
    print(f"\n📊 Gradient Descent Examples ({len(tutorial_result.gradient_descent_examples)}):")
    for i, example in enumerate(tutorial_result.gradient_descent_examples):
        print(f"  Example {i+1}:")
        print(f"    - Final loss: {example.convergence_info['final_loss']:.6f}")
        print(f"    - Steps: {example.convergence_info['total_steps']}")
        print(f"    - Converged: {example.convergence_info['converged']}")
    
    # Demonstrate Adam examples
    print(f"\n🎯 Adam Optimizer Examples ({len(tutorial_result.adam_examples)}):")
    for i, example in enumerate(tutorial_result.adam_examples):
        print(f"  Example {i+1}:")
        print(f"    - Final loss: {example.convergence_info['final_loss']:.6f}")
        print(f"    - Steps: {example.convergence_info['total_steps']}")
        print(f"    - Converged: {example.convergence_info['converged']}")
        if 'adam_hyperparameters' in example.convergence_info:
            params = example.convergence_info['adam_hyperparameters']
            print(f"    - β₁: {params['beta1']}, β₂: {params['beta2']}")
    
    # Show learning rate scheduling
    print(f"\n📈 Learning Rate Scheduling ({len(tutorial_result.learning_rate_examples)}):")
    for example in tutorial_result.learning_rate_examples:
        schedule_type = example['schedule_type']
        final_lr = example['results'][-1]['lr']
        print(f"  • {schedule_type}: Final LR = {final_lr:.6f}")
    
    # Show loss landscapes
    print(f"\n🏔️ Loss Landscapes ({len(tutorial_result.loss_landscape_examples)}):")
    for example in tutorial_result.loss_landscape_examples:
        landscape_type = example['landscape_type']
        properties = ', '.join(example['properties'])
        print(f"  • {landscape_type}: {properties}")
    
    # Demonstrate visualization capabilities
    print(f"\n🎨 Creating Visualizations...")
    visualizer = OptimizationLandscapeVisualizer()
    
    # Create loss landscape visualization
    landscape_data = tutorial_result.loss_landscape_examples[0]  # Quadratic
    gd_results = tutorial_result.gradient_descent_examples[:2]  # First two examples
    
    landscape_viz = visualizer.create_loss_landscape(
        landscape_data, gd_results, ['GD-Quadratic', 'GD-Rosenbrock']
    )
    print(f"  ✅ Loss landscape with {len(landscape_viz.optimizer_paths)} optimizer paths")
    
    # Create convergence plot
    convergence_data = visualizer.create_convergence_plot(
        tutorial_result.gradient_descent_examples[:2], ['GD-Quadratic', 'GD-Rosenbrock']
    )
    print(f"  ✅ Convergence plots for {len(convergence_data['optimizers'])} optimizers")
    
    # Create Adam components visualization
    adam_viz = visualizer.create_adam_components_visualization(
        tutorial_result.adam_examples[0]
    )
    print(f"  ✅ Adam components visualization with {len(adam_viz['iterations'])} steps")
    
    # Demonstrate interactive components
    print(f"\n🎮 Interactive Components:")
    interactive = OptimizationInteractiveComponents()
    
    # Learning rate demo
    lr_demo = interactive.create_learning_rate_demo()
    print(f"  ✅ Learning rate demo with {len(lr_demo.parameter_widgets)} controls")
    
    # Adam hyperparameter demo
    adam_demo = interactive.create_adam_hyperparameter_demo()
    print(f"  ✅ Adam hyperparameter demo with {len(adam_demo.parameter_widgets)} controls")
    
    # Optimizer comparison demo
    comparison_demo = interactive.create_optimizer_comparison_demo()
    print(f"  ✅ Optimizer comparison with {len(comparison_demo.parameter_widgets)} controls")
    
    # Show tutorial sections
    print(f"\n📚 Tutorial Sections ({len(tutorial_result.tutorial_sections)}):")
    for i, section in enumerate(tutorial_result.tutorial_sections):
        title = section['title']
        print(f"  {i+1}. {title}")
        if 'learning_objectives' in section:
            print(f"     Objectives: {len(section['learning_objectives'])}")
        if 'key_insights' in section:
            print(f"     Key insights: {len(section['key_insights'])}")
    
    # Demonstrate specific mathematical computations
    print(f"\n🔢 Mathematical Computation Examples:")
    
    # Show gradient descent step-by-step
    gd_example = tutorial_result.gradient_descent_examples[0]
    print(f"  Gradient Descent (first 3 steps):")
    for i, step in enumerate(gd_example.computation_steps[:6]):  # First 3 parameter updates
        print(f"    Step {step.step_number}: {step.operation_name}")
        print(f"      {step.operation_description}")
    
    # Show Adam step-by-step
    adam_example = tutorial_result.adam_examples[0]
    print(f"  Adam Optimizer (first step breakdown):")
    for i, step in enumerate(adam_example.computation_steps[:6]):  # First complete Adam step
        print(f"    {step.step_number}: {step.operation_name}")
        print(f"      {step.operation_description}")
    
    print(f"\n✨ Optimization Tutorial Complete!")
    print(f"📊 Generated {len(tutorial_result.gradient_descent_examples)} GD examples")
    print(f"🎯 Generated {len(tutorial_result.adam_examples)} Adam examples")
    print(f"📈 Generated {len(tutorial_result.learning_rate_examples)} LR schedules")
    print(f"🏔️ Generated {len(tutorial_result.loss_landscape_examples)} loss landscapes")
    print(f"📚 Created {len(tutorial_result.tutorial_sections)} tutorial sections")


if __name__ == "__main__":
    main()