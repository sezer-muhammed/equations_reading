"""
Demo script for GAN tutorial implementation.
Showcases minimax game theory, adversarial training dynamics, and Nash equilibrium analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.computation.generative_ops.gan_operations import GANOperations, GANParameters
from src.content.advanced.gan_tutorial import create_gan_tutorial
from src.visualization.operation_viz.gan_visualizer import create_gan_visualization_suite


def demo_gan_operations():
    """Demonstrate GAN mathematical operations."""
    print("=== GAN Mathematical Operations Demo ===\n")
    
    # Initialize GAN operations
    params = GANParameters(
        noise_dim=100, 
        data_dim=784, 
        generator_hidden_dim=256,
        discriminator_hidden_dim=256,
        batch_size=16
    )
    gan_ops = GANOperations(params)
    
    # Generate sample data
    real_data = np.random.randn(params.batch_size, params.data_dim)
    noise = np.random.randn(params.batch_size, params.noise_dim)
    
    print(f"Real data shape: {real_data.shape}")
    print(f"Noise shape: {noise.shape}")
    
    # Generator forward pass
    fake_data = gan_ops.generator_forward(noise)
    print(f"Generated data shape: {fake_data.shape}")
    print(f"Generated data range: [{np.min(fake_data):.3f}, {np.max(fake_data):.3f}]")
    
    # Discriminator evaluation
    D_real = gan_ops.discriminator_forward(real_data)
    D_fake = gan_ops.discriminator_forward(fake_data)
    print(f"D(real) mean: {np.mean(D_real):.3f} ± {np.std(D_real):.3f}")
    print(f"D(fake) mean: {np.mean(D_fake):.3f} ± {np.std(D_fake):.3f}")
    
    # Loss computations
    D_loss = gan_ops.compute_discriminator_loss(real_data, fake_data)
    G_loss = gan_ops.compute_generator_loss(fake_data)
    
    print(f"\nLoss Components:")
    print(f"  Discriminator Loss: {D_loss['total_loss']:.3f}")
    print(f"    - Real Loss: {D_loss['real_loss']:.3f}")
    print(f"    - Fake Loss: {D_loss['fake_loss']:.3f}")
    print(f"  Generator Loss: {G_loss['alternative_loss']:.3f}")
    
    # Minimax objective
    minimax = gan_ops.compute_minimax_objective(real_data, noise)
    print(f"  Value Function V(D,G): {minimax['value_function']:.3f}")
    
    return gan_ops, D_loss, G_loss, minimax


def demo_adversarial_training_steps():
    """Demonstrate step-by-step adversarial training."""
    print("\n=== Adversarial Training Steps Demo ===\n")
    
    params = GANParameters(noise_dim=100, data_dim=784, generator_hidden_dim=256, discriminator_hidden_dim=256)
    gan_ops = GANOperations(params)
    
    steps = gan_ops.generate_adversarial_training_steps()
    
    for step in steps:
        print(f"Step {step.step_number}: {step.operation_name}")
        print(f"  Description: {step.operation_description}")
        print(f"  Input keys: {list(step.input_values.keys())}")
        print(f"  Output keys: {list(step.output_values.keys())}")
        if step.visualization_hints:
            print(f"  Visualization: {step.visualization_hints}")
        print()


def demo_nash_equilibrium_analysis():
    """Demonstrate Nash equilibrium analysis and convergence."""
    print("=== Nash Equilibrium Analysis Demo ===\n")
    
    params = GANParameters(noise_dim=100, data_dim=784, generator_hidden_dim=256, discriminator_hidden_dim=256)
    gan_ops = GANOperations(params)
    
    # Generate sample real data
    real_data = np.random.randn(32, params.data_dim)
    
    print("Running adversarial training simulation...")
    history = gan_ops.analyze_nash_equilibrium(real_data, num_iterations=50)
    
    print(f"Training completed with {len(history['D_loss'])} iterations")
    
    # Analyze final state
    final_D_real = history['D_real'][-1]
    final_D_fake = history['D_fake'][-1]
    final_D_loss = history['D_loss'][-1]
    final_G_loss = history['G_loss'][-1]
    
    print(f"\nFinal Training State:")
    print(f"  D(real) = {final_D_real:.3f} (ideal: close to 0.5)")
    print(f"  D(fake) = {final_D_fake:.3f} (ideal: close to 0.5)")
    print(f"  D Loss = {final_D_loss:.3f}")
    print(f"  G Loss = {final_G_loss:.3f}")
    
    # Nash equilibrium proximity
    nash_distance = abs(final_D_real - 0.5) + abs(final_D_fake - 0.5)
    print(f"  Nash Distance = {nash_distance:.3f} (ideal: close to 0)")
    
    # Training stability
    recent_D_real = history['D_real'][-10:]
    recent_D_fake = history['D_fake'][-10:]
    stability = np.std(recent_D_real) + np.std(recent_D_fake)
    print(f"  Training Stability = {stability:.3f} (lower is better)")
    
    return history


def demo_mode_collapse_analysis():
    """Demonstrate mode collapse detection and analysis."""
    print("\n=== Mode Collapse Analysis Demo ===\n")
    
    params = GANParameters(noise_dim=100, data_dim=784, generator_hidden_dim=256, discriminator_hidden_dim=256)
    gan_ops = GANOperations(params)
    
    # Create multimodal real data
    real_data = np.random.randn(100, params.data_dim)
    
    print("Analyzing generator diversity...")
    collapse_info = gan_ops.demonstrate_mode_collapse(real_data, num_samples=50)
    
    print(f"Generated {collapse_info['generated_samples'].shape[0]} samples")
    print(f"Diversity Metrics:")
    print(f"  Mean Generated Distance: {collapse_info['mean_generated_distance']:.3f}")
    print(f"  Mean Real Distance: {collapse_info['mean_real_distance']:.3f}")
    print(f"  Diversity Ratio: {collapse_info['diversity_ratio']:.3f}")
    
    # Mode collapse indicators
    if collapse_info['diversity_ratio'] < 0.5:
        print("  ⚠️  Potential mode collapse detected (low diversity ratio)")
    else:
        print("  ✅ Good diversity maintained")
    
    if collapse_info['mean_generated_distance'] < 0.1:
        print("  ⚠️  Generated samples are very similar (possible collapse)")
    else:
        print("  ✅ Generated samples show good variation")
    
    return collapse_info


def demo_gan_tutorial_content():
    """Demonstrate GAN tutorial content structure."""
    print("\n=== GAN Tutorial Content Demo ===\n")
    
    gan_concept = create_gan_tutorial()
    
    print(f"Tutorial: {gan_concept.title}")
    print(f"Concept ID: {gan_concept.concept_id}")
    print(f"Difficulty Level: {gan_concept.difficulty_level}/5")
    print(f"Prerequisites: {', '.join(gan_concept.prerequisites)}")
    
    print(f"\nEquations ({len(gan_concept.equations)}):")
    for eq in gan_concept.equations:
        print(f"  - {eq.equation_id}: {len(eq.derivation_steps)} derivation steps")
        print(f"    Variables: {list(eq.variables.keys())}")
        print(f"    Complexity: {eq.complexity_level}/5")
    
    print(f"\nExplanations ({len(gan_concept.explanations)}):")
    for exp in gan_concept.explanations:
        print(f"  - {exp.explanation_type} (level {exp.mathematical_level})")
        print(f"    Length: {len(exp.content)} characters")
        print(f"    Prerequisites: {exp.prerequisites}")
    
    print(f"\nVisualizations ({len(gan_concept.visualizations)}):")
    for viz in gan_concept.visualizations:
        print(f"  - {viz.visualization_id}: {viz.visualization_type}")
        print(f"    Interactive: {viz.interactive}")
        print(f"    Prerequisites: {viz.prerequisites}")
    
    print(f"\nLearning Objectives ({len(gan_concept.learning_objectives)}):")
    for i, obj in enumerate(gan_concept.learning_objectives, 1):
        print(f"  {i}. {obj}")
    
    print(f"\nAssessment Criteria ({len(gan_concept.assessment_criteria)}):")
    for i, criterion in enumerate(gan_concept.assessment_criteria, 1):
        print(f"  {i}. {criterion}")


def demo_gan_visualizations():
    """Demonstrate GAN visualization components."""
    print("\n=== GAN Visualization Demo ===\n")
    
    viz_suite = create_gan_visualization_suite()
    
    # Architecture visualization
    arch_viz = viz_suite['architecture_visualizer']
    architecture = arch_viz.create_adversarial_architecture(100, 784, 256, 256)
    
    print("Adversarial Architecture:")
    print(f"  Generator layers: {len(architecture['generator']['layers'])}")
    for layer in architecture['generator']['layers']:
        print(f"    - {layer['name']}: size={layer['size']}, pos={layer['pos']}")
    
    print(f"  Discriminator layers: {len(architecture['discriminator']['layers'])}")
    for layer in architecture['discriminator']['layers']:
        print(f"    - {layer['name']}: size={layer['size']}, pos={layer['pos']}")
    
    # Minimax game visualization
    D_values = np.array([0.8, 0.6, 0.7, 0.5])
    G_values = np.array([0.3, 0.4, 0.2, 0.5])
    V_values = np.array([-0.5, -0.8, -0.6, -0.3])
    
    minimax_viz = arch_viz.create_minimax_game_visualization(D_values, G_values, V_values)
    print(f"\nMinimax Game Visualization:")
    print(f"  Operation: {minimax_viz.operation_type}")
    print(f"  Input matrices: {len(minimax_viz.input_matrices)}")
    print(f"  Animation frames: {len(minimax_viz.animation_sequence)}")
    
    for frame in minimax_viz.animation_sequence:
        print(f"    Frame {frame.frame_number}: {frame.description}")
    
    # Training dynamics visualization
    dynamics_viz = viz_suite['training_dynamics_visualizer']
    
    # Mock training history
    history = {
        'D_loss': [1.5, 1.2, 1.0, 0.8, 0.7, 0.6],
        'G_loss': [2.0, 1.8, 1.5, 1.2, 1.0, 0.9],
        'D_real': [0.9, 0.8, 0.7, 0.6, 0.55, 0.52],
        'D_fake': [0.1, 0.2, 0.3, 0.4, 0.45, 0.48],
        'value_function': [-0.8, -0.6, -0.4, -0.2, -0.1, -0.05]
    }
    
    training_data = dynamics_viz.create_training_curves(history)
    convergence_metrics = dynamics_viz.analyze_convergence_patterns(training_data)
    
    print(f"\nTraining Dynamics Analysis:")
    print(f"  Training iterations: {len(training_data['iterations'])}")
    print(f"  Nash distance: {convergence_metrics['nash_distance']:.3f}")
    print(f"  Stability score: {convergence_metrics['stability_score']:.3f}")
    print(f"  Convergence score: {convergence_metrics['convergence_score']:.3f}")
    print(f"  Mode collapse risk: {convergence_metrics['mode_collapse_risk']:.3f}")
    
    # Mode collapse visualization
    collapse_viz = viz_suite['mode_collapse_visualizer']
    
    # Create sample data
    real_samples = np.random.randn(100, 2)
    generated_samples = np.random.randn(50, 2) * 0.8  # Slightly less diverse
    
    collapse_data = collapse_viz.create_mode_collapse_analysis(real_samples, generated_samples)
    diversity_metrics = collapse_viz.compute_diversity_metrics(generated_samples)
    
    print(f"\nMode Collapse Analysis:")
    print(f"  Real samples: {collapse_data['real_samples_2d'].shape}")
    print(f"  Generated samples: {collapse_data['generated_samples_2d'].shape}")
    print(f"  Number of modes: {len(collapse_data['mode_centers'])}")
    print(f"  Collapsed modes: {len(collapse_data['collapsed_modes'])}")
    print(f"  Diversity score: {diversity_metrics['diversity_score']:.3f}")
    print(f"  Effective sample size: {diversity_metrics['effective_sample_size']}")
    
    # Interactive components
    interactive = viz_suite['interactive_components']
    
    controller = interactive.create_training_controller()
    print(f"\nInteractive Training Controller:")
    print(f"  Hyperparameters: {list(controller['hyperparameters'].keys())}")
    print(f"  Training controls: {list(controller['training_controls'].keys())}")
    print(f"  Visualization options: {list(controller['visualization_options'].keys())}")
    
    explorer = interactive.create_architecture_explorer()
    print(f"\nArchitecture Explorer:")
    print(f"  Generator options: {list(explorer['generator_architecture'].keys())}")
    print(f"  Discriminator options: {list(explorer['discriminator_architecture'].keys())}")
    print(f"  Loss variants: {list(explorer['loss_variants'].keys())}")


def main():
    """Run all GAN tutorial demos."""
    print("GAN Tutorial Implementation Demo")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Demo mathematical operations
        gan_ops, D_loss, G_loss, minimax = demo_gan_operations()
        
        # Demo adversarial training steps
        demo_adversarial_training_steps()
        
        # Demo Nash equilibrium analysis
        training_history = demo_nash_equilibrium_analysis()
        
        # Demo mode collapse analysis
        collapse_info = demo_mode_collapse_analysis()
        
        # Demo tutorial content
        demo_gan_tutorial_content()
        
        # Demo visualizations
        demo_gan_visualizations()
        
        print("\n" + "=" * 50)
        print("GAN Tutorial Demo completed successfully!")
        print("All components are working correctly.")
        
        # Summary statistics
        print(f"\nDemo Summary:")
        print(f"  Final D(real): {training_history['D_real'][-1]:.3f}")
        print(f"  Final D(fake): {training_history['D_fake'][-1]:.3f}")
        print(f"  Nash distance: {abs(training_history['D_real'][-1] - 0.5) + abs(training_history['D_fake'][-1] - 0.5):.3f}")
        print(f"  Diversity ratio: {collapse_info['diversity_ratio']:.3f}")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()