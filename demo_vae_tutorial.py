"""
Demo script for VAE tutorial implementation.
Showcases ELBO derivation, reparameterization trick, and latent space visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.computation.generative_ops.vae_operations import VAEOperations, VAEParameters
from src.content.advanced.vae_tutorial import create_vae_tutorial
from src.visualization.operation_viz.vae_visualizer import create_vae_visualization_suite


def demo_vae_operations():
    """Demonstrate VAE mathematical operations."""
    print("=== VAE Mathematical Operations Demo ===\n")
    
    # Initialize VAE operations
    params = VAEParameters(input_dim=784, latent_dim=20, hidden_dim=400, batch_size=8)
    vae_ops = VAEOperations(params)
    
    # Generate sample data
    x = np.random.randn(params.batch_size, params.input_dim)
    print(f"Input data shape: {x.shape}")
    
    # Forward pass through encoder
    mu, log_var = vae_ops.encoder_forward(x)
    print(f"Encoder output - μ shape: {mu.shape}, log σ² shape: {log_var.shape}")
    
    # Reparameterization trick
    z, epsilon, sigma = vae_ops.reparameterization_trick(mu, log_var)
    print(f"Latent sample z shape: {z.shape}")
    print(f"Sample statistics - μ mean: {np.mean(mu):.3f}, σ mean: {np.mean(sigma):.3f}")
    
    # Decoder reconstruction
    x_recon = vae_ops.decoder_forward(z)
    print(f"Reconstruction shape: {x_recon.shape}")
    
    # ELBO computation
    elbo_components = vae_ops.compute_elbo_components(x, mu, log_var, x_recon)
    print(f"\nELBO Components:")
    print(f"  Reconstruction Loss: {elbo_components['reconstruction_loss']:.3f}")
    print(f"  KL Divergence: {elbo_components['kl_divergence']:.3f}")
    print(f"  ELBO: {elbo_components['elbo']:.3f}")
    print(f"  Negative ELBO (loss): {elbo_components['negative_elbo']:.3f}")
    
    return vae_ops, elbo_components


def demo_vae_derivation_steps():
    """Demonstrate step-by-step ELBO derivation."""
    print("\n=== VAE ELBO Derivation Steps ===\n")
    
    params = VAEParameters(input_dim=784, latent_dim=20, hidden_dim=400)
    vae_ops = VAEOperations(params)
    
    steps = vae_ops.generate_elbo_derivation_steps()
    
    for step in steps:
        print(f"Step {step.step_number}: {step.operation_name}")
        print(f"  Description: {step.operation_description}")
        print(f"  Input keys: {list(step.input_values.keys())}")
        print(f"  Output keys: {list(step.output_values.keys())}")
        if step.visualization_hints:
            print(f"  Visualization: {step.visualization_hints}")
        print()


def demo_latent_interpolation():
    """Demonstrate latent space interpolation."""
    print("=== Latent Space Interpolation Demo ===\n")
    
    params = VAEParameters(input_dim=784, latent_dim=20, hidden_dim=400)
    vae_ops = VAEOperations(params)
    
    # Create two random latent points
    z1 = np.random.randn(params.latent_dim)
    z2 = np.random.randn(params.latent_dim)
    
    print(f"Interpolating between:")
    print(f"  z1: [{z1[0]:.3f}, {z1[1]:.3f}, ..., {z1[-1]:.3f}]")
    print(f"  z2: [{z2[0]:.3f}, {z2[1]:.3f}, ..., {z2[-1]:.3f}]")
    
    # Generate interpolation
    interpolations = vae_ops.generate_latent_interpolation(z1, z2, num_steps=5)
    
    print(f"\nGenerated {len(interpolations)} interpolation steps")
    for i, interp in enumerate(interpolations):
        print(f"  Step {i}: reconstruction shape {interp.shape}, mean: {np.mean(interp):.3f}")


def demo_vae_tutorial_content():
    """Demonstrate VAE tutorial content structure."""
    print("\n=== VAE Tutorial Content Demo ===\n")
    
    vae_concept = create_vae_tutorial()
    
    print(f"Tutorial: {vae_concept.title}")
    print(f"Concept ID: {vae_concept.concept_id}")
    print(f"Difficulty Level: {vae_concept.difficulty_level}/5")
    print(f"Prerequisites: {', '.join(vae_concept.prerequisites)}")
    
    print(f"\nEquations ({len(vae_concept.equations)}):")
    for eq in vae_concept.equations:
        print(f"  - {eq.equation_id}: {len(eq.derivation_steps)} derivation steps")
        print(f"    Variables: {list(eq.variables.keys())}")
    
    print(f"\nExplanations ({len(vae_concept.explanations)}):")
    for exp in vae_concept.explanations:
        print(f"  - {exp.explanation_type} (level {exp.mathematical_level})")
        print(f"    Length: {len(exp.content)} characters")
    
    print(f"\nVisualizations ({len(vae_concept.visualizations)}):")
    for viz in vae_concept.visualizations:
        print(f"  - {viz.visualization_id}: {viz.visualization_type}")
        print(f"    Interactive: {viz.interactive}")
    
    print(f"\nLearning Objectives ({len(vae_concept.learning_objectives)}):")
    for i, obj in enumerate(vae_concept.learning_objectives, 1):
        print(f"  {i}. {obj}")


def demo_vae_visualizations():
    """Demonstrate VAE visualization components."""
    print("\n=== VAE Visualization Demo ===\n")
    
    viz_suite = create_vae_visualization_suite()
    
    # Architecture visualization
    arch_viz = viz_suite['architecture_visualizer']
    layers = arch_viz.create_architecture_diagram(784, 20, 400)
    
    print("Architecture Layers:")
    for layer_name, layer_info in layers.items():
        print(f"  {layer_name}: size={layer_info['size']}, pos={layer_info['pos']}")
    
    # Reparameterization visualization
    mu = np.array([1.0, 0.5, -0.2])
    log_var = np.array([0.0, -1.0, 0.5])
    epsilon = np.array([0.5, -1.0, 1.5])
    
    reparam_viz = arch_viz.create_reparameterization_visualization(mu, log_var, epsilon)
    print(f"\nReparameterization Visualization:")
    print(f"  Operation: {reparam_viz.operation_type}")
    print(f"  Input matrices: {len(reparam_viz.input_matrices)}")
    print(f"  Animation frames: {len(reparam_viz.animation_sequence)}")
    
    for frame in reparam_viz.animation_sequence:
        print(f"    Frame {frame.frame_number}: {frame.description}")
    
    # Latent space visualization
    latent_viz = viz_suite['latent_space_visualizer']
    latent_samples = np.random.randn(50, 2)
    plot_data = latent_viz.create_latent_space_plot(latent_samples)
    
    print(f"\nLatent Space Plot Data:")
    print(f"  Latent points shape: {plot_data['latent_points'].shape}")
    print(f"  Prior grid shape: {plot_data['prior_grid_x'].shape}")
    print(f"  Prior density range: [{np.min(plot_data['prior_density']):.3f}, {np.max(plot_data['prior_density']):.3f}]")
    
    # Interactive components
    interactive = viz_suite['interactive_components']
    param_sliders = interactive.create_parameter_slider_data({
        'beta': (0.1, 10.0),
        'latent_dim': (2, 100),
        'learning_rate': (1e-5, 1e-2)
    })
    
    print(f"\nInteractive Parameter Sliders:")
    for param, config in param_sliders.items():
        print(f"  {param}: [{config['min']}, {config['max']}], default={config['default']}")


def main():
    """Run all VAE tutorial demos."""
    print("VAE Tutorial Implementation Demo")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Demo mathematical operations
        vae_ops, elbo_components = demo_vae_operations()
        
        # Demo derivation steps
        demo_vae_derivation_steps()
        
        # Demo latent interpolation
        demo_latent_interpolation()
        
        # Demo tutorial content
        demo_vae_tutorial_content()
        
        # Demo visualizations
        demo_vae_visualizations()
        
        print("\n" + "=" * 50)
        print("VAE Tutorial Demo completed successfully!")
        print("All components are working correctly.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()