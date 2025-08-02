"""
Demo script for Diffusion Models tutorial implementation.
Showcases forward/reverse diffusion processes, denoising objectives, and sampling procedures.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.computation.generative_ops.diffusion_operations import DiffusionOperations, DiffusionParameters
from src.content.advanced.diffusion_tutorial import create_diffusion_tutorial
from src.visualization.operation_viz.diffusion_visualizer import create_diffusion_visualization_suite


def demo_diffusion_operations():
    """Demonstrate diffusion model mathematical operations."""
    print("=== Diffusion Models Mathematical Operations Demo ===\n")
    
    # Initialize diffusion operations
    params = DiffusionParameters(
        data_dim=784,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        model_hidden_dim=256,
        batch_size=16
    )
    diffusion_ops = DiffusionOperations(params)
    
    # Generate sample data
    x_0 = np.random.randn(params.batch_size, params.data_dim)
    print(f"Clean data shape: {x_0.shape}")
    print(f"Clean data statistics: mean={np.mean(x_0):.3f}, std={np.std(x_0):.3f}")
    
    # Noise schedule analysis
    print(f"\nNoise Schedule:")
    print(f"  β_start: {params.beta_start}")
    print(f"  β_end: {params.beta_end}")
    print(f"  Number of timesteps: {params.num_timesteps}")
    print(f"  α_bar at t=100: {diffusion_ops.alpha_bars[100]:.4f}")
    print(f"  α_bar at t=500: {diffusion_ops.alpha_bars[500]:.4f}")
    print(f"  α_bar at t=900: {diffusion_ops.alpha_bars[900]:.4f}")
    
    # Forward diffusion demonstration
    t_demo = 500
    x_t, true_noise = diffusion_ops.forward_diffusion_direct(x_0, t_demo)
    print(f"\nForward Diffusion at t={t_demo}:")
    print(f"  Noisy data statistics: mean={np.mean(x_t):.3f}, std={np.std(x_t):.3f}")
    print(f"  Signal level: {np.sqrt(diffusion_ops.alpha_bars[t_demo]):.3f}")
    print(f"  Noise level: {np.sqrt(1 - diffusion_ops.alpha_bars[t_demo]):.3f}")
    
    # Noise prediction
    predicted_noise = diffusion_ops.predict_noise(x_t, t_demo)
    print(f"\nNoise Prediction:")
    print(f"  True noise statistics: mean={np.mean(true_noise):.3f}, std={np.std(true_noise):.3f}")
    print(f"  Predicted noise statistics: mean={np.mean(predicted_noise):.3f}, std={np.std(predicted_noise):.3f}")
    
    # Denoising loss
    loss_info = diffusion_ops.compute_denoising_loss(x_0, t_demo)
    print(f"\nDenoising Loss:")
    print(f"  MSE Loss: {loss_info['mse_loss']:.3f}")
    print(f"  Signal-to-Noise Ratio: {loss_info['signal_to_noise_ratio']:.3f}")
    print(f"  Noise Level: {loss_info['noise_level']:.3f}")
    
    # Reverse diffusion step
    x_t_minus_1 = diffusion_ops.reverse_diffusion_step(x_t, t_demo, predicted_noise)
    print(f"\nReverse Diffusion:")
    print(f"  Denoised data statistics: mean={np.mean(x_t_minus_1):.3f}, std={np.std(x_t_minus_1):.3f}")
    
    return diffusion_ops, loss_info


def demo_diffusion_process_steps():
    """Demonstrate step-by-step diffusion process."""
    print("\n=== Diffusion Process Steps Demo ===\n")
    
    params = DiffusionParameters(data_dim=784, num_timesteps=1000, model_hidden_dim=256)
    diffusion_ops = DiffusionOperations(params)
    
    steps = diffusion_ops.generate_diffusion_process_steps()
    
    for step in steps:
        print(f"Step {step.step_number}: {step.operation_name}")
        print(f"  Description: {step.operation_description}")
        print(f"  Input keys: {list(step.input_values.keys())}")
        print(f"  Output keys: {list(step.output_values.keys())}")
        if step.visualization_hints:
            print(f"  Visualization: {step.visualization_hints}")
        print()


def demo_sampling_procedures():
    """Demonstrate DDPM and DDIM sampling procedures."""
    print("=== Sampling Procedures Demo ===\n")
    
    params = DiffusionParameters(data_dim=784, num_timesteps=1000, model_hidden_dim=256)
    diffusion_ops = DiffusionOperations(params)
    
    shape = (4, params.data_dim)
    
    # DDPM sampling
    print("DDPM Sampling (stochastic):")
    ddpm_samples = diffusion_ops.sample_ddpm(shape, num_steps=20)
    print(f"  Generated {len(ddmp_samples)} samples")
    print(f"  Initial noise: mean={np.mean(ddpm_samples[0]):.3f}, std={np.std(ddpm_samples[0]):.3f}")
    print(f"  Final sample: mean={np.mean(ddpm_samples[-1]):.3f}, std={np.std(ddpm_samples[-1]):.3f}")
    
    # DDIM sampling
    print(f"\nDDIM Sampling (deterministic, η=0):")
    ddim_samples_det = diffusion_ops.sample_ddim(shape, num_steps=10, eta=0.0)
    print(f"  Generated {len(ddim_samples_det)} samples")
    print(f"  Initial noise: mean={np.mean(ddim_samples_det[0]):.3f}, std={np.std(ddim_samples_det[0]):.3f}")
    print(f"  Final sample: mean={np.mean(ddim_samples_det[-1]):.3f}, std={np.std(ddim_samples_det[-1]):.3f}")
    
    print(f"\nDDIM Sampling (stochastic, η=1):")
    ddim_samples_stoch = diffusion_ops.sample_ddim(shape, num_steps=10, eta=1.0)
    print(f"  Generated {len(ddim_samples_stoch)} samples")
    print(f"  Final sample: mean={np.mean(ddim_samples_stoch[-1]):.3f}, std={np.std(ddim_samples_stoch[-1]):.3f}")
    
    return ddpm_samples, ddim_samples_det, ddim_samples_stoch


def demo_noise_schedule_analysis():
    """Demonstrate noise schedule analysis."""
    print("\n=== Noise Schedule Analysis Demo ===\n")
    
    params = DiffusionParameters(data_dim=784, num_timesteps=1000)
    diffusion_ops = DiffusionOperations(params)
    
    analysis = diffusion_ops.analyze_noise_schedule()
    
    print("Noise Schedule Properties:")
    print(f"  Total timesteps: {len(analysis['timesteps'])}")
    print(f"  β range: [{np.min(analysis['betas']):.6f}, {np.max(analysis['betas']):.6f}]")
    print(f"  α_bar range: [{np.min(analysis['alpha_bars']):.6f}, {np.max(analysis['alpha_bars']):.6f}]")
    print(f"  SNR range: [{np.min(analysis['snr']):.3f}, {np.max(analysis['snr']):.3f}]")
    
    # Key timesteps analysis
    key_timesteps = [100, 250, 500, 750, 900]
    print(f"\nKey Timesteps Analysis:")
    for t in key_timesteps:
        snr = analysis['snr'][t]
        noise_level = analysis['noise_levels'][t]
        signal_level = analysis['signal_levels'][t]
        print(f"  t={t:3d}: SNR={snr:6.3f}, Signal={signal_level:.3f}, Noise={noise_level:.3f}")
    
    return analysis


def demo_variational_lower_bound():
    """Demonstrate VLB computation."""
    print("\n=== Variational Lower Bound Demo ===\n")
    
    params = DiffusionParameters(data_dim=784, num_timesteps=100, model_hidden_dim=256)  # Smaller for demo
    diffusion_ops = DiffusionOperations(params)
    
    x_0 = np.random.randn(8, params.data_dim)
    
    print("Computing Variational Lower Bound...")
    vlb_info = diffusion_ops.compute_variational_lower_bound(x_0)
    
    print(f"VLB Components:")
    print(f"  Total VLB Loss: {vlb_info['total_vlb_loss']:.3f}")
    print(f"  Reconstruction Loss (L_0): {vlb_info['reconstruction_loss']:.3f}")
    print(f"  Mean Denoising Loss: {vlb_info['mean_denoising_loss']:.3f}")
    print(f"  Number of denoising terms: {len(vlb_info['denoising_losses'])}")
    
    # Analyze denoising losses across timesteps
    denoising_losses = vlb_info['denoising_losses']
    print(f"\nDenoising Loss Statistics:")
    print(f"  Min: {np.min(denoising_losses):.3f}")
    print(f"  Max: {np.max(denoising_losses):.3f}")
    print(f"  Std: {np.std(denoising_losses):.3f}")
    
    return vlb_info


def demo_diffusion_tutorial_content():
    """Demonstrate diffusion tutorial content structure."""
    print("\n=== Diffusion Tutorial Content Demo ===\n")
    
    diffusion_concept = create_diffusion_tutorial()
    
    print(f"Tutorial: {diffusion_concept.title}")
    print(f"Concept ID: {diffusion_concept.concept_id}")
    print(f"Difficulty Level: {diffusion_concept.difficulty_level}/5")
    print(f"Prerequisites: {', '.join(diffusion_concept.prerequisites)}")
    
    print(f"\nEquations ({len(diffusion_concept.equations)}):")
    for eq in diffusion_concept.equations:
        print(f"  - {eq.equation_id}: {len(eq.derivation_steps)} derivation steps")
        print(f"    Variables: {list(eq.variables.keys())}")
        print(f"    Complexity: {eq.complexity_level}/5")
    
    print(f"\nExplanations ({len(diffusion_concept.explanations)}):")
    for exp in diffusion_concept.explanations:
        print(f"  - {exp.explanation_type} (level {exp.mathematical_level})")
        print(f"    Length: {len(exp.content)} characters")
        print(f"    Prerequisites: {exp.prerequisites}")
    
    print(f"\nVisualizations ({len(diffusion_concept.visualizations)}):")
    for viz in diffusion_concept.visualizations:
        print(f"  - {viz.visualization_id}: {viz.visualization_type}")
        print(f"    Interactive: {viz.interactive}")
        print(f"    Prerequisites: {viz.prerequisites}")
    
    print(f"\nLearning Objectives ({len(diffusion_concept.learning_objectives)}):")
    for i, obj in enumerate(diffusion_concept.learning_objectives, 1):
        print(f"  {i}. {obj}")
    
    print(f"\nAssessment Criteria ({len(diffusion_concept.assessment_criteria)}):")
    for i, criterion in enumerate(diffusion_concept.assessment_criteria, 1):
        print(f"  {i}. {criterion}")


def demo_diffusion_visualizations():
    """Demonstrate diffusion visualization components."""
    print("\n=== Diffusion Visualization Demo ===\n")
    
    viz_suite = create_diffusion_visualization_suite()
    
    # Process visualization
    process_viz = viz_suite['process_visualizer']
    
    # Noise schedule visualization
    schedule_data = process_viz.create_noise_schedule_visualization(num_timesteps=100)
    print("Noise Schedule Visualization:")
    print(f"  Timesteps: {len(schedule_data['timesteps'])}")
    print(f"  Linear β range: [{np.min(schedule_data['beta_linear']):.6f}, {np.max(schedule_data['beta_linear']):.6f}]")
    print(f"  Cosine β range: [{np.min(schedule_data['beta_cosine']):.6f}, {np.max(schedule_data['beta_cosine']):.6f}]")
    print(f"  Linear SNR range: [{np.min(schedule_data['snr_linear']):.3f}, {np.max(schedule_data['snr_linear']):.3f}]")
    print(f"  Cosine SNR range: [{np.min(schedule_data['snr_cosine']):.3f}, {np.max(schedule_data['snr_cosine']):.3f}]")
    
    # Diffusion process animation
    x_0 = np.random.randn(20)
    timesteps = [0, 25, 50, 75, 100]
    alpha_bars = np.linspace(1.0, 0.01, 101)
    
    animation = process_viz.create_diffusion_process_animation(x_0, timesteps, alpha_bars)
    print(f"\nDiffusion Process Animation:")
    print(f"  Operation type: {animation.operation_type}")
    print(f"  Input matrices: {len(animation.input_matrices)}")
    print(f"  Animation frames: {len(animation.animation_sequence)}")
    
    for i, frame in enumerate(animation.animation_sequence[:3]):  # Show first 3 frames
        print(f"    Frame {frame.frame_number}: {frame.description}")
    
    # Sampling visualization
    sampling_viz = viz_suite['sampling_visualizer']
    
    # Mock sampling trajectories
    trajectories = {
        'DDPM': [np.random.randn(20) * (1 - i/19) for i in range(20)],  # Decreasing noise
        'DDIM': [np.random.randn(20) * (1 - i/9) for i in range(10)]    # Fewer steps
    }
    
    comparison_data = sampling_viz.create_sampling_comparison(trajectories)
    print(f"\nSampling Methods Comparison:")
    for method, data in comparison_data.items():
        print(f"  {method}:")
        print(f"    Steps: {data['num_steps']}")
        print(f"    Final variance: {data['sample_variance']:.3f}")
        print(f"    Trajectory smoothness: {data['trajectory_smoothness']:.3f}")
    
    # DDPM vs DDIM analysis
    ddpm_ddim_analysis = sampling_viz.create_ddpm_vs_ddim_analysis(num_timesteps=1000)
    print(f"\nDDPM vs DDIM Analysis:")
    print(f"  DDPM steps: {len(ddpm_ddim_analysis['ddpm_steps'])}")
    print(f"  DDIM steps: {len(ddpm_ddim_analysis['ddim_steps'])}")
    print(f"  Quality comparison: {ddpm_ddim_analysis['quality_comparison']}")
    print(f"  Speed comparison: {ddpm_ddim_analysis['speed_comparison']}")
    
    # Loss visualization
    loss_viz = viz_suite['loss_visualizer']
    
    # Mock loss landscape
    timesteps = np.arange(50)
    losses = 2.0 * np.exp(-0.5 * ((timesteps - 25) / 10)**2) + 0.5  # Bell curve + baseline
    
    landscape = loss_viz.create_loss_landscape_analysis(timesteps, losses)
    print(f"\nLoss Landscape Analysis:")
    print(f"  Timesteps analyzed: {len(landscape['timesteps'])}")
    print(f"  Mean loss: {landscape['mean_loss']:.3f}")
    print(f"  Loss std: {landscape['std_loss']:.3f}")
    print(f"  Challenging timesteps: {len(landscape['challenging_timesteps'])}")
    
    # VLB decomposition
    vlb_data = loss_viz.create_vlb_decomposition(num_timesteps=100)
    print(f"\nVLB Decomposition:")
    print(f"  L_T term: {vlb_data['L_T_term']:.3f}")
    print(f"  L_0 term: {vlb_data['L_0_term']:.3f}")
    print(f"  Total VLB: {vlb_data['total_vlb']:.3f}")
    print(f"  L_simple: {vlb_data['L_simple']:.3f}")
    print(f"  Component values: {vlb_data['component_values']}")
    
    # Interactive components
    interactive = viz_suite['interactive_components']
    
    noise_controller = interactive.create_noise_schedule_controller()
    print(f"\nInteractive Noise Schedule Controller:")
    print(f"  Schedule parameters: {list(noise_controller['schedule_parameters'].keys())}")
    print(f"  Cosine parameters: {list(noise_controller['cosine_parameters'].keys())}")
    print(f"  Visualization options: {list(noise_controller['visualization_options'].keys())}")
    
    sampling_controller = interactive.create_sampling_controller()
    print(f"\nInteractive Sampling Controller:")
    print(f"  Sampling methods: {list(sampling_controller['sampling_method'].keys())}")
    print(f"  Generation parameters: {list(sampling_controller['generation_parameters'].keys())}")
    print(f"  Visualization options: {list(sampling_controller['visualization_options'].keys())}")


def main():
    """Run all diffusion models tutorial demos."""
    print("Diffusion Models Tutorial Implementation Demo")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Demo mathematical operations
        diffusion_ops, loss_info = demo_diffusion_operations()
        
        # Demo diffusion process steps
        demo_diffusion_process_steps()
        
        # Demo sampling procedures
        ddpm_samples, ddim_det, ddim_stoch = demo_sampling_procedures()
        
        # Demo noise schedule analysis
        schedule_analysis = demo_noise_schedule_analysis()
        
        # Demo VLB computation
        vlb_info = demo_variational_lower_bound()
        
        # Demo tutorial content
        demo_diffusion_tutorial_content()
        
        # Demo visualizations
        demo_diffusion_visualizations()
        
        print("\n" + "=" * 60)
        print("Diffusion Models Tutorial Demo completed successfully!")
        print("All components are working correctly.")
        
        # Summary statistics
        print(f"\nDemo Summary:")
        print(f"  Denoising MSE Loss: {loss_info['mse_loss']:.3f}")
        print(f"  Signal-to-Noise Ratio: {loss_info['signal_to_noise_ratio']:.3f}")
        print(f"  VLB Total Loss: {vlb_info['total_vlb_loss']:.3f}")
        print(f"  DDPM samples generated: {len(ddpm_samples)}")
        print(f"  DDIM samples generated: {len(ddim_det)}")
        print(f"  Noise schedule SNR range: [{np.min(schedule_analysis['snr']):.3f}, {np.max(schedule_analysis['snr']):.3f}]")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()