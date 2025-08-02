"""
Diffusion Models tutorial content with mathematical explanations.
Covers forward/reverse diffusion processes, denoising objectives, and sampling procedures.
"""

from typing import List
from ...core.models import (
    MathematicalConcept, Equation, VariableDefinition, DerivationStep,
    Explanation, Visualization, VisualizationData
)
from ...computation.generative_ops.diffusion_operations import create_diffusion_numerical_example


def create_diffusion_equations() -> List[Equation]:
    """Create diffusion model mathematical equations with detailed derivations."""
    
    equations = []
    
    # Forward Diffusion Process
    forward_variables = {
        "x_0": VariableDefinition(
            name="x_0",
            description="Original clean data",
            data_type="vector",
            color_code="#3498db"
        ),
        "x_t": VariableDefinition(
            name="x_t",
            description="Noisy data at timestep t",
            data_type="vector",
            color_code="#e67e22"
        ),
        "beta_t": VariableDefinition(
            name="β_t",
            description="Noise schedule at timestep t",
            data_type="scalar",
            constraints="0 < β_t < 1",
            color_code="#f39c12"
        ),
        "alpha_t": VariableDefinition(
            name="α_t",
            description="Signal retention coefficient",
            data_type="scalar",
            constraints="α_t = 1 - β_t",
            color_code="#27ae60"
        ),
        "alpha_bar_t": VariableDefinition(
            name="ᾱ_t",
            description="Cumulative signal retention",
            data_type="scalar",
            constraints="ᾱ_t = ∏_{s=1}^t α_s",
            color_code="#2ecc71"
        ),
        "epsilon": VariableDefinition(
            name="ε",
            description="Gaussian noise",
            data_type="vector",
            constraints="ε ~ N(0, I)",
            color_code="#95a5a6"
        )
    }
    
    forward_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)",
            explanation="Single-step forward diffusion adds Gaussian noise",
            mathematical_justification="Markovian noise addition process"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}",
            explanation="Reparameterization of forward step",
            mathematical_justification="Location-scale transformation of Gaussian"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)",
            explanation="Direct forward diffusion from x_0 to x_t",
            mathematical_justification="Closed-form solution using cumulative products"
        ),
        DerivationStep(
            step_number=4,
            latex_expression=r"x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon",
            explanation="Efficient sampling from q(x_t | x_0)",
            mathematical_justification="Reparameterization trick for direct sampling"
        )
    ]
    
    equations.append(Equation(
        equation_id="forward_diffusion",
        latex_expression=r"q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)",
        variables=forward_variables,
        derivation_steps=forward_derivation,
        mathematical_properties=[
            "Markovian forward process",
            "Gaussian transitions",
            "Closed-form sampling",
            "Signal decay over time"
        ],
        applications=[
            "Data corruption simulation",
            "Noise addition scheduling",
            "Training data generation"
        ],
        complexity_level=3
    ))
    
    # Reverse Diffusion Process
    reverse_variables = {
        "p_theta": VariableDefinition(
            name="p_θ",
            description="Learned reverse process",
            data_type="distribution",
            color_code="#e74c3c"
        ),
        "mu_theta": VariableDefinition(
            name="μ_θ",
            description="Predicted mean of reverse process",
            data_type="vector",
            color_code="#c0392b"
        ),
        "sigma_t": VariableDefinition(
            name="σ_t",
            description="Variance of reverse process",
            data_type="scalar",
            color_code="#a93226"
        ),
        "epsilon_theta": VariableDefinition(
            name="ε_θ",
            description="Predicted noise by neural network",
            data_type="vector",
            color_code="#922b21"
        )
    }
    
    reverse_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)",
            explanation="Parameterize reverse process as Gaussian",
            mathematical_justification="Tractable reverse diffusion approximation"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)",
            explanation="Optimal mean parameterization using noise prediction",
            mathematical_justification="Minimizes KL divergence with true posterior"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"\sigma_t^2 = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t",
            explanation="Optimal variance for reverse process",
            mathematical_justification="Matches posterior variance q(x_{t-1}|x_t,x_0)"
        ),
        DerivationStep(
            step_number=4,
            latex_expression=r"x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z",
            explanation="Sampling from reverse process",
            mathematical_justification="Reparameterization for ancestral sampling"
        )
    ]
    
    equations.append(Equation(
        equation_id="reverse_diffusion",
        latex_expression=r"p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)",
        variables=reverse_variables,
        derivation_steps=reverse_derivation,
        mathematical_properties=[
            "Learned denoising process",
            "Gaussian approximation",
            "Noise prediction parameterization",
            "Ancestral sampling"
        ],
        applications=[
            "Image generation",
            "Data denoising",
            "Conditional generation"
        ],
        complexity_level=4
    ))
    
    # Denoising Objective
    denoising_variables = {
        "L_simple": VariableDefinition(
            name="L_simple",
            description="Simplified denoising objective",
            data_type="scalar",
            color_code="#9b59b6"
        ),
        "L_vlb": VariableDefinition(
            name="L_vlb",
            description="Variational lower bound",
            data_type="scalar",
            color_code="#8e44ad"
        )
    }
    
    denoising_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"L_{vlb} = \mathbb{E}_q \left[ D_{KL}(q(x_T|x_0) || p(x_T)) + \sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0) || p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1) \right]",
            explanation="Variational lower bound decomposition",
            mathematical_justification="Evidence lower bound for diffusion models"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"L_t = \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2\sigma_t^2} \left\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\|^2 \right]",
            explanation="KL divergence term for timestep t",
            mathematical_justification="Gaussian KL divergence between posterior and model"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]",
            explanation="Simplified objective: predict the noise",
            mathematical_justification="Equivalent to weighted VLB with improved gradients"
        ),
        DerivationStep(
            step_number=4,
            latex_expression=r"L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \right\|^2 \right]",
            explanation="Final training objective",
            mathematical_justification="Direct noise prediction on randomly corrupted data"
        )
    ]
    
    equations.append(Equation(
        equation_id="denoising_objective",
        latex_expression=r"L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]",
        variables=denoising_variables,
        derivation_steps=denoising_derivation,
        mathematical_properties=[
            "MSE loss on noise prediction",
            "Equivalent to weighted VLB",
            "Stable training objective",
            "Scale-invariant gradients"
        ],
        applications=[
            "Diffusion model training",
            "Denoising network optimization",
            "Score matching"
        ],
        complexity_level=4
    ))
    
    # Sampling Procedures
    sampling_variables = {
        "DDPM": VariableDefinition(
            name="DDPM",
            description="Denoising Diffusion Probabilistic Models sampling",
            data_type="algorithm",
            color_code="#16a085"
        ),
        "DDIM": VariableDefinition(
            name="DDIM",
            description="Denoising Diffusion Implicit Models sampling",
            data_type="algorithm",
            color_code="#1abc9c"
        ),
        "eta": VariableDefinition(
            name="η",
            description="Stochasticity parameter in DDIM",
            data_type="scalar",
            constraints="0 ≤ η ≤ 1",
            color_code="#48c9b0"
        )
    }
    
    sampling_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z_t",
            explanation="DDPM sampling: stochastic reverse diffusion",
            mathematical_justification="Ancestral sampling from learned reverse process"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \epsilon_\theta(x_t, t) + \sigma_t z_t",
            explanation="DDIM sampling: deterministic with optional stochasticity",
            mathematical_justification="Non-Markovian reverse process with same marginals"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}",
            explanation="Predicted clean image from current state",
            mathematical_justification="Inversion of forward diffusion equation"
        ),
        DerivationStep(
            step_number=4,
            latex_expression=r"\sigma_t = \eta \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}} \sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}",
            explanation="DDIM variance schedule controlled by η",
            mathematical_justification="Interpolation between deterministic (η=0) and stochastic (η=1)"
        )
    ]
    
    equations.append(Equation(
        equation_id="sampling_procedures",
        latex_expression=r"x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \epsilon_\theta(x_t, t) + \sigma_t z_t",
        variables=sampling_variables,
        derivation_steps=sampling_derivation,
        mathematical_properties=[
            "Flexible sampling procedures",
            "Speed-quality trade-offs",
            "Deterministic and stochastic variants",
            "Fewer sampling steps possible"
        ],
        applications=[
            "Fast image generation",
            "Controllable sampling",
            "Interpolation in latent space"
        ],
        complexity_level=4
    ))
    
    return equations


def create_diffusion_explanations() -> List[Explanation]:
    """Create comprehensive explanations for diffusion model concepts."""
    
    explanations = []
    
    # Intuitive explanation
    explanations.append(Explanation(
        explanation_type="intuitive",
        content="""
        Diffusion models work like watching a drop of ink spread in water, but in reverse. 
        Imagine you have a clear image that gradually becomes more and more noisy until it's 
        pure random noise - this is the forward diffusion process. The model learns to reverse 
        this process, starting from noise and gradually removing it to recover the original image.
        
        The key insight is that if you can learn to remove just a little bit of noise at each 
        step, you can chain these small denoising steps together to go from complete noise back 
        to a clean image. It's like having a very patient artist who can see through the noise 
        and carefully restore the image one tiny step at a time.
        
        This approach is powerful because each denoising step is relatively easy to learn, 
        even though the overall transformation from noise to image is very complex.
        """,
        mathematical_level=2,
        prerequisites=["probability_theory", "neural_networks"]
    ))
    
    # Formal explanation
    explanations.append(Explanation(
        explanation_type="formal",
        content="""
        Diffusion models define a forward Markov chain that gradually corrupts data by adding 
        Gaussian noise, and learn a reverse Markov chain to denoise. The forward process 
        q(x₁:T | x₀) = ∏ᵢ₌₁ᵀ q(xₜ | xₜ₋₁) adds noise according to a variance schedule βₜ.
        
        The reverse process pθ(x₀:T₋₁ | xT) = p(xT) ∏ᵢ₌₁ᵀ pθ(xₜ₋₁ | xₜ) is learned to 
        approximate the intractable true reverse process. The training objective maximizes 
        the variational lower bound, which decomposes into KL divergence terms.
        
        The key breakthrough is parameterizing the reverse process to predict the noise εθ(xₜ, t) 
        rather than the mean directly. This leads to the simplified objective L_simple = 
        E[||ε - εθ(xₜ, t)||²], which is equivalent to denoising score matching and provides 
        stable gradients across all noise levels.
        """,
        mathematical_level=5,
        prerequisites=["variational_inference", "markov_chains", "score_matching", "stochastic_processes"]
    ))
    
    # Practical explanation
    explanations.append(Explanation(
        explanation_type="practical",
        content="""
        In practice, training a diffusion model involves:
        1. Sample a random timestep t and clean data x₀
        2. Add noise according to the schedule: xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε
        3. Train a neural network εθ(xₜ, t) to predict the added noise ε
        4. Minimize L = ||ε - εθ(xₜ, t)||² using standard backpropagation
        
        For generation:
        1. Start with pure Gaussian noise xT ~ N(0, I)
        2. For t = T, T-1, ..., 1: predict noise εθ(xₜ, t)
        3. Compute xₜ₋₁ using the reverse diffusion formula
        4. The final x₀ is the generated sample
        
        Key practical considerations include noise schedule design (linear, cosine), 
        network architecture (U-Net with attention), timestep embedding, and sampling 
        acceleration techniques (DDIM, DPM-Solver).
        """,
        mathematical_level=3,
        prerequisites=["neural_networks", "deep_learning", "optimization"]
    ))
    
    # Historical explanation
    explanations.append(Explanation(
        explanation_type="historical",
        content="""
        Diffusion models have roots in non-equilibrium thermodynamics and were first applied 
        to machine learning by Sohl-Dickstein et al. (2015). The breakthrough came with 
        DDPM (Ho et al., 2020), which simplified the training objective and achieved 
        state-of-the-art image generation quality.
        
        Key developments include:
        - DDIM (2020): Deterministic sampling with fewer steps
        - Improved DDPM (2021): Better noise schedules and architectures  
        - Classifier guidance (2021): Conditional generation with pretrained classifiers
        - Classifier-free guidance (2022): Conditional generation without external classifiers
        - Latent diffusion (2022): Diffusion in compressed latent space (Stable Diffusion)
        
        Diffusion models now achieve the best sample quality on many benchmarks and have 
        enabled applications like text-to-image generation, inpainting, and super-resolution.
        The field continues to evolve with faster sampling methods and new applications.
        """,
        mathematical_level=2,
        prerequisites=["machine_learning_history", "generative_models"]
    ))
    
    return explanations


def create_diffusion_visualizations() -> List[Visualization]:
    """Create diffusion model visualizations."""
    
    visualizations = []
    
    # Forward/Reverse Process Visualization
    process_viz_data = VisualizationData(
        visualization_type="diffusion_process",
        data={
            "forward_steps": list(range(0, 1001, 100)),  # Show every 100th step
            "reverse_steps": list(range(1000, -1, -100)),
            "noise_schedule": "linear",
            "sample_trajectory": True
        },
        color_mappings={
            "clean_data": "#3498db",
            "noisy_data": "#e67e22", 
            "pure_noise": "#95a5a6",
            "forward_arrow": "#e74c3c",
            "reverse_arrow": "#27ae60",
            "denoising_model": "#9b59b6"
        },
        interactive_elements=["timestep_slider", "noise_schedule_selector", "process_animation"]
    )
    
    visualizations.append(Visualization(
        visualization_id="diffusion_forward_reverse",
        visualization_type="process_diagram",
        title="Forward and Reverse Diffusion Processes",
        description="Interactive visualization of noise addition and denoising processes",
        data=process_viz_data,
        interactive=True,
        prerequisites=["stochastic_processes", "markov_chains"]
    ))
    
    # Noise Schedule Analysis
    schedule_viz_data = VisualizationData(
        visualization_type="noise_schedule_analysis",
        data={
            "schedule_types": ["linear", "cosine", "sigmoid"],
            "metrics": ["beta_t", "alpha_bar_t", "snr", "noise_level"],
            "timesteps": 1000
        },
        color_mappings={
            "linear_schedule": "#3498db",
            "cosine_schedule": "#e74c3c",
            "sigmoid_schedule": "#f39c12",
            "snr_curve": "#27ae60",
            "noise_level": "#9b59b6"
        },
        interactive_elements=["schedule_comparison", "parameter_adjustment", "metric_selection"]
    )
    
    visualizations.append(Visualization(
        visualization_id="diffusion_noise_schedule",
        visualization_type="line_plot",
        title="Noise Schedule Analysis and Comparison",
        description="Analysis of different noise schedules and their impact on training",
        data=schedule_viz_data,
        interactive=True,
        prerequisites=["optimization", "hyperparameter_tuning"]
    ))
    
    # Sampling Procedures Comparison
    sampling_viz_data = VisualizationData(
        visualization_type="sampling_comparison",
        data={
            "methods": ["DDPM", "DDIM", "DPM-Solver"],
            "num_steps": [1000, 50, 20],
            "quality_metrics": ["FID", "IS", "LPIPS"],
            "speed_metrics": ["inference_time", "nfe"]
        },
        color_mappings={
            "ddpm": "#3498db",
            "ddim": "#e74c3c",
            "dpm_solver": "#f39c12",
            "quality": "#27ae60",
            "speed": "#9b59b6"
        },
        interactive_elements=["method_selector", "step_adjustment", "quality_speed_tradeoff"]
    )
    
    visualizations.append(Visualization(
        visualization_id="diffusion_sampling_methods",
        visualization_type="comparison_plot",
        title="Sampling Methods: Quality vs Speed Trade-offs",
        description="Comparison of different sampling procedures and their characteristics",
        data=sampling_viz_data,
        interactive=True,
        prerequisites=["numerical_methods", "optimization"]
    ))
    
    # Denoising Objective Visualization
    objective_viz_data = VisualizationData(
        visualization_type="denoising_objective",
        data={
            "loss_components": ["mse_loss", "vlb_loss", "reconstruction_loss"],
            "timestep_weighting": True,
            "noise_prediction": True
        },
        color_mappings={
            "true_noise": "#95a5a6",
            "predicted_noise": "#3498db",
            "mse_loss": "#e74c3c",
            "vlb_loss": "#f39c12",
            "timestep_weight": "#9b59b6"
        },
        interactive_elements=["loss_decomposition", "timestep_analysis", "prediction_visualization"]
    )
    
    visualizations.append(Visualization(
        visualization_id="diffusion_denoising_objective",
        visualization_type="loss_analysis",
        title="Denoising Objective and Loss Components",
        description="Breakdown of the diffusion training objective and loss computation",
        data=objective_viz_data,
        interactive=True,
        prerequisites=["loss_functions", "optimization"]
    ))
    
    return visualizations


def create_diffusion_tutorial() -> MathematicalConcept:
    """Create complete diffusion models tutorial concept."""
    
    return MathematicalConcept(
        concept_id="diffusion_tutorial",
        title="Diffusion Models",
        prerequisites=[
            "probability_theory",
            "stochastic_processes",
            "markov_chains",
            "variational_inference",
            "neural_networks",
            "score_matching"
        ],
        equations=create_diffusion_equations(),
        explanations=create_diffusion_explanations(),
        examples=[create_diffusion_numerical_example()],
        visualizations=create_diffusion_visualizations(),
        difficulty_level=4,
        learning_objectives=[
            "Understand forward and reverse diffusion processes",
            "Master the denoising objective derivation and training procedure",
            "Analyze noise scheduling and its impact on generation quality",
            "Implement DDPM and DDIM sampling procedures",
            "Compare diffusion models with other generative approaches"
        ],
        assessment_criteria=[
            "Derive the forward diffusion process and its closed-form solution",
            "Prove the equivalence between VLB and simplified denoising objective",
            "Implement the complete training loop with proper noise scheduling",
            "Generate samples using both DDPM and DDIM procedures",
            "Analyze and optimize noise schedules for different data types"
        ]
    )