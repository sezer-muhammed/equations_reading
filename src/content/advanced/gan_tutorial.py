"""
Generative Adversarial Network (GAN) tutorial content with mathematical explanations.
Covers minimax game theory, adversarial training dynamics, and Nash equilibrium analysis.
"""

from typing import List
from ...core.models import (
    MathematicalConcept, Equation, VariableDefinition, DerivationStep,
    Explanation, Visualization, VisualizationData
)
from ...computation.generative_ops.gan_operations import create_gan_numerical_example


def create_gan_equations() -> List[Equation]:
    """Create GAN mathematical equations with detailed derivations."""
    
    equations = []
    
    # Minimax Game Equation
    minimax_variables = {
        "G": VariableDefinition(
            name="G",
            description="Generator network",
            data_type="function",
            color_code="#27ae60"
        ),
        "D": VariableDefinition(
            name="D",
            description="Discriminator network",
            data_type="function",
            color_code="#e74c3c"
        ),
        "x": VariableDefinition(
            name="x",
            description="Real data sample",
            data_type="vector",
            color_code="#3498db"
        ),
        "z": VariableDefinition(
            name="z",
            description="Noise vector",
            data_type="vector",
            constraints="z ~ p_z(z)",
            color_code="#f39c12"
        ),
        "p_data": VariableDefinition(
            name="p_data",
            description="Real data distribution",
            data_type="distribution",
            color_code="#3498db"
        ),
        "p_z": VariableDefinition(
            name="p_z",
            description="Noise prior distribution",
            data_type="distribution",
            color_code="#f39c12"
        )
    }
    
    minimax_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]",
            explanation="Define the value function for the two-player minimax game",
            mathematical_justification="Expected log-likelihood formulation for binary classification"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"\min_G \max_D V(D,G)",
            explanation="Generator minimizes while discriminator maximizes the value function",
            mathematical_justification="Adversarial optimization: opposing objectives create game dynamics"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}",
            explanation="Optimal discriminator given fixed generator",
            mathematical_justification="Maximizing V(D,G) w.r.t. D yields Bayes optimal classifier"
        ),
        DerivationStep(
            step_number=4,
            latex_expression=r"C(G) = \max_D V(D,G) = -\log(4) + 2 \cdot JS(p_{data} || p_g)",
            explanation="Generator objective becomes Jensen-Shannon divergence minimization",
            mathematical_justification="Substituting optimal discriminator into value function"
        )
    ]
    
    equations.append(Equation(
        equation_id="gan_minimax",
        latex_expression=r"\min_G \max_D V(D,G) = \min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]",
        variables=minimax_variables,
        derivation_steps=minimax_derivation,
        mathematical_properties=[
            "Two-player zero-sum game",
            "Non-convex optimization problem",
            "Nash equilibrium at p_g = p_data",
            "Jensen-Shannon divergence minimization"
        ],
        applications=[
            "Image generation",
            "Data augmentation",
            "Domain transfer",
            "Adversarial training"
        ],
        complexity_level=4
    ))
    
    # Nash Equilibrium Analysis
    nash_variables = {
        "p_g": VariableDefinition(
            name="p_g",
            description="Generator distribution",
            data_type="distribution",
            color_code="#27ae60"
        ),
        "JS": VariableDefinition(
            name="JS",
            description="Jensen-Shannon divergence",
            data_type="function",
            color_code="#9b59b6"
        )
    }
    
    nash_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"p_g^* = p_{data}",
            explanation="At Nash equilibrium, generator distribution equals data distribution",
            mathematical_justification="Global minimum of Jensen-Shannon divergence"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"D^*(x) = \frac{1}{2} \quad \forall x",
            explanation="Optimal discriminator cannot distinguish real from fake",
            mathematical_justification="When p_g = p_data, discriminator outputs uniform probability"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"V(D^*, G^*) = -\log(4)",
            explanation="Value function at Nash equilibrium",
            mathematical_justification="JS(p_data || p_data) = 0, so C(G*) = -log(4)"
        )
    ]
    
    equations.append(Equation(
        equation_id="gan_nash_equilibrium",
        latex_expression=r"p_g^* = p_{data}, \quad D^*(x) = \frac{1}{2}",
        variables=nash_variables,
        derivation_steps=nash_derivation,
        mathematical_properties=[
            "Unique global Nash equilibrium",
            "Generator perfectly mimics data distribution",
            "Discriminator achieves random guessing",
            "Theoretical convergence guarantee"
        ],
        applications=[
            "Training convergence analysis",
            "Model evaluation metrics",
            "Theoretical understanding"
        ],
        complexity_level=5
    ))
    
    # Training Dynamics
    training_variables = {
        "L_D": VariableDefinition(
            name="L_D",
            description="Discriminator loss",
            data_type="scalar",
            color_code="#e74c3c"
        ),
        "L_G": VariableDefinition(
            name="L_G",
            description="Generator loss",
            data_type="scalar",
            color_code="#27ae60"
        )
    }
    
    training_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"L_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]",
            explanation="Discriminator loss (negative of value function)",
            mathematical_justification="Minimize negative log-likelihood for binary classification"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"L_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]",
            explanation="Generator loss (alternative formulation for better gradients)",
            mathematical_justification="Equivalent to minimizing log(1 - D(G(z))) but with stronger gradients"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"\nabla_{\theta_D} L_D, \quad \nabla_{\theta_G} L_G",
            explanation="Alternating gradient updates for adversarial training",
            mathematical_justification="Approximate Nash equilibrium through alternating optimization"
        )
    ]
    
    equations.append(Equation(
        equation_id="gan_training_dynamics",
        latex_expression=r"L_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1-D(G(z)))], \quad L_G = -\mathbb{E}[\log D(G(z))]",
        variables=training_variables,
        derivation_steps=training_derivation,
        mathematical_properties=[
            "Alternating optimization",
            "Non-convex loss landscape",
            "Potential for oscillatory behavior",
            "Gradient-based adversarial training"
        ],
        applications=[
            "Practical GAN training",
            "Loss function design",
            "Training stability analysis"
        ],
        complexity_level=4
    ))
    
    return equations


def create_gan_explanations() -> List[Explanation]:
    """Create comprehensive explanations for GAN concepts."""
    
    explanations = []
    
    # Intuitive explanation
    explanations.append(Explanation(
        explanation_type="intuitive",
        content="""
        A Generative Adversarial Network (GAN) is like a counterfeiter (generator) trying to 
        create fake money that can fool a detective (discriminator). The counterfeiter gets 
        better at making realistic fakes, while the detective gets better at spotting them.
        
        This adversarial process creates a competitive dynamic where both networks improve 
        simultaneously. Eventually, the counterfeiter becomes so good that the detective 
        can't tell real from fake money - at this point, the generator has learned to 
        perfectly mimic the real data distribution.
        
        The mathematical framework treats this as a two-player minimax game where the 
        generator tries to minimize what the discriminator tries to maximize.
        """,
        mathematical_level=2,
        prerequisites=["game_theory", "neural_networks"]
    ))
    
    # Formal explanation
    explanations.append(Explanation(
        explanation_type="formal",
        content="""
        GANs formalize generative modeling as a two-player zero-sum game. The generator G 
        maps noise z ~ p_z to data space, while discriminator D outputs the probability 
        that input data is real rather than generated.
        
        The minimax objective min_G max_D V(D,G) creates adversarial dynamics. The optimal 
        discriminator D*(x) = p_data(x)/(p_data(x) + p_g(x)) is the Bayes classifier. 
        Substituting this into the value function shows that the generator minimizes the 
        Jensen-Shannon divergence between p_data and p_g.
        
        The unique global Nash equilibrium occurs when p_g = p_data and D*(x) = 1/2 
        everywhere. However, practical training uses alternating gradient descent, which 
        may not converge to this equilibrium due to the non-convex optimization landscape.
        """,
        mathematical_level=5,
        prerequisites=["game_theory", "information_theory", "optimization", "probability_theory"]
    ))
    
    # Practical explanation
    explanations.append(Explanation(
        explanation_type="practical",
        content="""
        In practice, GAN training involves:
        1. Sample noise z and real data x from respective distributions
        2. Generate fake data G(z) using current generator
        3. Train discriminator to classify real vs fake (maximize D accuracy)
        4. Train generator to fool discriminator (minimize D accuracy on fakes)
        5. Repeat alternating updates
        
        Key challenges include:
        - Mode collapse: Generator produces limited diversity
        - Training instability: Oscillatory or divergent behavior
        - Vanishing gradients: Poor generator gradients when D is too good
        - Hyperparameter sensitivity: Learning rates, architectures, update ratios
        
        Solutions include architectural innovations (DCGAN), loss modifications (WGAN), 
        and training techniques (spectral normalization, progressive growing).
        """,
        mathematical_level=3,
        prerequisites=["neural_networks", "optimization", "deep_learning"]
    ))
    
    # Historical explanation
    explanations.append(Explanation(
        explanation_type="historical",
        content="""
        GANs were introduced by Ian Goodfellow et al. in 2014, revolutionizing generative 
        modeling. The key insight was framing generation as an adversarial game rather than 
        likelihood maximization.
        
        Early GANs suffered from training instability and mode collapse. Subsequent work 
        addressed these issues:
        - DCGAN (2015): Architectural guidelines for stable training
        - WGAN (2017): Wasserstein distance for better loss function
        - Progressive GAN (2017): Gradual resolution increase for high-quality images
        - StyleGAN (2018): Style-based architecture for controllable generation
        
        The adversarial training paradigm has influenced many areas beyond generation, 
        including adversarial examples, domain adaptation, and robust optimization.
        """,
        mathematical_level=2,
        prerequisites=["machine_learning_history"]
    ))
    
    return explanations


def create_gan_visualizations() -> List[Visualization]:
    """Create GAN visualizations."""
    
    visualizations = []
    
    # Architecture visualization
    arch_viz_data = VisualizationData(
        visualization_type="adversarial_architecture",
        data={
            "generator": {
                "layers": [
                    {"name": "Noise Input", "size": 100, "type": "input"},
                    {"name": "G Hidden", "size": 256, "type": "hidden"},
                    {"name": "Generated Data", "size": 784, "type": "output"}
                ]
            },
            "discriminator": {
                "layers": [
                    {"name": "Data Input", "size": 784, "type": "input"},
                    {"name": "D Hidden", "size": 256, "type": "hidden"},
                    {"name": "Real/Fake", "size": 1, "type": "output"}
                ]
            },
            "adversarial_flow": {
                "real_path": ["real_data", "discriminator", "real_label"],
                "fake_path": ["noise", "generator", "fake_data", "discriminator", "fake_label"],
                "loss_flow": ["discriminator_loss", "generator_loss"]
            }
        },
        color_mappings={
            "generator": "#27ae60",
            "discriminator": "#e74c3c",
            "real_data": "#3498db",
            "fake_data": "#f39c12",
            "adversarial": "#9b59b6"
        },
        interactive_elements=["training_step", "loss_tracking", "sample_generation"]
    )
    
    visualizations.append(Visualization(
        visualization_id="gan_architecture",
        visualization_type="adversarial_network",
        title="GAN Architecture with Adversarial Training Flow",
        description="Complete GAN architecture showing generator-discriminator adversarial dynamics",
        data=arch_viz_data,
        interactive=True,
        prerequisites=["neural_networks", "game_theory"]
    ))
    
    # Training dynamics visualization
    dynamics_viz_data = VisualizationData(
        visualization_type="training_dynamics",
        data={
            "loss_curves": ["discriminator_loss", "generator_loss", "value_function"],
            "discriminator_outputs": ["D_real", "D_fake"],
            "convergence_metrics": ["js_divergence", "mode_coverage"],
            "instability_indicators": ["gradient_norms", "loss_oscillations"]
        },
        color_mappings={
            "discriminator_loss": "#e74c3c",
            "generator_loss": "#27ae60",
            "value_function": "#9b59b6",
            "convergence": "#3498db",
            "instability": "#e67e22"
        },
        interactive_elements=["training_control", "hyperparameter_adjustment", "stability_analysis"]
    )
    
    visualizations.append(Visualization(
        visualization_id="gan_training_dynamics",
        visualization_type="time_series",
        title="GAN Training Dynamics and Convergence Analysis",
        description="Training curves showing adversarial dynamics, convergence, and instability patterns",
        data=dynamics_viz_data,
        interactive=True,
        prerequisites=["optimization", "game_theory"]
    ))
    
    # Mode collapse visualization
    mode_collapse_viz_data = VisualizationData(
        visualization_type="mode_collapse_analysis",
        data={
            "real_distribution": {"type": "multimodal", "modes": 8},
            "generated_samples": {"diversity_metric": "pairwise_distance"},
            "mode_coverage": {"covered_modes": "variable", "collapse_severity": "adjustable"},
            "diversity_metrics": ["inception_score", "fid_score", "mode_score"]
        },
        color_mappings={
            "real_modes": "#3498db",
            "generated_samples": "#f39c12",
            "collapsed_mode": "#e74c3c",
            "diversity_high": "#27ae60",
            "diversity_low": "#e67e22"
        },
        interactive_elements=["collapse_simulation", "diversity_measurement", "recovery_strategies"]
    )
    
    visualizations.append(Visualization(
        visualization_id="gan_mode_collapse",
        visualization_type="distribution_comparison",
        title="Mode Collapse Analysis and Diversity Metrics",
        description="Visualization of mode collapse phenomenon and generator diversity analysis",
        data=mode_collapse_viz_data,
        interactive=True,
        prerequisites=["probability_distributions", "statistical_analysis"]
    ))
    
    return visualizations


def create_gan_tutorial() -> MathematicalConcept:
    """Create complete GAN tutorial concept."""
    
    return MathematicalConcept(
        concept_id="gan_tutorial",
        title="Generative Adversarial Networks (GAN)",
        prerequisites=[
            "game_theory",
            "neural_networks",
            "optimization",
            "probability_theory",
            "information_theory"
        ],
        equations=create_gan_equations(),
        explanations=create_gan_explanations(),
        examples=[create_gan_numerical_example()],
        visualizations=create_gan_visualizations(),
        difficulty_level=4,
        learning_objectives=[
            "Understand minimax game formulation and adversarial dynamics",
            "Analyze Nash equilibrium conditions and convergence properties",
            "Master generator and discriminator loss functions",
            "Identify and analyze mode collapse and training instability",
            "Compare GAN variants and architectural improvements"
        ],
        assessment_criteria=[
            "Derive minimax objective from first principles",
            "Prove optimal discriminator and Nash equilibrium conditions",
            "Implement adversarial training loop with proper loss functions",
            "Diagnose and address mode collapse and training instability",
            "Evaluate GAN performance using appropriate metrics"
        ]
    )