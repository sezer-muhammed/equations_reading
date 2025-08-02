"""
Variational Autoencoder (VAE) tutorial content with mathematical explanations.
Covers ELBO derivation, encoder-decoder architecture, and reparameterization trick.
"""

from typing import List
from ...core.models import (
    MathematicalConcept, Equation, VariableDefinition, DerivationStep,
    Explanation, Visualization, VisualizationData
)
from ...computation.generative_ops.vae_operations import create_vae_numerical_example


def create_vae_equations() -> List[Equation]:
    """Create VAE mathematical equations with detailed derivations."""
    
    equations = []
    
    # ELBO Equation
    elbo_variables = {
        "x": VariableDefinition(
            name="x", 
            description="Observed data point",
            data_type="vector",
            shape=(784,),
            color_code="#2c3e50"
        ),
        "z": VariableDefinition(
            name="z",
            description="Latent variable",
            data_type="vector", 
            shape=(20,),
            color_code="#f39c12"
        ),
        "theta": VariableDefinition(
            name="θ",
            description="Decoder parameters",
            data_type="tensor",
            color_code="#e74c3c"
        ),
        "phi": VariableDefinition(
            name="φ",
            description="Encoder parameters", 
            data_type="tensor",
            color_code="#3498db"
        )
    }
    
    elbo_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"\log p_\theta(x) = \log \int p_\theta(x|z) p(z) dz",
            explanation="Start with marginal likelihood of observed data",
            mathematical_justification="Marginalization over latent variables"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"\log p_\theta(x) = \log \int \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} q_\phi(z|x) dz",
            explanation="Introduce variational posterior q_φ(z|x)",
            mathematical_justification="Multiply and divide by variational posterior"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))",
            explanation="Apply Jensen's inequality to obtain ELBO",
            mathematical_justification="Jensen's inequality for concave logarithm function"
        ),
        DerivationStep(
            step_number=4,
            latex_expression=r"\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))",
            explanation="Define ELBO (Evidence Lower BOund)",
            mathematical_justification="Variational lower bound on log marginal likelihood"
        )
    ]
    
    equations.append(Equation(
        equation_id="vae_elbo",
        latex_expression=r"\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))",
        variables=elbo_variables,
        derivation_steps=elbo_derivation,
        mathematical_properties=[
            "Lower bound on log marginal likelihood",
            "Decomposition into reconstruction and regularization terms",
            "Enables tractable variational inference"
        ],
        applications=[
            "Generative modeling",
            "Dimensionality reduction", 
            "Representation learning",
            "Semi-supervised learning"
        ],
        complexity_level=4
    ))
    
    # Reparameterization Trick
    reparam_variables = {
        "mu": VariableDefinition(
            name="μ",
            description="Mean of variational posterior",
            data_type="vector",
            color_code="#3498db"
        ),
        "sigma": VariableDefinition(
            name="σ",
            description="Standard deviation of variational posterior",
            data_type="vector",
            color_code="#9b59b6"
        ),
        "epsilon": VariableDefinition(
            name="ε",
            description="Standard normal noise",
            data_type="vector",
            constraints="ε ~ N(0, I)",
            color_code="#95a5a6"
        )
    }
    
    reparam_derivation = [
        DerivationStep(
            step_number=1,
            latex_expression=r"z \sim q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))",
            explanation="Variational posterior is Gaussian",
            mathematical_justification="Gaussian assumption for tractability"
        ),
        DerivationStep(
            step_number=2,
            latex_expression=r"z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)",
            explanation="Reparameterize using deterministic transformation",
            mathematical_justification="Location-scale transformation of Gaussian"
        ),
        DerivationStep(
            step_number=3,
            latex_expression=r"\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[\nabla_\phi f(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon)]",
            explanation="Enable gradient estimation through sampling",
            mathematical_justification="Change of variables allows gradient flow"
        )
    ]
    
    equations.append(Equation(
        equation_id="vae_reparameterization",
        latex_expression=r"z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)",
        variables=reparam_variables,
        derivation_steps=reparam_derivation,
        mathematical_properties=[
            "Enables backpropagation through stochastic nodes",
            "Maintains distributional properties",
            "Low-variance gradient estimator"
        ],
        applications=[
            "Variational autoencoders",
            "Normalizing flows",
            "Stochastic optimization"
        ],
        complexity_level=3
    ))
    
    return equations


def create_vae_explanations() -> List[Explanation]:
    """Create comprehensive explanations for VAE concepts."""
    
    explanations = []
    
    # Intuitive explanation
    explanations.append(Explanation(
        explanation_type="intuitive",
        content="""
        A Variational Autoencoder (VAE) is like a sophisticated compression system that learns 
        to encode data into a compact latent representation and then decode it back. Unlike 
        regular autoencoders, VAEs learn a probabilistic encoding that captures uncertainty 
        and enables generation of new samples.
        
        The key insight is treating the latent space as a probability distribution rather than 
        fixed points. This allows the model to generate new data by sampling from the learned 
        latent distribution and decoding the samples.
        """,
        mathematical_level=2,
        prerequisites=["probability_theory", "neural_networks"]
    ))
    
    # Formal explanation
    explanations.append(Explanation(
        explanation_type="formal",
        content="""
        VAEs solve the intractable inference problem in latent variable models through 
        variational inference. Given observed data x, we want to learn p(z|x), but this 
        posterior is intractable. Instead, we approximate it with a variational posterior 
        q_φ(z|x) parameterized by a neural network.
        
        The ELBO objective maximizes a lower bound on the log marginal likelihood:
        - Reconstruction term: E[log p(x|z)] encourages accurate reconstruction
        - KL regularization: KL(q(z|x)||p(z)) keeps posterior close to prior
        
        The reparameterization trick enables gradient-based optimization by expressing 
        stochastic sampling as a deterministic function of parameters and noise.
        """,
        mathematical_level=4,
        prerequisites=["variational_inference", "information_theory", "optimization"]
    ))
    
    # Practical explanation
    explanations.append(Explanation(
        explanation_type="practical",
        content="""
        In practice, VAEs consist of:
        1. Encoder network: Maps input x to latent parameters (μ, σ)
        2. Sampling layer: Uses reparameterization trick to sample z
        3. Decoder network: Maps latent z back to reconstruction x̂
        
        Training involves:
        - Forward pass through encoder to get q(z|x) parameters
        - Sample z using reparameterization trick
        - Decode z to get reconstruction x̂
        - Compute ELBO loss and backpropagate
        
        Key hyperparameters: latent dimensionality, β-weighting of KL term, 
        architecture choices for encoder/decoder networks.
        """,
        mathematical_level=3,
        prerequisites=["neural_networks", "backpropagation"]
    ))
    
    return explanations


def create_vae_visualizations() -> List[Visualization]:
    """Create VAE visualizations."""
    
    visualizations = []
    
    # Architecture visualization
    arch_viz_data = VisualizationData(
        visualization_type="network_architecture",
        data={
            "layers": [
                {"name": "Input", "size": 784, "type": "input"},
                {"name": "Encoder Hidden", "size": 400, "type": "hidden"},
                {"name": "μ (Mean)", "size": 20, "type": "latent_param"},
                {"name": "log σ² (Log Var)", "size": 20, "type": "latent_param"},
                {"name": "z (Latent)", "size": 20, "type": "latent"},
                {"name": "Decoder Hidden", "size": 400, "type": "hidden"},
                {"name": "Reconstruction", "size": 784, "type": "output"}
            ],
            "connections": [
                {"from": 0, "to": 1, "type": "encoder"},
                {"from": 1, "to": 2, "type": "encoder"},
                {"from": 1, "to": 3, "type": "encoder"},
                {"from": [2, 3], "to": 4, "type": "reparameterization"},
                {"from": 4, "to": 5, "type": "decoder"},
                {"from": 5, "to": 6, "type": "decoder"}
            ]
        },
        color_mappings={
            "encoder": "#3498db",
            "decoder": "#e74c3c",
            "latent": "#f39c12",
            "reparameterization": "#9b59b6"
        },
        interactive_elements=["layer_details", "parameter_counts"]
    )
    
    visualizations.append(Visualization(
        visualization_id="vae_architecture",
        visualization_type="network_diagram",
        title="VAE Architecture with Reparameterization Trick",
        description="Complete VAE architecture showing encoder-decoder structure and reparameterization",
        data=arch_viz_data,
        interactive=True,
        prerequisites=["neural_networks"]
    ))
    
    # Latent space visualization
    latent_viz_data = VisualizationData(
        visualization_type="latent_space",
        data={
            "latent_dim": 2,  # 2D for visualization
            "sample_points": 100,
            "interpolation_paths": 5
        },
        color_mappings={
            "prior": "#95a5a6",
            "posterior": "#3498db", 
            "interpolation": "#e74c3c"
        },
        interactive_elements=["interpolation_control", "sampling"]
    )
    
    visualizations.append(Visualization(
        visualization_id="vae_latent_space",
        visualization_type="scatter_plot",
        title="VAE Latent Space and Interpolation",
        description="2D latent space showing prior distribution, posterior samples, and interpolation paths",
        data=latent_viz_data,
        interactive=True,
        prerequisites=["probability_distributions"]
    ))
    
    return visualizations


def create_vae_tutorial() -> MathematicalConcept:
    """Create complete VAE tutorial concept."""
    
    return MathematicalConcept(
        concept_id="vae_tutorial",
        title="Variational Autoencoders (VAE)",
        prerequisites=[
            "probability_theory",
            "neural_networks", 
            "variational_inference",
            "information_theory"
        ],
        equations=create_vae_equations(),
        explanations=create_vae_explanations(),
        examples=[create_vae_numerical_example()],
        visualizations=create_vae_visualizations(),
        difficulty_level=4,
        learning_objectives=[
            "Understand ELBO derivation and its components",
            "Master the reparameterization trick and its importance",
            "Analyze encoder-decoder architecture design",
            "Implement latent space interpolation",
            "Compare VAE with other generative models"
        ],
        assessment_criteria=[
            "Derive ELBO from first principles",
            "Explain why reparameterization trick enables gradients",
            "Implement VAE training loop",
            "Generate new samples from trained model",
            "Analyze latent space structure and interpolations"
        ]
    )