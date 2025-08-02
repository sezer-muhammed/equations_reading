"""
Comprehensive optimization algorithms tutorial with mathematical derivations.
Covers gradient descent, Adam optimizer, learning rate scheduling, and loss landscapes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import numpy as np

from ...computation.optimization.optimizers import (
    GradientDescentOptimizer, AdamOptimizer, OptimizationResult
)
from ...core.models import (
    MathematicalConcept, Equation, NumericalExample, Explanation, 
    VariableDefinition, DerivationStep, ComputationStep, VisualizationData
)


@dataclass
class OptimizationTutorialResult:
    """Complete optimization tutorial with all components."""
    concept: MathematicalConcept
    gradient_descent_examples: List[OptimizationResult]
    adam_examples: List[OptimizationResult]
    learning_rate_examples: List[Dict[str, any]]
    loss_landscape_examples: List[Dict[str, any]]
    tutorial_sections: List[Dict[str, any]]


class OptimizationTutorial:
    """Complete tutorial for optimization algorithms in deep learning."""
    
    def __init__(self):
        self.gd_optimizer = GradientDescentOptimizer()
        self.adam_optimizer = AdamOptimizer()
    
    def create_complete_tutorial(self) -> OptimizationTutorialResult:
        """Create the complete optimization algorithms tutorial."""
        
        # Create mathematical concept definition
        concept = self._create_optimization_concept()
        
        # Generate gradient descent examples
        gd_examples = self._generate_gradient_descent_examples()
        
        # Generate Adam optimizer examples
        adam_examples = self._generate_adam_examples()
        
        # Generate learning rate scheduling examples
        lr_examples = self._generate_learning_rate_examples()
        
        # Generate loss landscape examples
        landscape_examples = self._generate_loss_landscape_examples()
        
        # Create tutorial sections
        tutorial_sections = self._create_tutorial_sections(
            gd_examples, adam_examples, lr_examples, landscape_examples
        )
        
        return OptimizationTutorialResult(
            concept=concept,
            gradient_descent_examples=gd_examples,
            adam_examples=adam_examples,
            learning_rate_examples=lr_examples,
            loss_landscape_examples=landscape_examples,
            tutorial_sections=tutorial_sections
        )
    
    def _create_optimization_concept(self) -> MathematicalConcept:
        """Create the mathematical concept definition for optimization."""
        
        # Define core equations
        equations = [
            Equation(
                equation_id="gradient_descent",
                latex_expression=r"\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)",
                variables={
                    'θ': VariableDefinition("θ", "Model parameters", "vector", None),
                    'α': VariableDefinition("α", "Learning rate", "scalar", None),
                    '∇J': VariableDefinition("∇J", "Gradient of loss function", "vector", None),
                    't': VariableDefinition("t", "Time step", "scalar", None)
                },
                derivation_steps=[
                    DerivationStep(
                        step_number=1,
                        latex_expression=r"J(\theta + \epsilon) \approx J(\theta) + \epsilon^T \nabla J(\theta)",
                        explanation="First-order Taylor approximation of loss function",
                        mathematical_justification="Linear approximation for small ε"
                    ),
                    DerivationStep(
                        step_number=2,
                        latex_expression=r"\theta_{new} = \theta - \alpha \nabla J(\theta)",
                        explanation="Move in direction of negative gradient to minimize loss",
                        mathematical_justification="Steepest descent direction"
                    )
                ],
                mathematical_properties=[
                    "Converges to local minimum for convex functions",
                    "Learning rate controls convergence speed vs stability",
                    "Requires gradient computation at each step"
                ],
                applications=[
                    "Neural network training",
                    "Linear regression optimization",
                    "Logistic regression parameter estimation"
                ]
            ),
            Equation(
                equation_id="adam_optimizer",
                latex_expression=r"\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t",
                variables={
                    'm_t': VariableDefinition("m_t", "First moment estimate (momentum)", "vector", None),
                    'v_t': VariableDefinition("v_t", "Second moment estimate (velocity)", "vector", None),
                    'β₁': VariableDefinition("β₁", "First moment decay rate", "scalar", None),
                    'β₂': VariableDefinition("β₂", "Second moment decay rate", "scalar", None),
                    'ε': VariableDefinition("ε", "Small constant for numerical stability", "scalar", None)
                },
                derivation_steps=[
                    DerivationStep(
                        step_number=1,
                        latex_expression=r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t",
                        explanation="Update biased first moment estimate",
                        mathematical_justification="Exponential moving average of gradients"
                    ),
                    DerivationStep(
                        step_number=2,
                        latex_expression=r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2",
                        explanation="Update biased second moment estimate",
                        mathematical_justification="Exponential moving average of squared gradients"
                    ),
                    DerivationStep(
                        step_number=3,
                        latex_expression=r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t}",
                        explanation="Bias correction for first moment",
                        mathematical_justification="Corrects initialization bias"
                    ),
                    DerivationStep(
                        step_number=4,
                        latex_expression=r"\hat{v}_t = \frac{v_t}{1 - \beta_2^t}",
                        explanation="Bias correction for second moment",
                        mathematical_justification="Corrects initialization bias"
                    )
                ],
                mathematical_properties=[
                    "Adaptive learning rates per parameter",
                    "Combines momentum and RMSprop benefits",
                    "Bias correction prevents initial step bias"
                ],
                applications=[
                    "Deep neural network training",
                    "Transformer model optimization",
                    "Computer vision model training"
                ]
            )
        ]
        
        # Create explanations
        explanations = [
            Explanation(
                explanation_type="intuitive",
                content="""
                Optimization algorithms find the best parameters for machine learning models by iteratively
                adjusting them to minimize a loss function. Think of it like finding the lowest point in
                a mountainous landscape - you want to roll a ball downhill to reach the valley.
                
                Different algorithms use different strategies:
                - Gradient descent: Always go in the steepest downhill direction
                - Adam: Use momentum to avoid getting stuck and adapt step size per parameter
                """,
                mathematical_level=2
            ),
            Explanation(
                explanation_type="formal",
                content="""
                The fundamental optimization problem in machine learning is:
                
                θ* = argmin_θ J(θ)
                
                Where J(θ) is the loss function and θ are the model parameters. Since this is generally
                non-convex and high-dimensional, we use iterative first-order methods that rely on
                gradient information to make local improvements.
                """,
                mathematical_level=4
            )
        ]
        
        return MathematicalConcept(
            concept_id="optimization_algorithms",
            title="Optimization Algorithms for Deep Learning",
            prerequisites=["calculus", "linear_algebra", "probability_theory"],
            equations=equations,
            explanations=explanations,
            examples=[],  # Will be filled by numerical examples
            visualizations=[],  # Will be filled by computation results
            difficulty_level=3  # Intermediate to advanced level
        )   
    
    def _generate_gradient_descent_examples(self) -> List[OptimizationResult]:
        """Generate comprehensive gradient descent examples."""
        examples = []
        
        # Example 1: Simple quadratic function optimization
        def quadratic_loss(params):
            """Simple quadratic: f(x,y) = x² + 2y² + xy"""
            x, y = params[0], params[1]
            return x**2 + 2*y**2 + x*y
        
        def quadratic_gradient(params):
            """Gradient of quadratic function"""
            x, y = params[0], params[1]
            return np.array([2*x + y, 4*y + x])
        
        initial_params = np.array([3.0, 2.0])
        gd_result = self.gd_optimizer.optimize(
            initial_params, quadratic_loss, quadratic_gradient, num_steps=50
        )
        examples.append(gd_result)
        
        # Example 2: Rosenbrock function (challenging non-convex)
        def rosenbrock_loss(params):
            """Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²"""
            a, b = 1, 100
            x, y = params[0], params[1]
            return (a - x)**2 + b * (y - x**2)**2
        
        def rosenbrock_gradient(params):
            """Gradient of Rosenbrock function"""
            a, b = 1, 100
            x, y = params[0], params[1]
            dx = -2*(a - x) - 4*b*x*(y - x**2)
            dy = 2*b*(y - x**2)
            return np.array([dx, dy])
        
        initial_params_rb = np.array([-1.0, 1.0])
        self.gd_optimizer.learning_rate = 0.001  # Smaller LR for Rosenbrock
        gd_result_rb = self.gd_optimizer.optimize(
            initial_params_rb, rosenbrock_loss, rosenbrock_gradient, num_steps=200
        )
        examples.append(gd_result_rb)
        
        # Example 3: Different learning rates comparison
        learning_rates = [0.1, 0.01, 0.001]
        for lr in learning_rates:
            self.gd_optimizer.learning_rate = lr
            gd_result_lr = self.gd_optimizer.optimize(
                np.array([2.0, 1.5]), quadratic_loss, quadratic_gradient, num_steps=30
            )
            examples.append(gd_result_lr)
        
        return examples

    def _generate_adam_examples(self) -> List[OptimizationResult]:
        """Generate comprehensive Adam optimizer examples."""
        examples = []
        
        # Example 1: Same quadratic function with Adam
        def quadratic_loss(params):
            x, y = params[0], params[1]
            return x**2 + 2*y**2 + x*y
        
        def quadratic_gradient(params):
            x, y = params[0], params[1]
            return np.array([2*x + y, 4*y + x])
        
        initial_params = np.array([3.0, 2.0])
        adam_result = self.adam_optimizer.optimize(
            initial_params, quadratic_loss, quadratic_gradient, num_steps=50
        )
        examples.append(adam_result)
        
        # Example 2: Rosenbrock with Adam
        def rosenbrock_loss(params):
            a, b = 1, 100
            x, y = params[0], params[1]
            return (a - x)**2 + b * (y - x**2)**2
        
        def rosenbrock_gradient(params):
            a, b = 1, 100
            x, y = params[0], params[1]
            dx = -2*(a - x) - 4*b*x*(y - x**2)
            dy = 2*b*(y - x**2)
            return np.array([dx, dy])
        
        initial_params_rb = np.array([-1.0, 1.0])
        adam_result_rb = self.adam_optimizer.optimize(
            initial_params_rb, rosenbrock_loss, rosenbrock_gradient, num_steps=100
        )
        examples.append(adam_result_rb)
        
        # Example 3: Different Adam hyperparameters
        hyperparams = [
            {'beta1': 0.9, 'beta2': 0.999},  # Default
            {'beta1': 0.95, 'beta2': 0.999}, # Higher momentum
            {'beta1': 0.9, 'beta2': 0.99}    # Lower second moment decay
        ]
        
        for params in hyperparams:
            adam_opt = AdamOptimizer(beta1=params['beta1'], beta2=params['beta2'])
            adam_result_hp = adam_opt.optimize(
                np.array([2.0, 1.5]), quadratic_loss, quadratic_gradient, num_steps=30
            )
            examples.append(adam_result_hp)
        
        return examples

    def _generate_learning_rate_examples(self) -> List[Dict[str, any]]:
        """Generate learning rate scheduling examples."""
        examples = []
        
        # Define a simple loss function for demonstration
        def simple_loss(params):
            return np.sum(params**2)
        
        def simple_gradient(params):
            return 2 * params
        
        initial_params = np.array([2.0, 1.5])
        
        # Example 1: Constant learning rate
        constant_lr_results = []
        for step in range(50):
            lr = 0.1
            constant_lr_results.append({'step': step, 'lr': lr, 'loss': simple_loss(initial_params)})
            initial_params = initial_params - lr * simple_gradient(initial_params)
        
        examples.append({
            'schedule_type': 'constant',
            'results': constant_lr_results,
            'description': 'Constant learning rate throughout training'
        })
        
        # Example 2: Exponential decay
        initial_params = np.array([2.0, 1.5])
        exp_decay_results = []
        initial_lr = 0.1
        decay_rate = 0.95
        
        for step in range(50):
            lr = initial_lr * (decay_rate ** step)
            exp_decay_results.append({'step': step, 'lr': lr, 'loss': simple_loss(initial_params)})
            initial_params = initial_params - lr * simple_gradient(initial_params)
        
        examples.append({
            'schedule_type': 'exponential_decay',
            'results': exp_decay_results,
            'description': 'Exponentially decaying learning rate'
        })
        
        # Example 3: Step decay
        initial_params = np.array([2.0, 1.5])
        step_decay_results = []
        initial_lr = 0.1
        
        for step in range(50):
            lr = initial_lr * (0.5 ** (step // 10))  # Halve every 10 steps
            step_decay_results.append({'step': step, 'lr': lr, 'loss': simple_loss(initial_params)})
            initial_params = initial_params - lr * simple_gradient(initial_params)
        
        examples.append({
            'schedule_type': 'step_decay',
            'results': step_decay_results,
            'description': 'Step-wise learning rate decay'
        })
        
        return examples   
        
    def _generate_loss_landscape_examples(self) -> List[Dict[str, any]]:
        """Generate loss landscape exploration examples."""
        examples = []
        
        # Example 1: 2D Quadratic landscape
        x_range = np.linspace(-3, 3, 50)
        y_range = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Quadratic function: f(x,y) = x² + 2y² + xy
        Z_quadratic = X**2 + 2*Y**2 + X*Y
        
        examples.append({
            'landscape_type': 'quadratic',
            'X': X, 'Y': Y, 'Z': Z_quadratic,
            'description': 'Convex quadratic function with single global minimum',
            'properties': ['Convex', 'Single minimum', 'Elliptical contours']
        })
        
        # Example 2: Rosenbrock landscape
        Z_rosenbrock = (1 - X)**2 + 100 * (Y - X**2)**2
        
        examples.append({
            'landscape_type': 'rosenbrock',
            'X': X, 'Y': Y, 'Z': Z_rosenbrock,
            'description': 'Non-convex Rosenbrock function with narrow valley',
            'properties': ['Non-convex', 'Narrow valley', 'Challenging for optimization']
        })
        
        # Example 3: Multi-modal landscape
        Z_multimodal = (np.sin(X) * np.cos(Y) + 0.1 * (X**2 + Y**2))
        
        examples.append({
            'landscape_type': 'multimodal',
            'X': X, 'Y': Y, 'Z': Z_multimodal,
            'description': 'Multi-modal function with multiple local minima',
            'properties': ['Multiple local minima', 'Oscillatory', 'Requires global optimization']
        })
        
        return examples

    def _create_tutorial_sections(self, gd_examples: List[OptimizationResult],
                                adam_examples: List[OptimizationResult],
                                lr_examples: List[Dict[str, any]],
                                landscape_examples: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Create structured tutorial sections."""
        
        sections = [
            {
                'title': 'Introduction to Optimization in Deep Learning',
                'content': """
                Optimization is the heart of machine learning - it's how we find the best parameters
                for our models. This tutorial covers the mathematical foundations and practical
                implementations of key optimization algorithms used in deep learning.
                
                We'll explore gradient descent, the Adam optimizer, learning rate scheduling,
                and visualize how different optimizers navigate loss landscapes.
                """,
                'learning_objectives': [
                    'Understand gradient-based optimization principles',
                    'Master gradient descent and Adam optimizer mathematics',
                    'Learn learning rate scheduling strategies',
                    'Visualize optimizer behavior on different loss landscapes'
                ]
            },
            {
                'title': 'Gradient Descent: The Foundation',
                'content': """
                Gradient descent is the fundamental optimization algorithm in machine learning.
                It uses the gradient (slope) of the loss function to determine which direction
                to move the parameters to reduce the loss.
                
                The key insight: move in the direction opposite to the gradient (steepest ascent)
                to find the steepest descent direction.
                """,
                'mathematical_derivation': gd_examples[0].computation_steps,
                'numerical_examples': {
                    'quadratic': gd_examples[0],
                    'rosenbrock': gd_examples[1],
                    'learning_rate_comparison': gd_examples[2:5]
                },
                'key_insights': [
                    'Learning rate controls step size and convergence',
                    'Too large: oscillation or divergence',
                    'Too small: slow convergence',
                    'Works well on convex functions'
                ]
            },
            {
                'title': 'Adam Optimizer: Adaptive Moments',
                'content': """
                Adam (Adaptive Moment Estimation) combines the benefits of momentum and
                adaptive learning rates. It maintains running averages of both the gradients
                and their squared values, providing adaptive learning rates for each parameter.
                
                Key innovations:
                1. Momentum (first moment) for smoother updates
                2. Adaptive learning rates (second moment) per parameter
                3. Bias correction for initialization
                """,
                'mathematical_derivation': adam_examples[0].computation_steps,
                'numerical_examples': {
                    'quadratic': adam_examples[0],
                    'rosenbrock': adam_examples[1],
                    'hyperparameter_comparison': adam_examples[2:5]
                },
                'key_insights': [
                    'Combines momentum and RMSprop benefits',
                    'Adaptive per-parameter learning rates',
                    'Bias correction prevents initial step bias',
                    'Generally robust across different problems'
                ]
            },
            {
                'title': 'Learning Rate Scheduling',
                'content': """
                Learning rate scheduling adjusts the learning rate during training to improve
                convergence. Common strategies include exponential decay, step decay, and
                cosine annealing.
                
                The intuition: start with larger steps to make quick progress, then use
                smaller steps for fine-tuning near the optimum.
                """,
                'scheduling_examples': lr_examples,
                'strategies': [
                    {
                        'name': 'Constant',
                        'formula': 'α(t) = α₀',
                        'pros': ['Simple', 'Predictable'],
                        'cons': ['May not converge optimally']
                    },
                    {
                        'name': 'Exponential Decay',
                        'formula': 'α(t) = α₀ × γᵗ',
                        'pros': ['Smooth decay', 'Theoretical guarantees'],
                        'cons': ['May decay too quickly']
                    },
                    {
                        'name': 'Step Decay',
                        'formula': 'α(t) = α₀ × γ^⌊t/s⌋',
                        'pros': ['Stable periods', 'Easy to tune'],
                        'cons': ['Sudden changes can be disruptive']
                    }
                ]
            },
            {
                'title': 'Loss Landscape Exploration',
                'content': """
                Understanding loss landscapes helps us choose appropriate optimizers and
                hyperparameters. Different functions present different challenges:
                
                - Convex: Single global minimum, gradient descent works well
                - Non-convex: Multiple local minima, may need advanced techniques
                - Narrow valleys: Require careful learning rate tuning
                - Flat regions: Gradients are small, slow progress
                """,
                'landscape_examples': landscape_examples,
                'optimizer_comparisons': [
                    {
                        'landscape': 'Convex Quadratic',
                        'gradient_descent': 'Excellent - direct path to minimum',
                        'adam': 'Good - may overshoot slightly due to momentum',
                        'best_choice': 'Gradient Descent'
                    },
                    {
                        'landscape': 'Rosenbrock Valley',
                        'gradient_descent': 'Poor - oscillates in narrow valley',
                        'adam': 'Excellent - adaptive rates handle different scales',
                        'best_choice': 'Adam'
                    },
                    {
                        'landscape': 'Multi-modal',
                        'gradient_descent': 'Poor - gets stuck in local minima',
                        'adam': 'Better - momentum helps escape shallow minima',
                        'best_choice': 'Advanced methods (e.g., simulated annealing)'
                    }
                ]
            },
            {
                'title': 'Practical Guidelines and Best Practices',
                'content': """
                Choosing the right optimizer and hyperparameters depends on your specific problem.
                Here are practical guidelines based on the mathematical properties we've explored.
                """,
                'guidelines': [
                    {
                        'scenario': 'Deep Neural Networks',
                        'recommended': 'Adam with default hyperparameters',
                        'reasoning': 'Adaptive rates handle different parameter scales'
                    },
                    {
                        'scenario': 'Convex Problems',
                        'recommended': 'SGD with momentum',
                        'reasoning': 'Simpler, more predictable convergence'
                    },
                    {
                        'scenario': 'Large Batch Training',
                        'recommended': 'SGD with learning rate scheduling',
                        'reasoning': 'Better generalization, less overfitting'
                    },
                    {
                        'scenario': 'Fine-tuning Pre-trained Models',
                        'recommended': 'Adam with lower learning rate',
                        'reasoning': 'Gentle updates preserve learned features'
                    }
                ],
                'debugging_tips': [
                    'Loss not decreasing: Check learning rate (too small/large)',
                    'Loss oscillating: Reduce learning rate or add momentum',
                    'Slow convergence: Increase learning rate or use Adam',
                    'Overfitting: Add regularization or reduce learning rate'
                ]
            }
        ]
        
        return sections