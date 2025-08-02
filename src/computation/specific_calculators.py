"""
Specific equation calculators for common AI mathematical operations.
Implements softmax, cross-entropy, LSTM, transformer, VAE, and GAN calculations.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..core.models import (
    ColorCodedMatrix, OperationVisualization, HighlightPattern,
    AnimationFrame, ComputationStep, VisualizationData
)


@dataclass
class CalculationResult:
    """Result of a specific equation calculation."""
    result: np.ndarray
    intermediate_results: Dict[str, np.ndarray]
    computation_steps: List[ComputationStep]
    visualization: OperationVisualization
    properties: Dict[str, Any]


class SoftmaxCalculator:
    """Softmax function with numerical stability and detailed breakdown."""
    
    def __init__(self):
        self.color_palette = {
            'input': '#FF6B6B',
            'shifted': '#4ECDC4',
            'exponential': '#45B7D1',
            'sum': '#96CEB4',
            'output': '#FFEAA7'
        }
    
    def calculate(self, logits: np.ndarray, axis: int = -1, 
                 temperature: float = 1.0) -> CalculationResult:
        """
        Calculate softmax with numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        
        Args:
            logits: Input logits
            axis: Axis along which to apply softmax
            temperature: Temperature scaling parameter
            
        Returns:
            CalculationResult with detailed breakdown
        """
        computation_steps = []
        intermediate_results = {}
        
        # Step 1: Apply temperature scaling
        if temperature != 1.0:
            scaled_logits = logits / temperature
            step_1 = ComputationStep(
                step_number=1,
                operation_name="temperature_scaling",
                input_values={'logits': logits, 'temperature': np.array([temperature])},
                operation_description=f"Scale by temperature: x/T where T={temperature}",
                output_values={'scaled_logits': scaled_logits},
                visualization_hints={'temperature': temperature}
            )
            computation_steps.append(step_1)
            intermediate_results['scaled_logits'] = scaled_logits
        else:
            scaled_logits = logits
        
        # Step 2: Subtract maximum for numerical stability
        max_vals = np.max(scaled_logits, axis=axis, keepdims=True)
        shifted_logits = scaled_logits - max_vals
        
        step_2 = ComputationStep(
            step_number=len(computation_steps) + 1,
            operation_name="numerical_stability_shift",
            input_values={'scaled_logits': scaled_logits, 'max_vals': max_vals},
            operation_description="Subtract maximum for numerical stability: x - max(x)",
            output_values={'shifted_logits': shifted_logits},
            visualization_hints={'stability_shift': True}
        )
        computation_steps.append(step_2)
        intermediate_results['shifted_logits'] = shifted_logits
        intermediate_results['max_vals'] = max_vals
        
        # Step 3: Compute exponentials
        exp_logits = np.exp(shifted_logits)
        
        step_3 = ComputationStep(
            step_number=len(computation_steps) + 1,
            operation_name="exponential",
            input_values={'shifted_logits': shifted_logits},
            operation_description="Compute exponentials: exp(x - max(x))",
            output_values={'exp_logits': exp_logits},
            visualization_hints={'operation': 'exponential'}
        )
        computation_steps.append(step_3)
        intermediate_results['exp_logits'] = exp_logits
        
        # Step 4: Compute sum of exponentials
        exp_sum = np.sum(exp_logits, axis=axis, keepdims=True)
        
        step_4 = ComputationStep(
            step_number=len(computation_steps) + 1,
            operation_name="sum_exponentials",
            input_values={'exp_logits': exp_logits},
            operation_description=f"Sum exponentials along axis {axis}",
            output_values={'exp_sum': exp_sum},
            visualization_hints={'sum_axis': axis}
        )
        computation_steps.append(step_4)
        intermediate_results['exp_sum'] = exp_sum
        
        # Step 5: Divide to get probabilities
        probabilities = exp_logits / exp_sum
        
        step_5 = ComputationStep(
            step_number=len(computation_steps) + 1,
            operation_name="normalize",
            input_values={'exp_logits': exp_logits, 'exp_sum': exp_sum},
            operation_description="Normalize: exp(x) / sum(exp(x))",
            output_values={'probabilities': probabilities},
            visualization_hints={'normalization': True}
        )
        computation_steps.append(step_5)
        
        # Create visualization
        visualization = self._create_softmax_visualization(
            logits, shifted_logits, exp_logits, probabilities
        )
        
        # Calculate properties
        properties = {
            'operation': 'softmax',
            'input_shape': logits.shape,
            'output_shape': probabilities.shape,
            'temperature': temperature,
            'axis': axis,
            'max_probability': np.max(probabilities),
            'min_probability': np.min(probabilities),
            'entropy': -np.sum(probabilities * np.log(probabilities + 1e-10)),
            'numerical_stability_shift': np.max(max_vals)
        }
        
        return CalculationResult(
            result=probabilities,
            intermediate_results=intermediate_results,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _create_softmax_visualization(self, logits: np.ndarray, shifted: np.ndarray,
                                    exp_vals: np.ndarray, output: np.ndarray) -> OperationVisualization:
        """Create visualization for softmax calculation."""
        
        input_colored = ColorCodedMatrix(
            matrix_data=logits,
            color_mapping={'default': self.color_palette['input']},
            element_labels={},
            highlight_patterns=[]
        )
        
        shifted_colored = ColorCodedMatrix(
            matrix_data=shifted,
            color_mapping={'default': self.color_palette['shifted']},
            element_labels={},
            highlight_patterns=[]
        )
        
        exp_colored = ColorCodedMatrix(
            matrix_data=exp_vals,
            color_mapping={'default': self.color_palette['exponential']},
            element_labels={},
            highlight_patterns=[]
        )
        
        output_colored = ColorCodedMatrix(
            matrix_data=output,
            color_mapping={'default': self.color_palette['output']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="softmax",
            input_matrices=[input_colored],
            intermediate_steps=[shifted_colored, exp_colored],
            output_matrix=output_colored,
            animation_sequence=[]
        )


class CrossEntropyCalculator:
    """Cross-entropy loss calculation with detailed breakdown."""
    
    def __init__(self):
        self.color_palette = {
            'predictions': '#FF6B6B',
            'targets': '#4ECDC4',
            'log_probs': '#45B7D1',
            'losses': '#96CEB4',
            'total_loss': '#FFEAA7'
        }
    
    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                 reduction: str = 'mean', epsilon: float = 1e-10) -> CalculationResult:
        """
        Calculate cross-entropy loss: -sum(targets * log(predictions + epsilon))
        
        Args:
            predictions: Predicted probabilities (must sum to 1)
            targets: Target probabilities or one-hot encoded labels
            reduction: 'mean', 'sum', or 'none'
            epsilon: Small value to prevent log(0)
            
        Returns:
            CalculationResult with detailed breakdown
        """
        computation_steps = []
        intermediate_results = {}
        
        # Step 1: Add epsilon for numerical stability
        stable_predictions = predictions + epsilon
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="numerical_stability",
            input_values={'predictions': predictions, 'epsilon': np.array([epsilon])},
            operation_description=f"Add epsilon for stability: pred + {epsilon}",
            output_values={'stable_predictions': stable_predictions},
            visualization_hints={'epsilon': epsilon}
        )
        computation_steps.append(step_1)
        intermediate_results['stable_predictions'] = stable_predictions
        
        # Step 2: Compute log probabilities
        log_probs = np.log(stable_predictions)
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="log_probabilities",
            input_values={'stable_predictions': stable_predictions},
            operation_description="Compute log probabilities: log(pred + ε)",
            output_values={'log_probs': log_probs},
            visualization_hints={'operation': 'logarithm'}
        )
        computation_steps.append(step_2)
        intermediate_results['log_probs'] = log_probs
        
        # Step 3: Multiply by targets
        weighted_log_probs = targets * log_probs
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="weight_by_targets",
            input_values={'targets': targets, 'log_probs': log_probs},
            operation_description="Weight by targets: targets * log(pred)",
            output_values={'weighted_log_probs': weighted_log_probs},
            visualization_hints={'operation': 'elementwise_multiply'}
        )
        computation_steps.append(step_3)
        intermediate_results['weighted_log_probs'] = weighted_log_probs
        
        # Step 4: Sum over classes (negative sum)
        class_losses = -np.sum(weighted_log_probs, axis=-1)
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="sum_over_classes",
            input_values={'weighted_log_probs': weighted_log_probs},
            operation_description="Sum over classes: -sum(targets * log(pred))",
            output_values={'class_losses': class_losses},
            visualization_hints={'sum_axis': -1, 'negative': True}
        )
        computation_steps.append(step_4)
        intermediate_results['class_losses'] = class_losses
        
        # Step 5: Apply reduction
        if reduction == 'mean':
            final_loss = np.mean(class_losses)
            reduction_desc = "Average over batch"
        elif reduction == 'sum':
            final_loss = np.sum(class_losses)
            reduction_desc = "Sum over batch"
        else:  # 'none'
            final_loss = class_losses
            reduction_desc = "No reduction applied"
        
        step_5 = ComputationStep(
            step_number=5,
            operation_name="apply_reduction",
            input_values={'class_losses': class_losses},
            operation_description=reduction_desc,
            output_values={'final_loss': final_loss},
            visualization_hints={'reduction': reduction}
        )
        computation_steps.append(step_5)
        
        # Create visualization
        visualization = self._create_crossentropy_visualization(
            predictions, targets, log_probs, class_losses
        )
        
        # Calculate properties
        properties = {
            'operation': 'cross_entropy',
            'reduction': reduction,
            'epsilon': epsilon,
            'batch_size': predictions.shape[0] if len(predictions.shape) > 1 else 1,
            'num_classes': predictions.shape[-1],
            'mean_loss': np.mean(class_losses) if reduction != 'mean' else final_loss,
            'max_loss': np.max(class_losses),
            'min_loss': np.min(class_losses)
        }
        
        return CalculationResult(
            result=final_loss,
            intermediate_results=intermediate_results,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _create_crossentropy_visualization(self, predictions: np.ndarray, targets: np.ndarray,
                                         log_probs: np.ndarray, losses: np.ndarray) -> OperationVisualization:
        """Create visualization for cross-entropy calculation."""
        
        pred_colored = ColorCodedMatrix(
            matrix_data=predictions,
            color_mapping={'default': self.color_palette['predictions']},
            element_labels={},
            highlight_patterns=[]
        )
        
        target_colored = ColorCodedMatrix(
            matrix_data=targets,
            color_mapping={'default': self.color_palette['targets']},
            element_labels={},
            highlight_patterns=[]
        )
        
        log_colored = ColorCodedMatrix(
            matrix_data=log_probs,
            color_mapping={'default': self.color_palette['log_probs']},
            element_labels={},
            highlight_patterns=[]
        )
        
        loss_colored = ColorCodedMatrix(
            matrix_data=losses.reshape(-1, 1) if losses.ndim == 1 else losses,
            color_mapping={'default': self.color_palette['losses']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="cross_entropy",
            input_matrices=[pred_colored, target_colored],
            intermediate_steps=[log_colored],
            output_matrix=loss_colored,
            animation_sequence=[]
        )


class LSTMCalculator:
    """LSTM cell state and gate computations with detailed breakdown."""
    
    def __init__(self):
        self.color_palette = {
            'input': '#FF6B6B',
            'hidden': '#4ECDC4',
            'cell': '#45B7D1',
            'forget_gate': '#96CEB4',
            'input_gate': '#FFEAA7',
            'output_gate': '#DDA0DD',
            'candidate': '#98FB98'
        }
    
    def calculate(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                 W_f: np.ndarray, W_i: np.ndarray, W_o: np.ndarray, W_c: np.ndarray,
                 U_f: np.ndarray, U_i: np.ndarray, U_o: np.ndarray, U_c: np.ndarray,
                 b_f: np.ndarray, b_i: np.ndarray, b_o: np.ndarray, b_c: np.ndarray) -> CalculationResult:
        """
        Calculate LSTM forward pass with all gates and cell state updates.
        
        LSTM equations:
        f_t = σ(W_f * x_t + U_f * h_{t-1} + b_f)  # Forget gate
        i_t = σ(W_i * x_t + U_i * h_{t-1} + b_i)  # Input gate
        C̃_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)  # Candidate values
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t  # Cell state
        o_t = σ(W_o * x_t + U_o * h_{t-1} + b_o)  # Output gate
        h_t = o_t ⊙ tanh(C_t)  # Hidden state
        """
        computation_steps = []
        intermediate_results = {}
        
        # Helper function for sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip for numerical stability
        
        # Step 1: Compute forget gate
        f_linear = W_f @ x_t + U_f @ h_prev + b_f
        f_t = sigmoid(f_linear)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="forget_gate",
            input_values={'x_t': x_t, 'h_prev': h_prev, 'W_f': W_f, 'U_f': U_f, 'b_f': b_f},
            operation_description="Forget gate: f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)",
            output_values={'f_linear': f_linear, 'f_t': f_t},
            visualization_hints={'gate_type': 'forget', 'activation': 'sigmoid'}
        )
        computation_steps.append(step_1)
        intermediate_results['forget_gate'] = f_t
        intermediate_results['forget_linear'] = f_linear
        
        # Step 2: Compute input gate
        i_linear = W_i @ x_t + U_i @ h_prev + b_i
        i_t = sigmoid(i_linear)
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="input_gate",
            input_values={'x_t': x_t, 'h_prev': h_prev, 'W_i': W_i, 'U_i': U_i, 'b_i': b_i},
            operation_description="Input gate: i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)",
            output_values={'i_linear': i_linear, 'i_t': i_t},
            visualization_hints={'gate_type': 'input', 'activation': 'sigmoid'}
        )
        computation_steps.append(step_2)
        intermediate_results['input_gate'] = i_t
        intermediate_results['input_linear'] = i_linear
        
        # Step 3: Compute candidate values
        c_linear = W_c @ x_t + U_c @ h_prev + b_c
        c_tilde = np.tanh(c_linear)
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="candidate_values",
            input_values={'x_t': x_t, 'h_prev': h_prev, 'W_c': W_c, 'U_c': U_c, 'b_c': b_c},
            operation_description="Candidate values: C̃_t = tanh(W_c·x_t + U_c·h_{t-1} + b_c)",
            output_values={'c_linear': c_linear, 'c_tilde': c_tilde},
            visualization_hints={'gate_type': 'candidate', 'activation': 'tanh'}
        )
        computation_steps.append(step_3)
        intermediate_results['candidate_values'] = c_tilde
        intermediate_results['candidate_linear'] = c_linear
        
        # Step 4: Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="update_cell_state",
            input_values={'f_t': f_t, 'c_prev': c_prev, 'i_t': i_t, 'c_tilde': c_tilde},
            operation_description="Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t",
            output_values={'c_t': c_t},
            visualization_hints={'operation': 'cell_update'}
        )
        computation_steps.append(step_4)
        intermediate_results['cell_state'] = c_t
        
        # Step 5: Compute output gate
        o_linear = W_o @ x_t + U_o @ h_prev + b_o
        o_t = sigmoid(o_linear)
        
        step_5 = ComputationStep(
            step_number=5,
            operation_name="output_gate",
            input_values={'x_t': x_t, 'h_prev': h_prev, 'W_o': W_o, 'U_o': U_o, 'b_o': b_o},
            operation_description="Output gate: o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)",
            output_values={'o_linear': o_linear, 'o_t': o_t},
            visualization_hints={'gate_type': 'output', 'activation': 'sigmoid'}
        )
        computation_steps.append(step_5)
        intermediate_results['output_gate'] = o_t
        intermediate_results['output_linear'] = o_linear
        
        # Step 6: Compute hidden state
        h_t = o_t * np.tanh(c_t)
        
        step_6 = ComputationStep(
            step_number=6,
            operation_name="update_hidden_state",
            input_values={'o_t': o_t, 'c_t': c_t},
            operation_description="Hidden state: h_t = o_t ⊙ tanh(C_t)",
            output_values={'h_t': h_t, 'tanh_c_t': np.tanh(c_t)},
            visualization_hints={'operation': 'hidden_update'}
        )
        computation_steps.append(step_6)
        
        # Create visualization
        visualization = self._create_lstm_visualization(
            x_t, h_prev, c_prev, f_t, i_t, o_t, c_tilde, c_t, h_t
        )
        
        # Calculate properties
        properties = {
            'operation': 'lstm_cell',
            'input_size': x_t.shape[0],
            'hidden_size': h_t.shape[0],
            'forget_gate_mean': np.mean(f_t),
            'input_gate_mean': np.mean(i_t),
            'output_gate_mean': np.mean(o_t),
            'cell_state_norm': np.linalg.norm(c_t),
            'hidden_state_norm': np.linalg.norm(h_t),
            'candidate_values_range': (np.min(c_tilde), np.max(c_tilde))
        }
        
        return CalculationResult(
            result=np.concatenate([h_t, c_t]),  # Return both hidden and cell state
            intermediate_results=intermediate_results,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _create_lstm_visualization(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                                 f_t: np.ndarray, i_t: np.ndarray, o_t: np.ndarray,
                                 c_tilde: np.ndarray, c_t: np.ndarray, h_t: np.ndarray) -> OperationVisualization:
        """Create visualization for LSTM computation."""
        
        input_matrices = [
            ColorCodedMatrix(x_t.reshape(-1, 1), {'default': self.color_palette['input']}, {}, []),
            ColorCodedMatrix(h_prev.reshape(-1, 1), {'default': self.color_palette['hidden']}, {}, []),
            ColorCodedMatrix(c_prev.reshape(-1, 1), {'default': self.color_palette['cell']}, {}, [])
        ]
        
        intermediate_steps = [
            ColorCodedMatrix(f_t.reshape(-1, 1), {'default': self.color_palette['forget_gate']}, {}, []),
            ColorCodedMatrix(i_t.reshape(-1, 1), {'default': self.color_palette['input_gate']}, {}, []),
            ColorCodedMatrix(o_t.reshape(-1, 1), {'default': self.color_palette['output_gate']}, {}, []),
            ColorCodedMatrix(c_tilde.reshape(-1, 1), {'default': self.color_palette['candidate']}, {}, [])
        ]
        
        output_matrix = ColorCodedMatrix(
            np.vstack([h_t.reshape(-1, 1), c_t.reshape(-1, 1)]),
            {'default': self.color_palette['hidden']},
            {},
            []
        )
        
        return OperationVisualization(
            operation_type="lstm_cell",
            input_matrices=input_matrices,
            intermediate_steps=intermediate_steps,
            output_matrix=output_matrix,
            animation_sequence=[]
        )


class VAEELBOCalculator:
    """Variational Autoencoder ELBO calculation with KL divergence and reconstruction terms."""
    
    def __init__(self):
        self.color_palette = {
            'input': '#FF6B6B',
            'mu': '#4ECDC4',
            'logvar': '#45B7D1',
            'z': '#96CEB4',
            'reconstruction': '#FFEAA7',
            'kl_divergence': '#DDA0DD'
        }
    
    def calculate(self, x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, 
                 logvar: np.ndarray, beta: float = 1.0) -> CalculationResult:
        """
        Calculate VAE ELBO: ELBO = -Reconstruction_Loss - β * KL_Divergence
        
        KL divergence for standard normal prior:
        KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Beta parameter for β-VAE
            
        Returns:
            CalculationResult with ELBO breakdown
        """
        computation_steps = []
        intermediate_results = {}
        
        # Step 1: Calculate reconstruction loss (MSE or BCE)
        reconstruction_loss = np.mean((x - x_recon) ** 2)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="reconstruction_loss",
            input_values={'x': x, 'x_recon': x_recon},
            operation_description="Reconstruction loss: MSE(x, x_recon)",
            output_values={'reconstruction_loss': np.array([reconstruction_loss])},
            visualization_hints={'loss_type': 'mse'}
        )
        computation_steps.append(step_1)
        intermediate_results['reconstruction_loss'] = reconstruction_loss
        
        # Step 2: Calculate KL divergence components
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        
        # Component 1: mu^2
        mu_squared = mu ** 2
        step_2a = ComputationStep(
            step_number=2,
            operation_name="mu_squared",
            input_values={'mu': mu},
            operation_description="Compute μ²",
            output_values={'mu_squared': mu_squared},
            visualization_hints={'operation': 'square'}
        )
        computation_steps.append(step_2a)
        intermediate_results['mu_squared'] = mu_squared
        
        # Component 2: exp(logvar)
        exp_logvar = np.exp(logvar)
        step_2b = ComputationStep(
            step_number=3,
            operation_name="exp_logvar",
            input_values={'logvar': logvar},
            operation_description="Compute exp(log σ²) = σ²",
            output_values={'exp_logvar': exp_logvar},
            visualization_hints={'operation': 'exponential'}
        )
        computation_steps.append(step_2b)
        intermediate_results['exp_logvar'] = exp_logvar
        
        # Step 3: Combine KL divergence terms
        kl_terms = 1 + logvar - mu_squared - exp_logvar
        kl_divergence = -0.5 * np.sum(kl_terms)
        
        step_3 = ComputationStep(
            step_number=4,
            operation_name="kl_divergence",
            input_values={'logvar': logvar, 'mu_squared': mu_squared, 'exp_logvar': exp_logvar},
            operation_description="KL divergence: -0.5 * sum(1 + logvar - μ² - σ²)",
            output_values={'kl_terms': kl_terms, 'kl_divergence': np.array([kl_divergence])},
            visualization_hints={'kl_formula': True}
        )
        computation_steps.append(step_3)
        intermediate_results['kl_divergence'] = kl_divergence
        intermediate_results['kl_terms'] = kl_terms
        
        # Step 4: Calculate ELBO
        elbo = -reconstruction_loss - beta * kl_divergence
        
        step_4 = ComputationStep(
            step_number=5,
            operation_name="elbo_calculation",
            input_values={
                'reconstruction_loss': np.array([reconstruction_loss]),
                'kl_divergence': np.array([kl_divergence]),
                'beta': np.array([beta])
            },
            operation_description=f"ELBO = -Recon_Loss - β*KL_Div (β={beta})",
            output_values={'elbo': np.array([elbo])},
            visualization_hints={'beta': beta}
        )
        computation_steps.append(step_4)
        
        # Create visualization
        visualization = self._create_vae_visualization(
            x, x_recon, mu, logvar, reconstruction_loss, kl_divergence
        )
        
        # Calculate properties
        properties = {
            'operation': 'vae_elbo',
            'reconstruction_loss': reconstruction_loss,
            'kl_divergence': kl_divergence,
            'beta': beta,
            'elbo': elbo,
            'latent_dim': mu.shape[0],
            'input_dim': x.size,
            'mu_mean': np.mean(mu),
            'mu_std': np.std(mu),
            'logvar_mean': np.mean(logvar),
            'sigma_mean': np.mean(np.sqrt(exp_logvar))
        }
        
        return CalculationResult(
            result=np.array([elbo]),
            intermediate_results=intermediate_results,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _create_vae_visualization(self, x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray,
                                logvar: np.ndarray, recon_loss: float, kl_div: float) -> OperationVisualization:
        """Create visualization for VAE ELBO calculation."""
        
        input_matrices = [
            ColorCodedMatrix(x.reshape(-1, 1), {'default': self.color_palette['input']}, {}, []),
            ColorCodedMatrix(x_recon.reshape(-1, 1), {'default': self.color_palette['reconstruction']}, {}, [])
        ]
        
        intermediate_steps = [
            ColorCodedMatrix(mu.reshape(-1, 1), {'default': self.color_palette['mu']}, {}, []),
            ColorCodedMatrix(logvar.reshape(-1, 1), {'default': self.color_palette['logvar']}, {}, [])
        ]
        
        output_matrix = ColorCodedMatrix(
            np.array([[recon_loss], [kl_div]]),
            {'default': self.color_palette['kl_divergence']},
            {},
            []
        )
        
        return OperationVisualization(
            operation_type="vae_elbo",
            input_matrices=input_matrices,
            intermediate_steps=intermediate_steps,
            output_matrix=output_matrix,
            animation_sequence=[]
        )


class GANObjectiveCalculator:
    """GAN min-max objective calculation with generator and discriminator losses."""
    
    def __init__(self):
        self.color_palette = {
            'real_data': '#FF6B6B',
            'fake_data': '#4ECDC4',
            'real_scores': '#45B7D1',
            'fake_scores': '#96CEB4',
            'generator_loss': '#FFEAA7',
            'discriminator_loss': '#DDA0DD'
        }
    
    def calculate(self, real_scores: np.ndarray, fake_scores: np.ndarray,
                 loss_type: str = 'standard') -> CalculationResult:
        """
        Calculate GAN objective losses.
        
        Standard GAN:
        L_D = -E[log(D(x))] - E[log(1 - D(G(z)))]
        L_G = -E[log(D(G(z)))]  (or E[log(1 - D(G(z)))] for original formulation)
        
        Args:
            real_scores: Discriminator scores for real data
            fake_scores: Discriminator scores for fake data
            loss_type: 'standard', 'wasserstein', or 'lsgan'
            
        Returns:
            CalculationResult with generator and discriminator losses
        """
        computation_steps = []
        intermediate_results = {}
        
        if loss_type == 'standard':
            return self._calculate_standard_gan(real_scores, fake_scores, computation_steps, intermediate_results)
        elif loss_type == 'wasserstein':
            return self._calculate_wgan(real_scores, fake_scores, computation_steps, intermediate_results)
        elif loss_type == 'lsgan':
            return self._calculate_lsgan(real_scores, fake_scores, computation_steps, intermediate_results)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _calculate_standard_gan(self, real_scores: np.ndarray, fake_scores: np.ndarray,
                              computation_steps: List, intermediate_results: Dict) -> CalculationResult:
        """Calculate standard GAN losses with binary cross-entropy."""
        
        # Numerical stability
        epsilon = 1e-10
        real_scores_stable = np.clip(real_scores, epsilon, 1 - epsilon)
        fake_scores_stable = np.clip(fake_scores, epsilon, 1 - epsilon)
        
        # Step 1: Discriminator loss on real data
        real_loss = -np.mean(np.log(real_scores_stable))
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="discriminator_real_loss",
            input_values={'real_scores': real_scores, 'real_scores_stable': real_scores_stable},
            operation_description="D loss (real): -E[log(D(x))]",
            output_values={'real_loss': np.array([real_loss])},
            visualization_hints={'loss_component': 'discriminator_real'}
        )
        computation_steps.append(step_1)
        intermediate_results['discriminator_real_loss'] = real_loss
        
        # Step 2: Discriminator loss on fake data
        fake_loss = -np.mean(np.log(1 - fake_scores_stable))
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="discriminator_fake_loss",
            input_values={'fake_scores': fake_scores, 'fake_scores_stable': fake_scores_stable},
            operation_description="D loss (fake): -E[log(1 - D(G(z)))]",
            output_values={'fake_loss': np.array([fake_loss])},
            visualization_hints={'loss_component': 'discriminator_fake'}
        )
        computation_steps.append(step_2)
        intermediate_results['discriminator_fake_loss'] = fake_loss
        
        # Step 3: Total discriminator loss
        discriminator_loss = real_loss + fake_loss
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="total_discriminator_loss",
            input_values={'real_loss': np.array([real_loss]), 'fake_loss': np.array([fake_loss])},
            operation_description="Total D loss: L_D = L_real + L_fake",
            output_values={'discriminator_loss': np.array([discriminator_loss])},
            visualization_hints={'loss_component': 'discriminator_total'}
        )
        computation_steps.append(step_3)
        intermediate_results['discriminator_loss'] = discriminator_loss
        
        # Step 4: Generator loss (alternative formulation for better gradients)
        generator_loss = -np.mean(np.log(fake_scores_stable))
        
        step_4 = ComputationStep(
            step_number=4,
            operation_name="generator_loss",
            input_values={'fake_scores': fake_scores, 'fake_scores_stable': fake_scores_stable},
            operation_description="G loss: -E[log(D(G(z)))]",
            output_values={'generator_loss': np.array([generator_loss])},
            visualization_hints={'loss_component': 'generator'}
        )
        computation_steps.append(step_4)
        intermediate_results['generator_loss'] = generator_loss
        
        # Create visualization
        visualization = self._create_gan_visualization(
            real_scores, fake_scores, discriminator_loss, generator_loss
        )
        
        # Calculate properties
        properties = {
            'operation': 'standard_gan',
            'discriminator_loss': discriminator_loss,
            'generator_loss': generator_loss,
            'discriminator_real_loss': real_loss,
            'discriminator_fake_loss': fake_loss,
            'real_scores_mean': np.mean(real_scores),
            'fake_scores_mean': np.mean(fake_scores),
            'discriminator_accuracy': (np.mean(real_scores > 0.5) + np.mean(fake_scores < 0.5)) / 2,
            'batch_size': len(real_scores)
        }
        
        return CalculationResult(
            result=np.array([discriminator_loss, generator_loss]),
            intermediate_results=intermediate_results,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _calculate_wgan(self, real_scores: np.ndarray, fake_scores: np.ndarray,
                       computation_steps: List, intermediate_results: Dict) -> CalculationResult:
        """Calculate Wasserstein GAN losses."""
        
        # Step 1: Discriminator (Critic) loss
        discriminator_loss = np.mean(fake_scores) - np.mean(real_scores)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="wgan_discriminator_loss",
            input_values={'real_scores': real_scores, 'fake_scores': fake_scores},
            operation_description="WGAN D loss: E[D(G(z))] - E[D(x)]",
            output_values={'discriminator_loss': np.array([discriminator_loss])},
            visualization_hints={'loss_type': 'wasserstein'}
        )
        computation_steps.append(step_1)
        intermediate_results['discriminator_loss'] = discriminator_loss
        
        # Step 2: Generator loss
        generator_loss = -np.mean(fake_scores)
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="wgan_generator_loss",
            input_values={'fake_scores': fake_scores},
            operation_description="WGAN G loss: -E[D(G(z))]",
            output_values={'generator_loss': np.array([generator_loss])},
            visualization_hints={'loss_type': 'wasserstein'}
        )
        computation_steps.append(step_2)
        intermediate_results['generator_loss'] = generator_loss
        
        # Create visualization
        visualization = self._create_gan_visualization(
            real_scores, fake_scores, discriminator_loss, generator_loss
        )
        
        properties = {
            'operation': 'wasserstein_gan',
            'discriminator_loss': discriminator_loss,
            'generator_loss': generator_loss,
            'wasserstein_distance': -discriminator_loss,
            'real_scores_mean': np.mean(real_scores),
            'fake_scores_mean': np.mean(fake_scores),
            'batch_size': len(real_scores)
        }
        
        return CalculationResult(
            result=np.array([discriminator_loss, generator_loss]),
            intermediate_results=intermediate_results,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _calculate_lsgan(self, real_scores: np.ndarray, fake_scores: np.ndarray,
                        computation_steps: List, intermediate_results: Dict) -> CalculationResult:
        """Calculate Least Squares GAN losses."""
        
        # Step 1: Discriminator loss
        real_loss = np.mean((real_scores - 1) ** 2)
        fake_loss = np.mean(fake_scores ** 2)
        discriminator_loss = 0.5 * (real_loss + fake_loss)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="lsgan_discriminator_loss",
            input_values={'real_scores': real_scores, 'fake_scores': fake_scores},
            operation_description="LSGAN D loss: 0.5 * [E[(D(x)-1)²] + E[D(G(z))²]]",
            output_values={
                'real_loss': np.array([real_loss]),
                'fake_loss': np.array([fake_loss]),
                'discriminator_loss': np.array([discriminator_loss])
            },
            visualization_hints={'loss_type': 'least_squares'}
        )
        computation_steps.append(step_1)
        intermediate_results['discriminator_loss'] = discriminator_loss
        intermediate_results['discriminator_real_loss'] = real_loss
        intermediate_results['discriminator_fake_loss'] = fake_loss
        
        # Step 2: Generator loss
        generator_loss = 0.5 * np.mean((fake_scores - 1) ** 2)
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="lsgan_generator_loss",
            input_values={'fake_scores': fake_scores},
            operation_description="LSGAN G loss: 0.5 * E[(D(G(z))-1)²]",
            output_values={'generator_loss': np.array([generator_loss])},
            visualization_hints={'loss_type': 'least_squares'}
        )
        computation_steps.append(step_2)
        intermediate_results['generator_loss'] = generator_loss
        
        # Create visualization
        visualization = self._create_gan_visualization(
            real_scores, fake_scores, discriminator_loss, generator_loss
        )
        
        properties = {
            'operation': 'lsgan',
            'discriminator_loss': discriminator_loss,
            'generator_loss': generator_loss,
            'discriminator_real_loss': real_loss,
            'discriminator_fake_loss': fake_loss,
            'real_scores_mean': np.mean(real_scores),
            'fake_scores_mean': np.mean(fake_scores),
            'batch_size': len(real_scores)
        }
        
        return CalculationResult(
            result=np.array([discriminator_loss, generator_loss]),
            intermediate_results=intermediate_results,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _create_gan_visualization(self, real_scores: np.ndarray, fake_scores: np.ndarray,
                                d_loss: float, g_loss: float) -> OperationVisualization:
        """Create visualization for GAN objective calculation."""
        
        input_matrices = [
            ColorCodedMatrix(real_scores.reshape(-1, 1), {'default': self.color_palette['real_scores']}, {}, []),
            ColorCodedMatrix(fake_scores.reshape(-1, 1), {'default': self.color_palette['fake_scores']}, {}, [])
        ]
        
        intermediate_steps = []
        
        output_matrix = ColorCodedMatrix(
            np.array([[d_loss], [g_loss]]),
            {'default': self.color_palette['discriminator_loss']},
            {},
            []
        )
        
        return OperationVisualization(
            operation_type="gan_objective",
            input_matrices=input_matrices,
            intermediate_steps=intermediate_steps,
            output_matrix=output_matrix,
            animation_sequence=[]
        )