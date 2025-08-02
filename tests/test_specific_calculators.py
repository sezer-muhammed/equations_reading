"""
Tests for specific equation calculators.
"""

import numpy as np
import pytest
from src.computation.specific_calculators import (
    SoftmaxCalculator, CrossEntropyCalculator, LSTMCalculator,
    VAEELBOCalculator, GANObjectiveCalculator
)


class TestSoftmaxCalculator:
    """Test softmax calculation with numerical stability."""
    
    def setup_method(self):
        self.calculator = SoftmaxCalculator()
    
    def test_basic_softmax(self):
        """Test basic softmax calculation."""
        logits = np.array([1.0, 2.0, 3.0])
        result = self.calculator.calculate(logits)
        
        # Check that probabilities sum to 1
        assert abs(np.sum(result.result) - 1.0) < 1e-6
        
        # Check that all probabilities are positive
        assert np.all(result.result > 0)
        
        # Check that higher logits get higher probabilities
        assert result.result[2] > result.result[1] > result.result[0]
        
        # Check computation steps
        assert len(result.computation_steps) >= 4
        assert any(step.operation_name == "numerical_stability_shift" for step in result.computation_steps)
        assert any(step.operation_name == "exponential" for step in result.computation_steps)
        assert any(step.operation_name == "normalize" for step in result.computation_steps)
    
    def test_softmax_with_temperature(self):
        """Test softmax with temperature scaling."""
        logits = np.array([1.0, 2.0, 3.0])
        
        # High temperature (more uniform)
        result_high_temp = self.calculator.calculate(logits, temperature=10.0)
        
        # Low temperature (more peaked)
        result_low_temp = self.calculator.calculate(logits, temperature=0.1)
        
        # High temperature should be more uniform
        entropy_high = -np.sum(result_high_temp.result * np.log(result_high_temp.result + 1e-10))
        entropy_low = -np.sum(result_low_temp.result * np.log(result_low_temp.result + 1e-10))
        
        assert entropy_high > entropy_low
        
        # Check properties
        assert result_high_temp.properties['temperature'] == 10.0
        assert result_low_temp.properties['temperature'] == 0.1
    
    def test_softmax_numerical_stability(self):
        """Test softmax with extreme values."""
        # Very large values
        large_logits = np.array([100.0, 101.0, 102.0])
        result_large = self.calculator.calculate(large_logits)
        
        # Should not produce NaN or inf
        assert not np.any(np.isnan(result_large.result))
        assert not np.any(np.isinf(result_large.result))
        assert abs(np.sum(result_large.result) - 1.0) < 1e-6
        
        # Very small (negative) values
        small_logits = np.array([-100.0, -101.0, -102.0])
        result_small = self.calculator.calculate(small_logits)
        
        assert not np.any(np.isnan(result_small.result))
        assert not np.any(np.isinf(result_small.result))
        assert abs(np.sum(result_small.result) - 1.0) < 1e-6
    
    def test_multidimensional_softmax(self):
        """Test softmax on multidimensional arrays."""
        logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = self.calculator.calculate(logits, axis=1)
        
        # Check that each row sums to 1
        row_sums = np.sum(result.result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2), decimal=6)
        
        # Check shape preservation
        assert result.result.shape == logits.shape


class TestCrossEntropyCalculator:
    """Test cross-entropy loss calculation."""
    
    def setup_method(self):
        self.calculator = CrossEntropyCalculator()
    
    def test_basic_cross_entropy(self):
        """Test basic cross-entropy calculation."""
        # Perfect predictions should give zero loss
        predictions = np.array([0.9, 0.05, 0.05])
        targets = np.array([1.0, 0.0, 0.0])
        
        result = self.calculator.calculate(predictions, targets)
        
        # Loss should be small for good predictions
        assert result.result < 1.0
        
        # Check computation steps
        assert len(result.computation_steps) == 5
        assert any(step.operation_name == "numerical_stability" for step in result.computation_steps)
        assert any(step.operation_name == "log_probabilities" for step in result.computation_steps)
        assert any(step.operation_name == "apply_reduction" for step in result.computation_steps)
    
    def test_cross_entropy_reductions(self):
        """Test different reduction methods."""
        predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        # Test mean reduction
        result_mean = self.calculator.calculate(predictions, targets, reduction='mean')
        
        # Test sum reduction
        result_sum = self.calculator.calculate(predictions, targets, reduction='sum')
        
        # Test no reduction
        result_none = self.calculator.calculate(predictions, targets, reduction='none')
        
        # Check relationships
        assert result_sum.result == result_mean.result * 2  # 2 samples
        assert result_none.result.shape == (2,)  # One loss per sample
        assert abs(np.mean(result_none.result) - result_mean.result) < 1e-6
    
    def test_cross_entropy_numerical_stability(self):
        """Test cross-entropy with extreme probability values."""
        # Very confident (but not perfect) predictions
        predictions = np.array([0.999, 0.0005, 0.0005])
        targets = np.array([1.0, 0.0, 0.0])
        
        result = self.calculator.calculate(predictions, targets, epsilon=1e-10)
        
        # Should not produce NaN
        assert not np.isnan(result.result)
        assert not np.isinf(result.result)
        
        # Check that epsilon was used
        assert 'epsilon' in result.properties
        assert result.properties['epsilon'] == 1e-10


class TestLSTMCalculator:
    """Test LSTM cell computation."""
    
    def setup_method(self):
        self.calculator = LSTMCalculator()
        
        # Set up LSTM parameters
        self.input_size = 3
        self.hidden_size = 4
        
        # Initialize weights and biases (normally these would be learned)
        np.random.seed(42)
        self.W_f = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.W_i = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.W_o = np.random.randn(self.hidden_size, self.input_size) * 0.1
        self.W_c = np.random.randn(self.hidden_size, self.input_size) * 0.1
        
        self.U_f = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.U_i = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.U_o = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.U_c = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        
        self.b_f = np.zeros(self.hidden_size)
        self.b_i = np.zeros(self.hidden_size)
        self.b_o = np.zeros(self.hidden_size)
        self.b_c = np.zeros(self.hidden_size)
    
    def test_lstm_forward_pass(self):
        """Test complete LSTM forward pass."""
        x_t = np.random.randn(self.input_size)
        h_prev = np.random.randn(self.hidden_size)
        c_prev = np.random.randn(self.hidden_size)
        
        result = self.calculator.calculate(
            x_t, h_prev, c_prev,
            self.W_f, self.W_i, self.W_o, self.W_c,
            self.U_f, self.U_i, self.U_o, self.U_c,
            self.b_f, self.b_i, self.b_o, self.b_c
        )
        
        # Check output shape (concatenated hidden and cell state)
        assert result.result.shape == (self.hidden_size * 2,)
        
        # Check that we have all computation steps
        assert len(result.computation_steps) == 6
        step_names = [step.operation_name for step in result.computation_steps]
        assert "forget_gate" in step_names
        assert "input_gate" in step_names
        assert "candidate_values" in step_names
        assert "update_cell_state" in step_names
        assert "output_gate" in step_names
        assert "update_hidden_state" in step_names
        
        # Check intermediate results
        assert 'forget_gate' in result.intermediate_results
        assert 'input_gate' in result.intermediate_results
        assert 'output_gate' in result.intermediate_results
        assert 'candidate_values' in result.intermediate_results
        assert 'cell_state' in result.intermediate_results
        
        # Check gate value ranges (should be between 0 and 1 for sigmoid gates)
        forget_gate = result.intermediate_results['forget_gate']
        input_gate = result.intermediate_results['input_gate']
        output_gate = result.intermediate_results['output_gate']
        
        assert np.all(forget_gate >= 0) and np.all(forget_gate <= 1)
        assert np.all(input_gate >= 0) and np.all(input_gate <= 1)
        assert np.all(output_gate >= 0) and np.all(output_gate <= 1)
        
        # Check candidate values (should be between -1 and 1 for tanh)
        candidate_values = result.intermediate_results['candidate_values']
        assert np.all(candidate_values >= -1) and np.all(candidate_values <= 1)
    
    def test_lstm_properties(self):
        """Test LSTM computation properties."""
        x_t = np.random.randn(self.input_size)
        h_prev = np.random.randn(self.hidden_size)
        c_prev = np.random.randn(self.hidden_size)
        
        result = self.calculator.calculate(
            x_t, h_prev, c_prev,
            self.W_f, self.W_i, self.W_o, self.W_c,
            self.U_f, self.U_i, self.U_o, self.U_c,
            self.b_f, self.b_i, self.b_o, self.b_c
        )
        
        # Check properties
        assert result.properties['operation'] == 'lstm_cell'
        assert result.properties['input_size'] == self.input_size
        assert result.properties['hidden_size'] == self.hidden_size
        assert 'forget_gate_mean' in result.properties
        assert 'input_gate_mean' in result.properties
        assert 'output_gate_mean' in result.properties
        assert 'cell_state_norm' in result.properties
        assert 'hidden_state_norm' in result.properties


class TestVAEELBOCalculator:
    """Test VAE ELBO calculation."""
    
    def setup_method(self):
        self.calculator = VAEELBOCalculator()
    
    def test_vae_elbo_calculation(self):
        """Test VAE ELBO calculation with KL divergence."""
        # Create test data
        x = np.random.randn(10)  # Original input
        x_recon = x + np.random.randn(10) * 0.1  # Reconstructed input (with small error)
        mu = np.random.randn(5)  # Latent mean
        logvar = np.random.randn(5) * 0.5  # Latent log variance
        
        result = self.calculator.calculate(x, x_recon, mu, logvar, beta=1.0)
        
        # Check computation steps
        assert len(result.computation_steps) == 5
        step_names = [step.operation_name for step in result.computation_steps]
        assert "reconstruction_loss" in step_names
        assert "mu_squared" in step_names
        assert "exp_logvar" in step_names
        assert "kl_divergence" in step_names
        assert "elbo_calculation" in step_names
        
        # Check intermediate results
        assert 'reconstruction_loss' in result.intermediate_results
        assert 'kl_divergence' in result.intermediate_results
        assert 'mu_squared' in result.intermediate_results
        assert 'exp_logvar' in result.intermediate_results
        
        # Check properties
        assert result.properties['operation'] == 'vae_elbo'
        assert result.properties['beta'] == 1.0
        assert result.properties['latent_dim'] == 5
        assert 'reconstruction_loss' in result.properties
        assert 'kl_divergence' in result.properties
        assert 'elbo' in result.properties
    
    def test_vae_beta_scaling(self):
        """Test VAE with different beta values (β-VAE)."""
        x = np.random.randn(10)
        x_recon = x + np.random.randn(10) * 0.1
        mu = np.random.randn(5)
        logvar = np.random.randn(5) * 0.5
        
        # Test with different beta values
        result_beta1 = self.calculator.calculate(x, x_recon, mu, logvar, beta=1.0)
        result_beta2 = self.calculator.calculate(x, x_recon, mu, logvar, beta=2.0)
        
        # KL divergence should be the same
        kl1 = result_beta1.intermediate_results['kl_divergence']
        kl2 = result_beta2.intermediate_results['kl_divergence']
        assert abs(kl1 - kl2) < 1e-10
        
        # But ELBO should be different due to beta scaling
        elbo1 = result_beta1.properties['elbo']
        elbo2 = result_beta2.properties['elbo']
        assert abs(elbo1 - elbo2) > 1e-6
        
        # Check beta in properties
        assert result_beta2.properties['beta'] == 2.0
    
    def test_vae_kl_divergence_properties(self):
        """Test KL divergence calculation properties."""
        # Test with standard normal latent (should have low KL)
        x = np.random.randn(10)
        x_recon = x
        mu = np.zeros(5)  # Mean close to 0
        logvar = np.zeros(5)  # Variance close to 1 (log(1) = 0)
        
        result = self.calculator.calculate(x, x_recon, mu, logvar)
        
        # KL divergence should be close to 0 for standard normal
        kl_div = result.intermediate_results['kl_divergence']
        assert abs(kl_div) < 1.0  # Should be small
        
        # Test with non-standard latent (should have higher KL)
        mu_large = np.ones(5) * 3.0  # Mean far from 0
        logvar_large = np.ones(5) * 2.0  # Large variance
        
        result_large = self.calculator.calculate(x, x_recon, mu_large, logvar_large)
        kl_div_large = result_large.intermediate_results['kl_divergence']
        
        assert kl_div_large > kl_div  # Should be larger


class TestGANObjectiveCalculator:
    """Test GAN objective calculation."""
    
    def setup_method(self):
        self.calculator = GANObjectiveCalculator()
    
    def test_standard_gan(self):
        """Test standard GAN loss calculation."""
        # Simulate discriminator scores
        real_scores = np.random.uniform(0.7, 0.9, 10)  # High scores for real data
        fake_scores = np.random.uniform(0.1, 0.3, 10)  # Low scores for fake data
        
        result = self.calculator.calculate(real_scores, fake_scores, loss_type='standard')
        
        # Check that we get both discriminator and generator losses
        assert result.result.shape == (2,)
        discriminator_loss, generator_loss = result.result
        
        # Check computation steps
        assert len(result.computation_steps) == 4
        step_names = [step.operation_name for step in result.computation_steps]
        assert "discriminator_real_loss" in step_names
        assert "discriminator_fake_loss" in step_names
        assert "total_discriminator_loss" in step_names
        assert "generator_loss" in step_names
        
        # Check properties
        assert result.properties['operation'] == 'standard_gan'
        assert 'discriminator_accuracy' in result.properties
        assert result.properties['batch_size'] == 10
    
    def test_wasserstein_gan(self):
        """Test Wasserstein GAN loss calculation."""
        real_scores = np.random.uniform(-1, 1, 10)  # WGAN scores can be any real number
        fake_scores = np.random.uniform(-1, 1, 10)
        
        result = self.calculator.calculate(real_scores, fake_scores, loss_type='wasserstein')
        
        # Check computation steps
        assert len(result.computation_steps) == 2
        step_names = [step.operation_name for step in result.computation_steps]
        assert "wgan_discriminator_loss" in step_names
        assert "wgan_generator_loss" in step_names
        
        # Check properties
        assert result.properties['operation'] == 'wasserstein_gan'
        assert 'wasserstein_distance' in result.properties
        
        # Wasserstein distance should be negative of discriminator loss
        w_distance = result.properties['wasserstein_distance']
        d_loss = result.properties['discriminator_loss']
        assert abs(w_distance + d_loss) < 1e-10
    
    def test_lsgan(self):
        """Test Least Squares GAN loss calculation."""
        real_scores = np.random.uniform(0.8, 1.2, 10)  # Around 1 for real data
        fake_scores = np.random.uniform(-0.2, 0.2, 10)  # Around 0 for fake data
        
        result = self.calculator.calculate(real_scores, fake_scores, loss_type='lsgan')
        
        # Check computation steps
        assert len(result.computation_steps) == 2
        step_names = [step.operation_name for step in result.computation_steps]
        assert "lsgan_discriminator_loss" in step_names
        assert "lsgan_generator_loss" in step_names
        
        # Check properties
        assert result.properties['operation'] == 'lsgan'
        assert 'discriminator_real_loss' in result.properties
        assert 'discriminator_fake_loss' in result.properties
    
    def test_gan_loss_comparison(self):
        """Test that different GAN formulations give different results."""
        real_scores = np.random.uniform(0.6, 0.9, 10)
        fake_scores = np.random.uniform(0.1, 0.4, 10)
        
        standard_result = self.calculator.calculate(real_scores, fake_scores, 'standard')
        wgan_result = self.calculator.calculate(real_scores, fake_scores, 'wasserstein')
        lsgan_result = self.calculator.calculate(real_scores, fake_scores, 'lsgan')
        
        # All should give different loss values
        standard_d_loss = standard_result.result[0]
        wgan_d_loss = wgan_result.result[0]
        lsgan_d_loss = lsgan_result.result[0]
        
        assert abs(standard_d_loss - wgan_d_loss) > 1e-6
        assert abs(standard_d_loss - lsgan_d_loss) > 1e-6
        assert abs(wgan_d_loss - lsgan_d_loss) > 1e-6
    
    def test_invalid_loss_type(self):
        """Test error handling for invalid loss type."""
        real_scores = np.random.uniform(0.5, 1.0, 5)
        fake_scores = np.random.uniform(0.0, 0.5, 5)
        
        with pytest.raises(ValueError, match="Unknown loss type"):
            self.calculator.calculate(real_scores, fake_scores, loss_type='invalid')


if __name__ == "__main__":
    pytest.main([__file__])