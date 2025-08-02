"""
Comprehensive transformer architecture tutorial with complete mathematical derivations.
Covers transformer blocks, layer normalization, feed-forward networks, and embeddings.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ...computation.attention_ops.transformer_components import (
    TransformerComponents, TransformerBlockResult, LayerNormResult, FeedForwardResult
)
from ...computation.attention_ops.embedding_tokenization import (
    EmbeddingTokenization, EmbeddingResult, TokenizationResult
)
from ...core.models import MathematicalConcept, Equation, NumericalExample, Explanation, VariableDefinition


@dataclass
class TransformerTutorialResult:
    """Complete transformer tutorial with all components."""
    concept: MathematicalConcept
    transformer_examples: List[TransformerBlockResult]
    layer_norm_comparison: Dict[str, LayerNormResult]
    feedforward_examples: List[FeedForwardResult]
    embedding_examples: List[EmbeddingResult]
    tokenization_examples: List[TokenizationResult]
    tutorial_sections: List[Dict[str, any]]


class TransformerTutorial:
    """Complete tutorial for transformer architecture."""
    
    def __init__(self):
        self.transformer_components = TransformerComponents()
        self.embedding_tokenization = EmbeddingTokenization()
    
    def create_complete_tutorial(self) -> TransformerTutorialResult:
        """Create the complete transformer architecture tutorial."""
        
        # Create mathematical concept definition
        concept = self._create_transformer_concept()
        
        # Generate transformer block examples
        transformer_examples = self._generate_transformer_examples()
        
        # Generate layer normalization comparison
        layer_norm_comparison = self._generate_layer_norm_comparison()
        
        # Generate feed-forward network examples
        feedforward_examples = self._generate_feedforward_examples()
        
        # Generate embedding examples
        embedding_examples = self._generate_embedding_examples()
        
        # Generate tokenization examples
        tokenization_examples = self._generate_tokenization_examples()
        
        # Create tutorial sections
        tutorial_sections = self._create_tutorial_sections(
            transformer_examples, layer_norm_comparison, feedforward_examples,
            embedding_examples, tokenization_examples
        )
        
        return TransformerTutorialResult(
            concept=concept,
            transformer_examples=transformer_examples,
            layer_norm_comparison=layer_norm_comparison,
            feedforward_examples=feedforward_examples,
            embedding_examples=embedding_examples,
            tokenization_examples=tokenization_examples,
            tutorial_sections=tutorial_sections
        )
    
    def _create_transformer_concept(self) -> MathematicalConcept:
        """Create the mathematical concept definition for transformer architecture."""
        
        # Define core equations
        equations = [
            Equation(
                equation_id="transformer_block",
                latex_expression=r"\text{TransformerBlock}(x) = \text{LayerNorm}(x + \text{FFN}(\text{LayerNorm}(x + \text{MultiHead}(x, x, x))))",
                variables={
                    'x': VariableDefinition("x", "Input sequence (seq_len, d_model)", "matrix", (None, None)),
                    'MultiHead': VariableDefinition("MultiHead", "Multi-head attention function", "function", None),
                    'FFN': VariableDefinition("FFN", "Feed-forward network", "function", None),
                    'LayerNorm': VariableDefinition("LayerNorm", "Layer normalization", "function", None)
                },
                derivation_steps=[],
                mathematical_properties=[
                    "Residual connections enable gradient flow",
                    "Layer normalization stabilizes training",
                    "Self-attention captures long-range dependencies",
                    "Feed-forward provides non-linear transformations"
                ],
                applications=[
                    "Language modeling (GPT family)",
                    "Machine translation (original transformer)",
                    "Text classification (BERT)",
                    "Vision tasks (Vision Transformer)"
                ]
            ),
            Equation(
                equation_id="layer_normalization",
                latex_expression=r"\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta",
                variables={
                    'x': VariableDefinition("x", "Input tensor", "tensor", None),
                    'μ': VariableDefinition("μ", "Mean along feature dimension", "scalar", None),
                    'σ²': VariableDefinition("σ²", "Variance along feature dimension", "scalar", None),
                    'γ': VariableDefinition("γ", "Learnable scale parameter", "vector", None),
                    'β': VariableDefinition("β", "Learnable shift parameter", "vector", None),
                    'ε': VariableDefinition("ε", "Small constant for numerical stability", "scalar", None)
                },
                derivation_steps=[],
                mathematical_properties=[
                    "Normalizes each sample independently",
                    "Reduces internal covariate shift",
                    "Enables higher learning rates",
                    "Invariant to input scale"
                ],
                applications=[
                    "Transformer architectures",
                    "Residual networks",
                    "Recurrent neural networks",
                    "Any deep architecture requiring stable training"
                ]
            ),
            Equation(
                equation_id="feed_forward_network",
                latex_expression=r"\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2",
                variables={
                    'x': VariableDefinition("x", "Input tensor (seq_len, d_model)", "matrix", (None, None)),
                    'W₁': VariableDefinition("W₁", "First weight matrix (d_model, d_ff)", "matrix", (None, None)),
                    'b₁': VariableDefinition("b₁", "First bias vector (d_ff,)", "vector", None),
                    'W₂': VariableDefinition("W₂", "Second weight matrix (d_ff, d_model)", "matrix", (None, None)),
                    'b₂': VariableDefinition("b₂", "Second bias vector (d_model,)", "vector", None),
                    'd_ff': VariableDefinition("d_ff", "Hidden dimension (typically 4×d_model)", "scalar", None)
                },
                derivation_steps=[],
                mathematical_properties=[
                    "Two-layer MLP with ReLU activation",
                    "Expansion-contraction architecture",
                    "Position-wise application (same across sequence)",
                    "Provides non-linear transformations"
                ],
                applications=[
                    "Transformer feed-forward sublayers",
                    "MLP blocks in various architectures",
                    "Feature transformation layers",
                    "Non-linear processing components"
                ]
            ),
            Equation(
                equation_id="token_embedding",
                latex_expression=r"E = \text{Embedding}(\text{tokens}) + \text{PositionalEncoding}(\text{positions})",
                variables={
                    'E': VariableDefinition("E", "Final embeddings (seq_len, d_model)", "matrix", (None, None)),
                    'tokens': VariableDefinition("tokens", "Token IDs (seq_len,)", "vector", None),
                    'positions': VariableDefinition("positions", "Position indices (seq_len,)", "vector", None),
                    'Embedding': VariableDefinition("Embedding", "Token embedding lookup", "function", None),
                    'PositionalEncoding': VariableDefinition("PositionalEncoding", "Position encoding function", "function", None)
                },
                derivation_steps=[],
                mathematical_properties=[
                    "Combines semantic and positional information",
                    "Learnable token embeddings",
                    "Fixed or learnable positional encodings",
                    "Input representation for transformer"
                ],
                applications=[
                    "All transformer-based models",
                    "Sequence-to-sequence tasks",
                    "Language modeling",
                    "Text classification"
                ]
            )
        ]
        
        # Create explanations
        explanations = [
            Explanation(
                explanation_type="intuitive",
                content="""
                The transformer architecture revolutionized deep learning by replacing recurrent connections
                with self-attention mechanisms. This allows for parallel processing and better capture of
                long-range dependencies in sequences.
                
                Key innovations:
                1. Self-attention replaces recurrence
                2. Residual connections enable deep networks
                3. Layer normalization stabilizes training
                4. Position encodings provide sequence order information
                """,
                mathematical_level=3
            ),
            Explanation(
                explanation_type="formal",
                content="""
                The transformer block applies two main transformations with residual connections:
                
                1. Multi-head self-attention: Captures relationships between all positions
                2. Position-wise feed-forward: Applies non-linear transformations
                
                Each sublayer is wrapped with residual connection and layer normalization:
                LayerNorm(x + Sublayer(x))
                
                This design enables training very deep networks while maintaining gradient flow.
                """,
                mathematical_level=4
            ),
            Explanation(
                explanation_type="practical",
                content="""
                Layer normalization vs batch normalization in transformers:
                
                - Layer norm: Normalizes across features for each sample independently
                - Batch norm: Normalizes across batch dimension for each feature
                
                Layer norm is preferred in transformers because:
                1. Works with variable sequence lengths
                2. No dependence on batch statistics during inference
                3. More stable for sequential data
                4. Better performance in practice for NLP tasks
                """,
                mathematical_level=4
            )
        ]
        
        return MathematicalConcept(
            concept_id="transformer_architecture",
            title="Transformer Architecture and Components",
            prerequisites=["attention_mechanisms", "neural_networks", "linear_algebra"],
            equations=equations,
            explanations=explanations,
            examples=[],  # Will be filled by numerical examples
            visualizations=[],  # Will be filled by computation results
            difficulty_level=4  # Graduate level
        )
    
    def _generate_transformer_examples(self) -> List[TransformerBlockResult]:
        """Generate comprehensive transformer block examples."""
        examples = []
        
        # Example 1: Small transformer block
        seq_len, d_model, num_heads, d_ff = 6, 12, 3, 48
        x_small = np.random.randn(seq_len, d_model) * 0.5
        
        transformer_result = self.transformer_components.transformer_block(
            x_small, num_heads, d_ff
        )
        examples.append(transformer_result)
        
        # Example 2: Larger transformer block
        seq_len, d_model, num_heads, d_ff = 8, 16, 4, 64
        x_large = np.random.randn(seq_len, d_model) * 0.5
        
        transformer_result_large = self.transformer_components.transformer_block(
            x_large, num_heads, d_ff
        )
        examples.append(transformer_result_large)
        
        return examples
    
    def _generate_layer_norm_comparison(self) -> Dict[str, LayerNormResult]:
        """Generate layer normalization vs batch normalization comparison."""
        
        # Create sample data (batch_size=3, seq_len=4, d_model=6)
        x = np.random.randn(3, 4, 6) * 2.0 + 1.0  # Add some bias and scale
        
        # Layer normalization
        layer_norm_result = self.transformer_components.layer_normalization(x)
        
        # Batch normalization
        batch_norm_result = self.transformer_components.batch_normalization(x)
        
        return {
            'layer_norm': layer_norm_result,
            'batch_norm': batch_norm_result
        }
    
    def _generate_feedforward_examples(self) -> List[FeedForwardResult]:
        """Generate feed-forward network examples."""
        examples = []
        
        # Example 1: Standard feed-forward with ReLU
        seq_len, d_model, d_ff = 5, 8, 32
        x_relu = np.random.randn(seq_len, d_model) * 0.5
        
        ff_relu_result = self.transformer_components.feed_forward_network(
            x_relu, d_ff, activation='relu'
        )
        examples.append(ff_relu_result)
        
        # Example 2: Feed-forward with GELU activation
        ff_gelu_result = self.transformer_components.feed_forward_network(
            x_relu, d_ff, activation='gelu'
        )
        examples.append(ff_gelu_result)
        
        return examples
    
    def _generate_embedding_examples(self) -> List[EmbeddingResult]:
        """Generate embedding layer examples."""
        examples = []
        
        # Example 1: Simple embedding with positional encoding
        token_ids = np.array([1, 5, 3, 8, 2, 7])  # Sample token sequence
        vocab_size, d_model, max_seq_len = 10, 8, 20
        
        embedding_result = self.embedding_tokenization.embedding_layer(
            token_ids, vocab_size, d_model, max_seq_len, use_positional=True
        )
        examples.append(embedding_result)
        
        # Example 2: Embedding without positional encoding
        embedding_no_pos = self.embedding_tokenization.embedding_layer(
            token_ids, vocab_size, d_model, max_seq_len, use_positional=False
        )
        examples.append(embedding_no_pos)
        
        return examples
    
    def _generate_tokenization_examples(self) -> List[TokenizationResult]:
        """Generate tokenization examples."""
        examples = []
        
        # Example 1: Simple word-level tokenization
        text = "The quick brown fox jumps over the lazy dog"
        simple_tokenization = self.embedding_tokenization.simple_tokenization(text)
        examples.append(simple_tokenization)
        
        # Example 2: BPE tokenization demonstration
        bpe_text = "hello world this is a test"
        bpe_tokenization = self.embedding_tokenization.bpe_tokenization_demo(bpe_text, max_merges=3)
        examples.append(bpe_tokenization)
        
        return examples
    
    def _create_tutorial_sections(self, transformer_examples: List[TransformerBlockResult],
                                layer_norm_comparison: Dict[str, LayerNormResult],
                                feedforward_examples: List[FeedForwardResult],
                                embedding_examples: List[EmbeddingResult],
                                tokenization_examples: List[TokenizationResult]) -> List[Dict[str, any]]:
        """Create structured tutorial sections."""
        
        sections = [
            {
                'title': 'Introduction to Transformer Architecture',
                'content': """
                The transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017),
                revolutionized sequence modeling by replacing recurrent connections with self-attention mechanisms.
                This tutorial covers the complete mathematical foundations and implementation details.
                """,
                'learning_objectives': [
                    'Understand the complete transformer block architecture',
                    'Master layer normalization vs batch normalization',
                    'Implement feed-forward networks with different activations',
                    'Build embedding layers with positional encoding',
                    'Understand tokenization processes (word-level and BPE)'
                ],
                'key_innovations': [
                    'Self-attention replaces recurrence for parallelization',
                    'Residual connections enable very deep networks',
                    'Layer normalization provides training stability',
                    'Position encodings inject sequence order information'
                ]
            },
            {
                'title': 'Complete Transformer Block',
                'content': """
                A transformer block consists of two main sublayers, each wrapped with residual connections
                and layer normalization:
                
                1. Multi-head self-attention sublayer
                2. Position-wise feed-forward network sublayer
                
                The mathematical formulation is:
                x₁ = LayerNorm(x + MultiHeadAttention(x, x, x))
                x₂ = LayerNorm(x₁ + FFN(x₁))
                """,
                'mathematical_derivation': transformer_examples[0].computation_steps,
                'numerical_example': transformer_examples[0],
                'architecture_details': {
                    'residual_connections': 'Enable gradient flow in deep networks',
                    'layer_normalization': 'Applied after each sublayer',
                    'self_attention': 'Q, K, V all come from the same input',
                    'feed_forward': 'Position-wise MLP with expansion-contraction'
                },
                'parameter_analysis': transformer_examples[0].properties
            },
            {
                'title': 'Layer Normalization vs Batch Normalization',
                'content': """
                Layer normalization is preferred over batch normalization in transformers due to several
                key advantages for sequential data processing.
                """,
                'comparison_table': {
                    'normalization_axis': {
                        'layer_norm': 'Across features (last dimension)',
                        'batch_norm': 'Across batch (first dimension)'
                    },
                    'independence': {
                        'layer_norm': 'Each sample normalized independently',
                        'batch_norm': 'Depends on batch statistics'
                    },
                    'inference': {
                        'layer_norm': 'No running statistics needed',
                        'batch_norm': 'Uses running mean/variance'
                    },
                    'sequence_length': {
                        'layer_norm': 'Works with variable lengths',
                        'batch_norm': 'Fixed statistics per position'
                    }
                },
                'layer_norm_example': layer_norm_comparison['layer_norm'],
                'batch_norm_example': layer_norm_comparison['batch_norm'],
                'mathematical_comparison': {
                    'layer_norm_formula': 'γ ⊙ (x - μₗ)/σₗ + β where μₗ, σₗ computed per sample',
                    'batch_norm_formula': 'γ ⊙ (x - μᵦ)/σᵦ + β where μᵦ, σᵦ computed per batch'
                }
            },
            {
                'title': 'Feed-Forward Networks',
                'content': """
                The position-wise feed-forward network applies the same MLP to each position independently.
                It consists of two linear transformations with an activation function in between.
                """,
                'architecture': {
                    'expansion_ratio': 'Typically d_ff = 4 × d_model',
                    'activation_functions': ['ReLU (original)', 'GELU (modern variants)', 'SwiGLU (recent)'],
                    'parameter_sharing': 'Same weights applied to all positions',
                    'computational_complexity': 'O(seq_len × d_model × d_ff)'
                },
                'relu_example': feedforward_examples[0],
                'gelu_example': feedforward_examples[1],
                'activation_comparison': {
                    'relu': {
                        'formula': 'max(0, x)',
                        'properties': ['Simple computation', 'Sparse activations', 'Dead neuron problem'],
                        'sparsity': feedforward_examples[0].properties.get('sparsity', 0)
                    },
                    'gelu': {
                        'formula': '0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))',
                        'properties': ['Smooth activation', 'Better gradients', 'Used in BERT/GPT'],
                        'sparsity': feedforward_examples[1].properties.get('sparsity', 0)
                    }
                }
            },
            {
                'title': 'Embedding Layers and Positional Encoding',
                'content': """
                Embedding layers convert discrete tokens into continuous vector representations.
                Positional encodings are added to provide sequence order information since
                self-attention is permutation-invariant.
                """,
                'embedding_process': {
                    'token_embedding': 'Learnable lookup table: vocab_size × d_model',
                    'positional_encoding': 'Fixed sinusoidal or learnable position embeddings',
                    'combination': 'Element-wise addition of token and position embeddings'
                },
                'with_positional_example': embedding_examples[0],
                'without_positional_example': embedding_examples[1],
                'positional_encoding_types': {
                    'sinusoidal': {
                        'formula': 'PE(pos,2i) = sin(pos/10000^(2i/d_model))',
                        'properties': ['Fixed (no parameters)', 'Extrapolates to longer sequences', 'Relative position info']
                    },
                    'learned': {
                        'formula': 'Learnable embedding matrix',
                        'properties': ['Trainable parameters', 'Fixed maximum length', 'Task-specific optimization']
                    }
                },
                'mathematical_properties': {
                    'embedding_dimension': embedding_examples[0].properties['d_model'],
                    'vocabulary_size': embedding_examples[0].properties['vocab_size'],
                    'parameter_count': embedding_examples[0].properties['parameter_count']
                }
            },
            {
                'title': 'Tokenization Processes',
                'content': """
                Tokenization converts raw text into discrete tokens that can be processed by the model.
                Different tokenization strategies have various trade-offs between vocabulary size,
                representation efficiency, and handling of out-of-vocabulary words.
                """,
                'tokenization_strategies': {
                    'word_level': {
                        'description': 'Split text by whitespace and punctuation',
                        'advantages': ['Simple implementation', 'Preserves word boundaries'],
                        'disadvantages': ['Large vocabulary', 'OOV problem', 'Morphologically poor'],
                        'example': tokenization_examples[0]
                    },
                    'subword_bpe': {
                        'description': 'Byte Pair Encoding merges frequent character pairs',
                        'advantages': ['Smaller vocabulary', 'Handles OOV', 'Morphological awareness'],
                        'disadvantages': ['More complex', 'Requires preprocessing'],
                        'example': tokenization_examples[1]
                    }
                },
                'bpe_algorithm': {
                    'steps': [
                        '1. Initialize with character-level tokens',
                        '2. Count adjacent character pair frequencies',
                        '3. Merge most frequent pair',
                        '4. Repeat until desired vocabulary size'
                    ],
                    'merge_history': tokenization_examples[1].properties.get('merge_history', []),
                    'compression_ratio': tokenization_examples[1].properties.get('compression_ratio', 1.0)
                }
            },
            {
                'title': 'Complete Architecture Integration',
                'content': """
                The complete transformer architecture integrates all components:
                
                1. Tokenization → Token IDs
                2. Embedding Layer → Token + Positional Embeddings  
                3. N × Transformer Blocks → Contextualized Representations
                4. Output Layer → Task-specific Predictions
                """,
                'architecture_flow': {
                    'input_processing': 'Text → Tokens → Embeddings',
                    'representation_learning': 'Embeddings → Transformer Blocks → Contextualized Features',
                    'output_generation': 'Features → Linear Layer → Predictions'
                },
                'scaling_properties': {
                    'parameter_count': 'O(vocab_size × d_model + N × d_model²)',
                    'computational_complexity': 'O(N × seq_len² × d_model)',
                    'memory_usage': 'O(N × seq_len × d_model)'
                },
                'practical_considerations': [
                    'Gradient clipping for training stability',
                    'Learning rate scheduling (warmup + decay)',
                    'Dropout for regularization',
                    'Mixed precision training for efficiency'
                ]
            }
        ]
        
        return sections