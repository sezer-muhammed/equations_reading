"""
Comprehensive attention mechanism tutorial with complete mathematical derivations.
Covers scaled dot-product attention, multi-head attention, and positional encodings.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ...computation.attention_ops.attention_mechanisms import AttentionMechanisms, AttentionResult
from ...computation.attention_ops.positional_encoding import PositionalEncodings, PositionalEncodingResult
from ...core.models import MathematicalConcept, Equation, NumericalExample, Explanation, VariableDefinition


@dataclass
class AttentionTutorialResult:
    """Complete attention tutorial with all components."""
    concept: MathematicalConcept
    attention_examples: List[AttentionResult]
    positional_examples: List[PositionalEncodingResult]
    tutorial_sections: List[Dict[str, any]]


class AttentionTutorial:
    """Complete tutorial for attention mechanisms and positional encodings."""
    
    def __init__(self):
        self.attention_mechanisms = AttentionMechanisms()
        self.positional_encodings = PositionalEncodings()
    
    def create_complete_tutorial(self) -> AttentionTutorialResult:
        """Create the complete attention mechanism tutorial."""
        
        # Create mathematical concept definition
        concept = self._create_attention_concept()
        
        # Generate attention examples
        attention_examples = self._generate_attention_examples()
        
        # Generate positional encoding examples
        positional_examples = self._generate_positional_examples()
        
        # Create tutorial sections
        tutorial_sections = self._create_tutorial_sections(
            attention_examples, positional_examples
        )
        
        return AttentionTutorialResult(
            concept=concept,
            attention_examples=attention_examples,
            positional_examples=positional_examples,
            tutorial_sections=tutorial_sections
        )
    
    def _create_attention_concept(self) -> MathematicalConcept:
        """Create the mathematical concept definition for attention."""
        
        # Define core equations
        equations = [
            Equation(
                equation_id="scaled_dot_product_attention",
                latex_expression=r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
                variables={
                    'Q': VariableDefinition("Q", "Query matrix (seq_len_q, d_k)", "matrix", (None, None)),
                    'K': VariableDefinition("K", "Key matrix (seq_len_k, d_k)", "matrix", (None, None)), 
                    'V': VariableDefinition("V", "Value matrix (seq_len_v, d_v)", "matrix", (None, None)),
                    'd_k': VariableDefinition("d_k", "Key dimension for scaling", "scalar", None)
                },
                derivation_steps=[],  # Will be filled with DerivationStep objects if needed
                mathematical_properties=[
                    "Attention weights sum to 1 (probability distribution)",
                    "Permutation equivariant with respect to key-value pairs",
                    "Scale invariant after softmax normalization"
                ],
                applications=[
                    "Machine translation (encoder-decoder attention)",
                    "Self-attention in transformers",
                    "Image captioning and visual attention"
                ]
            ),
            Equation(
                equation_id="multi_head_attention",
                latex_expression=r"\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O",
                variables={
                    'h': VariableDefinition("h", "Number of attention heads", "scalar", None),
                    'W^O': VariableDefinition("W^O", "Output projection matrix (d_model, d_model)", "matrix", (None, None))
                },
                derivation_steps=[],  # Will be filled with DerivationStep objects if needed
                mathematical_properties=[
                    "Each head learns different attention patterns",
                    "Parallel computation across heads",
                    "Total parameters: 4 * d_model^2"
                ],
                applications=[
                    "Transformer encoder and decoder layers",
                    "BERT and GPT architectures",
                    "Vision transformers (ViT)"
                ]
            ),
            Equation(
                equation_id="absolute_positional_encoding",
                latex_expression=r"PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)",
                variables={
                    'pos': VariableDefinition("pos", "Position in sequence", "scalar", None),
                    'i': VariableDefinition("i", "Dimension index", "scalar", None),
                    'd_model': VariableDefinition("d_model", "Model dimension", "scalar", None)
                },
                derivation_steps=[],  # Will be filled with DerivationStep objects if needed
                mathematical_properties=[
                    "Unique encoding for each position",
                    "Relative position information preserved",
                    "Extrapolates to longer sequences"
                ],
                applications=[
                    "Original transformer positional encoding",
                    "BERT absolute position embeddings",
                    "Sequence modeling tasks"
                ]
            ),
            Equation(
                equation_id="rotary_positional_embedding",
                latex_expression=r"\text{RoPE}(x, pos) = \begin{pmatrix} \cos(pos \cdot \theta) & -\sin(pos \cdot \theta) \\ \sin(pos \cdot \theta) & \cos(pos \cdot \theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}",
                variables={
                    'x': VariableDefinition("x", "Input vector pair (x_1, x_2)", "vector", (2,)),
                    'pos': VariableDefinition("pos", "Position index", "scalar", None),
                    'θ': VariableDefinition("θ", "Frequency: 1/10000^(2i/d)", "scalar", None)
                },
                derivation_steps=[],  # Will be filled with DerivationStep objects if needed
                mathematical_properties=[
                    "Preserves relative position in attention scores",
                    "Rotation matrices maintain vector norms",
                    "Extrapolates naturally to longer sequences"
                ],
                applications=[
                    "GPT-NeoX and PaLM architectures",
                    "Long sequence modeling",
                    "Improved extrapolation capabilities"
                ]
            )
        ]
        
        # Create explanations
        explanations = [
            Explanation(
                explanation_type="intuitive",
                content="""
                Attention mechanisms allow models to focus on relevant parts of the input when processing each element.
                Think of it as a spotlight that can dynamically adjust its focus based on the current context.
                
                The key insight is that not all input elements are equally important for predicting each output element.
                Attention learns these importance weights automatically from data.
                """,
                mathematical_level=2
            ),
            Explanation(
                explanation_type="formal",
                content="""
                The scaling factor √d_k prevents the dot products from becoming too large, which would push the softmax
                function into regions with extremely small gradients (saturation).
                
                For large d_k, dot products QK^T have variance d_k. Scaling by √d_k normalizes the variance to 1,
                keeping the softmax input in a reasonable range for stable gradients.
                """,
                mathematical_level=4
            ),
            Explanation(
                explanation_type="practical",
                content="""
                Multi-head attention allows the model to attend to information from different representation subspaces
                at different positions simultaneously.
                
                Each head can learn to focus on different types of relationships:
                - Syntactic relationships (subject-verb agreement)
                - Semantic relationships (word meanings)
                - Positional relationships (nearby words)
                - Long-range dependencies
                """,
                mathematical_level=4
            )
        ]
        
        return MathematicalConcept(
            concept_id="attention_mechanisms",
            title="Attention Mechanisms and Positional Encodings",
            prerequisites=["linear_algebra", "probability_theory", "neural_networks"],
            equations=equations,
            explanations=explanations,
            examples=[],  # Will be filled by numerical examples
            visualizations=[],  # Will be filled by computation results
            difficulty_level=4  # Graduate level
        )
    
    def _generate_attention_examples(self) -> List[AttentionResult]:
        """Generate comprehensive attention mechanism examples."""
        examples = []
        
        # Example 1: Small-scale scaled dot-product attention
        seq_len, d_k, d_v = 4, 8, 6
        Q = np.random.randn(seq_len, d_k) * 0.5
        K = np.random.randn(seq_len, d_k) * 0.5
        V = np.random.randn(seq_len, d_v) * 0.5
        
        attention_result = self.attention_mechanisms.scaled_dot_product_attention(Q, K, V)
        examples.append(attention_result)
        
        # Example 2: Multi-head attention
        seq_len, d_model, num_heads = 6, 12, 3
        Q_multi = np.random.randn(seq_len, d_model) * 0.5
        K_multi = np.random.randn(seq_len, d_model) * 0.5
        V_multi = np.random.randn(seq_len, d_model) * 0.5
        
        multihead_result = self.attention_mechanisms.multi_head_attention(
            Q_multi, K_multi, V_multi, num_heads, d_model
        )
        examples.append(multihead_result)
        
        # Example 3: Attention with masking (causal attention)
        seq_len = 5
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)  # Upper triangular mask
        Q_masked = np.random.randn(seq_len, d_k) * 0.5
        K_masked = np.random.randn(seq_len, d_k) * 0.5
        V_masked = np.random.randn(seq_len, d_v) * 0.5
        
        masked_result = self.attention_mechanisms.scaled_dot_product_attention(
            Q_masked, K_masked, V_masked, mask=mask
        )
        examples.append(masked_result)
        
        return examples
    
    def _generate_positional_examples(self) -> List[PositionalEncodingResult]:
        """Generate positional encoding examples."""
        examples = []
        
        # Example 1: Absolute positional encoding
        seq_len, d_model = 8, 16
        abs_pe_result = self.positional_encodings.absolute_positional_encoding(seq_len, d_model)
        examples.append(abs_pe_result)
        
        # Example 2: RoPE (Rotary Positional Embedding)
        seq_len, d_head = 6, 8
        x_input = np.random.randn(seq_len, d_head) * 0.5
        rope_result = self.positional_encodings.rotary_positional_embedding(x_input, seq_len, d_head)
        examples.append(rope_result)
        
        # Example 3: Relative positional encoding
        seq_len, d_model = 6, 12
        rel_pe_result = self.positional_encodings.relative_positional_encoding(seq_len, d_model)
        examples.append(rel_pe_result)
        
        return examples
    
    def _create_tutorial_sections(self, attention_examples: List[AttentionResult],
                                positional_examples: List[PositionalEncodingResult]) -> List[Dict[str, any]]:
        """Create structured tutorial sections."""
        
        sections = [
            {
                'title': 'Introduction to Attention Mechanisms',
                'content': """
                Attention mechanisms revolutionized deep learning by allowing models to dynamically focus on 
                relevant parts of the input. This tutorial covers the mathematical foundations and practical 
                implementations of attention, from basic scaled dot-product attention to advanced positional 
                encoding schemes.
                """,
                'learning_objectives': [
                    'Understand the mathematical formulation of attention',
                    'Implement scaled dot-product and multi-head attention',
                    'Master positional encoding techniques',
                    'Apply attention mechanisms to real problems'
                ]
            },
            {
                'title': 'Scaled Dot-Product Attention',
                'content': """
                The core attention mechanism computes a weighted sum of values, where weights are determined
                by the compatibility between queries and keys.
                """,
                'mathematical_derivation': attention_examples[0].computation_steps,
                'numerical_example': attention_examples[0],
                'key_insights': [
                    'Scaling prevents softmax saturation',
                    'Attention weights form a probability distribution',
                    'Computational complexity is O(n²d) for sequence length n'
                ]
            },
            {
                'title': 'Multi-Head Attention',
                'content': """
                Multi-head attention allows the model to jointly attend to information from different 
                representation subspaces at different positions.
                """,
                'mathematical_derivation': attention_examples[1].computation_steps,
                'numerical_example': attention_examples[1],
                'key_insights': [
                    'Each head learns different attention patterns',
                    'Parallel computation improves efficiency',
                    'Concatenation preserves information from all heads'
                ]
            },
            {
                'title': 'Positional Encodings',
                'content': """
                Since attention mechanisms are permutation-invariant, we need to inject positional 
                information to distinguish between different positions in the sequence.
                """,
                'subsections': [
                    {
                        'title': 'Absolute Sinusoidal Encoding',
                        'example': positional_examples[0],
                        'properties': [
                            'Unique encoding for each position',
                            'Deterministic (no learned parameters)',
                            'Extrapolates to longer sequences'
                        ]
                    },
                    {
                        'title': 'Rotary Positional Embedding (RoPE)',
                        'example': positional_examples[1],
                        'properties': [
                            'Preserves relative position information',
                            'Better extrapolation to longer sequences',
                            'Used in modern large language models'
                        ]
                    },
                    {
                        'title': 'Relative Positional Encoding',
                        'example': positional_examples[2],
                        'properties': [
                            'Learned relative position embeddings',
                            'Captures local and global dependencies',
                            'Flexible but requires more parameters'
                        ]
                    }
                ]
            },
            {
                'title': 'Advanced Topics and Applications',
                'content': """
                Attention mechanisms form the backbone of modern transformer architectures and have 
                applications across many domains.
                """,
                'topics': [
                    'Causal (masked) attention for autoregressive models',
                    'Cross-attention for encoder-decoder architectures',
                    'Sparse attention patterns for long sequences',
                    'Attention visualization and interpretability'
                ],
                'masked_example': attention_examples[2]
            }
        ]
        
        return sections