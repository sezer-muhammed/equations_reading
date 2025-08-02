"""
Embedding layer and tokenization process implementations.
Covers word embeddings, positional embeddings, and tokenization visualization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ...core.models import (
    ColorCodedMatrix, OperationVisualization, ComputationStep, VisualizationData
)


@dataclass
class TokenizationResult:
    """Result of tokenization process."""
    tokens: List[str]
    token_ids: np.ndarray
    subword_mapping: Dict[int, List[str]]
    computation_steps: List[ComputationStep]
    properties: Dict[str, any]


@dataclass
class EmbeddingResult:
    """Result of embedding layer computation."""
    embeddings: np.ndarray
    token_embeddings: np.ndarray
    positional_embeddings: np.ndarray
    computation_steps: List[ComputationStep]
    visualization: OperationVisualization
    properties: Dict[str, any]


class EmbeddingTokenization:
    """Implementation of embedding layers and tokenization processes."""
    
    def __init__(self):
        self.color_palette = {
            'tokens': '#FF6B6B',          # Red for tokens
            'embeddings': '#4ECDC4',      # Teal for embeddings
            'positional': '#45B7D1',      # Blue for positional
            'combined': '#96CEB4',        # Green for combined
            'vocabulary': '#FFEAA7',      # Yellow for vocabulary
            'subwords': '#DDA0DD'         # Purple for subwords
        }
    
    def simple_tokenization(self, text: str, vocabulary: Optional[Dict[str, int]] = None) -> TokenizationResult:
        """
        Perform simple word-level tokenization for demonstration.
        
        Args:
            text: Input text to tokenize
            vocabulary: Optional vocabulary mapping
            
        Returns:
            TokenizationResult with tokenization breakdown
        """
        computation_steps = []
        
        # Step 1: Split text into words
        words = text.lower().split()
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="text_splitting",
            input_values={'text': np.array([text])},
            operation_description="Split text into words (whitespace tokenization)",
            output_values={'words': np.array(words)},
            visualization_hints={'operation': 'text_split'}
        )
        computation_steps.append(step_1)
        
        # Create vocabulary if not provided
        if vocabulary is None:
            unique_words = list(set(words))
            vocabulary = {word: idx for idx, word in enumerate(unique_words)}
            vocabulary['<UNK>'] = len(vocabulary)  # Unknown token
            vocabulary['<PAD>'] = len(vocabulary)  # Padding token
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="vocabulary_creation",
            input_values={'words': np.array(words)},
            operation_description=f"Create vocabulary with {len(vocabulary)} tokens",
            output_values={'vocabulary': np.array(list(vocabulary.keys()))},
            visualization_hints={'operation': 'vocabulary_build'}
        )
        computation_steps.append(step_2)
        
        # Step 3: Convert words to token IDs
        token_ids = np.array([vocabulary.get(word, vocabulary['<UNK>']) for word in words])
        
        step_3 = ComputationStep(
            step_number=3,
            operation_name="word_to_id_mapping",
            input_values={'words': np.array(words), 'vocabulary': np.array(list(vocabulary.keys()))},
            operation_description="Map words to token IDs using vocabulary",
            output_values={'token_ids': token_ids},
            visualization_hints={'operation': 'id_mapping'}
        )
        computation_steps.append(step_3)
        
        # Create subword mapping (for simple tokenization, each word is one subword)
        subword_mapping = {i: [words[i]] for i in range(len(words))}
        
        properties = {
            'operation': 'simple_tokenization',
            'input_text': text,
            'num_tokens': len(words),
            'vocabulary_size': len(vocabulary),
            'unique_tokens': len(set(words)),
            'tokenization_type': 'word_level'
        }
        
        return TokenizationResult(
            tokens=words,
            token_ids=token_ids,
            subword_mapping=subword_mapping,
            computation_steps=computation_steps,
            properties=properties
        )
    
    def bpe_tokenization_demo(self, text: str, max_merges: int = 5) -> TokenizationResult:
        """
        Demonstrate Byte Pair Encoding (BPE) tokenization process.
        
        Args:
            text: Input text to tokenize
            max_merges: Maximum number of merge operations to perform
            
        Returns:
            TokenizationResult with BPE breakdown
        """
        computation_steps = []
        
        # Step 1: Initialize with character-level tokens
        words = text.lower().split()
        char_tokens = []
        for word in words:
            char_tokens.extend(list(word) + ['</w>'])  # End of word marker
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="character_initialization",
            input_values={'text': np.array([text])},
            operation_description="Initialize with character-level tokens",
            output_values={'char_tokens': np.array(char_tokens)},
            visualization_hints={'operation': 'char_tokenization'}
        )
        computation_steps.append(step_1)
        
        # Step 2: Count character pairs
        current_tokens = char_tokens.copy()
        merge_history = []
        
        for merge_step in range(max_merges):
            # Count adjacent pairs
            pair_counts = {}
            for i in range(len(current_tokens) - 1):
                pair = (current_tokens[i], current_tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            if not pair_counts:
                break
            
            # Find most frequent pair
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            
            # Merge the most frequent pair
            new_tokens = []
            i = 0
            while i < len(current_tokens):
                if (i < len(current_tokens) - 1 and 
                    current_tokens[i] == most_frequent_pair[0] and 
                    current_tokens[i + 1] == most_frequent_pair[1]):
                    new_tokens.append(most_frequent_pair[0] + most_frequent_pair[1])
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
            
            merge_history.append({
                'step': merge_step + 1,
                'merged_pair': most_frequent_pair,
                'frequency': pair_counts[most_frequent_pair],
                'tokens_before': len(current_tokens),
                'tokens_after': len(new_tokens)
            })
            
            current_tokens = new_tokens
        
        step_2 = ComputationStep(
            step_number=2,
            operation_name="bpe_merging",
            input_values={'char_tokens': np.array(char_tokens)},
            operation_description=f"Perform {len(merge_history)} BPE merge operations",
            output_values={'final_tokens': np.array(current_tokens)},
            visualization_hints={'operation': 'bpe_merge', 'merge_history': merge_history}
        )
        computation_steps.append(step_2)
        
        # Create vocabulary and token IDs
        vocabulary = {token: idx for idx, token in enumerate(set(current_tokens))}
        token_ids = np.array([vocabulary[token] for token in current_tokens])
        
        # Create subword mapping
        subword_mapping = {i: [current_tokens[i]] for i in range(len(current_tokens))}
        
        properties = {
            'operation': 'bpe_tokenization',
            'input_text': text,
            'num_tokens': len(current_tokens),
            'vocabulary_size': len(vocabulary),
            'num_merges': len(merge_history),
            'compression_ratio': len(char_tokens) / len(current_tokens),
            'merge_history': merge_history
        }
        
        return TokenizationResult(
            tokens=current_tokens,
            token_ids=token_ids,
            subword_mapping=subword_mapping,
            computation_steps=computation_steps,
            properties=properties
        )
    
    def embedding_layer(self, token_ids: np.ndarray, vocab_size: int, d_model: int,
                       max_seq_len: int, use_positional: bool = True) -> EmbeddingResult:
        """
        Compute embedding layer with token and positional embeddings.
        
        Args:
            token_ids: Token IDs (seq_len,)
            vocab_size: Vocabulary size
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            use_positional: Whether to add positional embeddings
            
        Returns:
            EmbeddingResult with detailed computation breakdown
        """
        seq_len = len(token_ids)
        computation_steps = []
        
        # Initialize embedding matrices (normally learned parameters)
        token_embedding_matrix = np.random.randn(vocab_size, d_model) * 0.1
        
        # Step 1: Token embedding lookup
        token_embeddings = token_embedding_matrix[token_ids]  # (seq_len, d_model)
        
        step_1 = ComputationStep(
            step_number=1,
            operation_name="token_embedding_lookup",
            input_values={'token_ids': token_ids, 'embedding_matrix': token_embedding_matrix},
            operation_description=f"Look up token embeddings from {vocab_size}×{d_model} matrix",
            output_values={'token_embeddings': token_embeddings},
            visualization_hints={'operation': 'embedding_lookup'}
        )
        computation_steps.append(step_1)
        
        if use_positional:
            # Step 2: Generate positional embeddings
            positional_embeddings = self._generate_positional_embeddings(seq_len, d_model)
            
            step_2 = ComputationStep(
                step_number=2,
                operation_name="positional_embedding_generation",
                input_values={'seq_len': np.array([seq_len]), 'd_model': np.array([d_model])},
                operation_description="Generate sinusoidal positional embeddings",
                output_values={'positional_embeddings': positional_embeddings},
                visualization_hints={'operation': 'positional_generation'}
            )
            computation_steps.append(step_2)
            
            # Step 3: Combine token and positional embeddings
            combined_embeddings = token_embeddings + positional_embeddings
            
            step_3 = ComputationStep(
                step_number=3,
                operation_name="embedding_combination",
                input_values={'token_embeddings': token_embeddings, 
                            'positional_embeddings': positional_embeddings},
                operation_description="Add token and positional embeddings",
                output_values={'combined_embeddings': combined_embeddings},
                visualization_hints={'operation': 'embedding_addition'}
            )
            computation_steps.append(step_3)
        else:
            positional_embeddings = np.zeros_like(token_embeddings)
            combined_embeddings = token_embeddings
        
        # Create visualization
        visualization = self._create_embedding_visualization(
            token_ids, token_embeddings, positional_embeddings, combined_embeddings
        )
        
        # Calculate properties
        properties = {
            'operation': 'embedding_layer',
            'vocab_size': vocab_size,
            'd_model': d_model,
            'seq_len': seq_len,
            'use_positional': use_positional,
            'parameter_count': vocab_size * d_model,
            'embedding_norm': np.linalg.norm(combined_embeddings),
            'token_embedding_stats': {
                'mean': np.mean(token_embeddings),
                'std': np.std(token_embeddings)
            }
        }
        
        return EmbeddingResult(
            embeddings=combined_embeddings,
            token_embeddings=token_embeddings,
            positional_embeddings=positional_embeddings,
            computation_steps=computation_steps,
            visualization=visualization,
            properties=properties
        )
    
    def _generate_positional_embeddings(self, seq_len: int, d_model: int) -> np.ndarray:
        """Generate sinusoidal positional embeddings."""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def _create_embedding_visualization(self, token_ids: np.ndarray, token_emb: np.ndarray,
                                      pos_emb: np.ndarray, combined_emb: np.ndarray) -> OperationVisualization:
        """Create visualization for embedding layer."""
        
        # Token IDs as input
        token_colored = ColorCodedMatrix(
            matrix_data=token_ids.reshape(-1, 1),
            color_mapping={'default': self.color_palette['tokens']},
            element_labels={},
            highlight_patterns=[]
        )
        
        # Token embeddings
        token_emb_colored = ColorCodedMatrix(
            matrix_data=token_emb,
            color_mapping={'default': self.color_palette['embeddings']},
            element_labels={},
            highlight_patterns=[]
        )
        
        # Positional embeddings
        pos_emb_colored = ColorCodedMatrix(
            matrix_data=pos_emb,
            color_mapping={'default': self.color_palette['positional']},
            element_labels={},
            highlight_patterns=[]
        )
        
        # Combined embeddings
        combined_colored = ColorCodedMatrix(
            matrix_data=combined_emb,
            color_mapping={'default': self.color_palette['combined']},
            element_labels={},
            highlight_patterns=[]
        )
        
        return OperationVisualization(
            operation_type="embedding_layer",
            input_matrices=[token_colored],
            intermediate_steps=[token_emb_colored, pos_emb_colored],
            output_matrix=combined_colored,
            animation_sequence=[]
        )