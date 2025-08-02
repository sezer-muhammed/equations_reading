#!/usr/bin/env python3
"""
Demo script for the transformer architecture tutorial.
Demonstrates complete transformer block, layer normalization comparison,
feed-forward networks, and embedding layers.
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.content.architectures.transformer_tutorial import TransformerTutorial


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_subsection_header(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def demonstrate_transformer_tutorial():
    """Demonstrate the complete transformer architecture tutorial."""
    
    print_section_header("TRANSFORMER ARCHITECTURE TUTORIAL DEMONSTRATION")
    
    # Initialize tutorial
    tutorial = TransformerTutorial()
    
    print("Creating comprehensive transformer architecture tutorial...")
    result = tutorial.create_complete_tutorial()
    
    # Display concept overview
    print_subsection_header("Mathematical Concept Overview")
    concept = result.concept
    print(f"Concept ID: {concept.concept_id}")
    print(f"Title: {concept.title}")
    print(f"Prerequisites: {', '.join(concept.prerequisites)}")
    print(f"Difficulty Level: {concept.difficulty_level}/5")
    print(f"Number of Equations: {len(concept.equations)}")
    print(f"Number of Explanations: {len(concept.explanations)}")
    
    # Display core equations
    print_subsection_header("Core Transformer Equations")
    for i, equation in enumerate(concept.equations, 1):
        print(f"\n{i}. {equation.equation_id}:")
        print(f"   LaTeX: {equation.latex_expression}")
        print(f"   Variables: {len(equation.variables)}")
        print(f"   Properties: {len(equation.mathematical_properties)}")
        print(f"   Applications: {len(equation.applications)}")
    
    # Demonstrate transformer block
    print_section_header("TRANSFORMER BLOCK DEMONSTRATION")
    
    for i, transformer_example in enumerate(result.transformer_examples, 1):
        print_subsection_header(f"Transformer Block Example {i}")
        props = transformer_example.properties
        print(f"Input Shape: {props['input_shape']}")
        print(f"Number of Heads: {props['num_heads']}")
        print(f"Feed-forward Dimension: {props['d_ff']}")
        print(f"Total Parameters: {props['total_parameters']:,}")
        print(f"Computation Steps: {len(transformer_example.computation_steps)}")
        
        # Show computation steps
        print("\nComputation Steps:")
        for step in transformer_example.computation_steps:
            print(f"  {step.step_number}. {step.operation_name}: {step.operation_description}")
        
        # Show output statistics
        output = transformer_example.output
        print(f"\nOutput Statistics:")
        print(f"  Shape: {output.shape}")
        print(f"  Mean: {np.mean(output):.6f}")
        print(f"  Std: {np.std(output):.6f}")
        print(f"  Min: {np.min(output):.6f}")
        print(f"  Max: {np.max(output):.6f}")
    
    # Demonstrate layer normalization comparison
    print_section_header("LAYER NORMALIZATION VS BATCH NORMALIZATION")
    
    layer_norm_result = result.layer_norm_comparison['layer_norm']
    batch_norm_result = result.layer_norm_comparison['batch_norm']
    
    print_subsection_header("Layer Normalization")
    ln_props = layer_norm_result.properties
    print(f"Operation: {ln_props['operation']}")
    print(f"Input Shape: {ln_props['input_shape']}")
    print(f"Output Mean: {ln_props['output_mean']:.6f}")
    print(f"Output Std: {ln_props['output_std']:.6f}")
    print(f"Parameter Count: {ln_props['parameter_count']}")
    
    print_subsection_header("Batch Normalization")
    bn_props = batch_norm_result.properties
    print(f"Operation: {bn_props['operation']}")
    print(f"Input Shape: {bn_props['input_shape']}")
    print(f"Output Mean: {bn_props['output_mean']:.6f}")
    print(f"Output Std: {bn_props['output_std']:.6f}")
    print(f"Parameter Count: {bn_props['parameter_count']}")
    
    print_subsection_header("Normalization Comparison")
    print("Key Differences:")
    print("  Layer Norm: Normalizes across features for each sample")
    print("  Batch Norm: Normalizes across batch for each feature")
    print("  Layer Norm: Independent of batch size and other samples")
    print("  Batch Norm: Depends on batch statistics")
    
    # Demonstrate feed-forward networks
    print_section_header("FEED-FORWARD NETWORK DEMONSTRATION")
    
    for i, ff_example in enumerate(result.feedforward_examples, 1):
        print_subsection_header(f"Feed-Forward Example {i}")
        props = ff_example.properties
        print(f"Activation: {props['activation']}")
        print(f"Input Shape: {props['input_shape']}")
        print(f"Hidden Dimension: {props['hidden_dimension']}")
        print(f"Expansion Ratio: {props['expansion_ratio']:.1f}x")
        print(f"Parameter Count: {props['parameter_count']:,}")
        if 'sparsity' in props:
            print(f"Activation Sparsity: {props['sparsity']:.3f}")
        
        # Show computation steps
        print("\nComputation Steps:")
        for step in ff_example.computation_steps:
            print(f"  {step.step_number}. {step.operation_name}: {step.operation_description}")
    
    # Demonstrate embedding layers
    print_section_header("EMBEDDING LAYER DEMONSTRATION")
    
    for i, embedding_example in enumerate(result.embedding_examples, 1):
        print_subsection_header(f"Embedding Example {i}")
        props = embedding_example.properties
        print(f"Vocabulary Size: {props['vocab_size']}")
        print(f"Model Dimension: {props['d_model']}")
        print(f"Sequence Length: {props['seq_len']}")
        print(f"Use Positional: {props['use_positional']}")
        print(f"Parameter Count: {props['parameter_count']:,}")
        print(f"Embedding Norm: {props['embedding_norm']:.6f}")
        
        # Show token embedding statistics
        token_stats = props['token_embedding_stats']
        print(f"Token Embedding Mean: {token_stats['mean']:.6f}")
        print(f"Token Embedding Std: {token_stats['std']:.6f}")
        
        # Show computation steps
        print("\nComputation Steps:")
        for step in embedding_example.computation_steps:
            print(f"  {step.step_number}. {step.operation_name}: {step.operation_description}")
    
    # Demonstrate tokenization
    print_section_header("TOKENIZATION DEMONSTRATION")
    
    for i, tokenization_example in enumerate(result.tokenization_examples, 1):
        print_subsection_header(f"Tokenization Example {i}")
        props = tokenization_example.properties
        print(f"Operation: {props['operation']}")
        print(f"Input Text: '{props['input_text']}'")
        print(f"Number of Tokens: {props['num_tokens']}")
        print(f"Vocabulary Size: {props['vocabulary_size']}")
        
        if 'compression_ratio' in props:
            print(f"Compression Ratio: {props['compression_ratio']:.2f}")
        if 'merge_history' in props:
            print(f"Number of Merges: {len(props['merge_history'])}")
        
        print(f"Tokens: {tokenization_example.tokens}")
        print(f"Token IDs: {tokenization_example.token_ids}")
        
        # Show BPE merge history if available
        if 'merge_history' in props and props['merge_history']:
            print("\nBPE Merge History:")
            for merge in props['merge_history']:
                print(f"  Step {merge['step']}: Merged {merge['merged_pair']} "
                      f"(freq: {merge['frequency']}, tokens: {merge['tokens_before']} → {merge['tokens_after']})")
    
    # Display tutorial sections overview
    print_section_header("TUTORIAL SECTIONS OVERVIEW")
    
    for i, section in enumerate(result.tutorial_sections, 1):
        print_subsection_header(f"Section {i}: {section['title']}")
        print(f"Content Length: {len(section['content'])} characters")
        
        if 'learning_objectives' in section:
            print(f"Learning Objectives: {len(section['learning_objectives'])}")
        if 'mathematical_derivation' in section:
            print(f"Mathematical Steps: {len(section['mathematical_derivation'])}")
        if 'numerical_example' in section:
            print("Includes numerical example with visualization")
    
    print_section_header("TUTORIAL DEMONSTRATION COMPLETE")
    print("The transformer architecture tutorial includes:")
    print("✓ Complete transformer block implementation")
    print("✓ Layer normalization vs batch normalization comparison")
    print("✓ Feed-forward networks with ReLU and GELU activations")
    print("✓ Embedding layers with positional encoding")
    print("✓ Tokenization processes (word-level and BPE)")
    print("✓ Comprehensive mathematical derivations")
    print("✓ Detailed visualizations and examples")
    print("✓ Graduate-level explanations and insights")


if __name__ == "__main__":
    demonstrate_transformer_tutorial()