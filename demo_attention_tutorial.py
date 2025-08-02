"""
Demonstration of the attention mechanism tutorial.
Shows scaled dot-product attention, multi-head attention, and positional encodings.
"""

import numpy as np
from src.content.attention.attention_tutorial import AttentionTutorial


def demonstrate_attention_tutorial():
    """Demonstrate the complete attention mechanism tutorial."""
    print("=" * 80)
    print("ATTENTION MECHANISM TUTORIAL DEMONSTRATION")
    print("=" * 80)
    
    # Create tutorial
    tutorial = AttentionTutorial()
    result = tutorial.create_complete_tutorial()
    
    print(f"\nTutorial: {result.concept.title}")
    print(f"Difficulty Level: {result.concept.difficulty_level}/5")
    print(f"Prerequisites: {', '.join(result.concept.prerequisites)}")
    
    # Show equations
    print("\n" + "=" * 60)
    print("MATHEMATICAL EQUATIONS")
    print("=" * 60)
    
    for i, eq in enumerate(result.concept.equations, 1):
        print(f"\n{i}. {eq.latex_expression}")
        print("   Variables:")
        for var, desc in eq.variables.items():
            print(f"     {var}: {desc}")
        print("   Properties:")
        for prop in eq.mathematical_properties:
            print(f"     • {prop}")
    
    # Demonstrate attention examples
    print("\n" + "=" * 60)
    print("ATTENTION MECHANISM EXAMPLES")
    print("=" * 60)
    
    for i, example in enumerate(result.attention_examples, 1):
        print(f"\nExample {i}: {example.properties['operation']}")
        print(f"Attention weights shape: {example.attention_weights.shape}")
        print(f"Output shape: {example.output.shape}")
        
        # Show key properties
        props = example.properties
        if 'attention_entropy' in props:
            print(f"Attention entropy: {props['attention_entropy']:.3f}")
        if 'max_attention_weight' in props:
            print(f"Max attention weight: {props['max_attention_weight']:.3f}")
        if 'num_heads' in props:
            print(f"Number of heads: {props['num_heads']}")
        
        # Show computation steps
        print("Computation steps:")
        for step in example.computation_steps:
            print(f"  {step.step_number}. {step.operation_description}")
    
    # Demonstrate positional encoding examples
    print("\n" + "=" * 60)
    print("POSITIONAL ENCODING EXAMPLES")
    print("=" * 60)
    
    for i, example in enumerate(result.positional_examples, 1):
        print(f"\nExample {i}: {example.properties['encoding_type']}")
        print(f"Encoding shape: {example.encoded_positions.shape}")
        
        # Show key properties
        props = example.properties
        if 'sequence_length' in props:
            print(f"Sequence length: {props['sequence_length']}")
        if 'd_model' in props:
            print(f"Model dimension: {props['d_model']}")
        if 'd_head' in props:
            print(f"Head dimension: {props['d_head']}")
        if 'frequency_range' in props:
            freq_min, freq_max = props['frequency_range']
            print(f"Frequency range: [{freq_min:.6f}, {freq_max:.6f}]")
        
        # Show computation steps
        print("Computation steps:")
        for step in example.computation_steps:
            print(f"  {step.step_number}. {step.operation_description}")
    
    # Show tutorial sections
    print("\n" + "=" * 60)
    print("TUTORIAL STRUCTURE")
    print("=" * 60)
    
    for i, section in enumerate(result.tutorial_sections, 1):
        print(f"\n{i}. {section['title']}")
        if 'learning_objectives' in section:
            print("   Learning Objectives:")
            for obj in section['learning_objectives']:
                print(f"     • {obj}")
        if 'key_insights' in section:
            print("   Key Insights:")
            for insight in section['key_insights']:
                print(f"     • {insight}")
        if 'subsections' in section:
            print("   Subsections:")
            for subsec in section['subsections']:
                print(f"     - {subsec['title']}")
    
    # Demonstrate specific attention patterns
    print("\n" + "=" * 60)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 60)
    
    # Analyze the first attention example
    attention_example = result.attention_examples[0]
    attention_weights = attention_example.attention_weights
    
    print(f"\nAttention Weight Matrix ({attention_weights.shape}):")
    print(attention_weights)
    
    # Show which positions attend to which
    print("\nAttention Focus Analysis:")
    for i in range(attention_weights.shape[0]):
        max_attention_pos = np.argmax(attention_weights[i])
        max_attention_val = attention_weights[i, max_attention_pos]
        print(f"Position {i} attends most to position {max_attention_pos} (weight: {max_attention_val:.3f})")
    
    # Analyze positional encoding patterns
    print("\n" + "=" * 60)
    print("POSITIONAL ENCODING ANALYSIS")
    print("=" * 60)
    
    # Analyze absolute positional encoding
    abs_pe_example = result.positional_examples[0]
    pe_matrix = abs_pe_example.encoded_positions
    
    print(f"\nAbsolute Positional Encoding Matrix ({pe_matrix.shape}):")
    print("First few positions and dimensions:")
    print(pe_matrix[:4, :8])  # Show first 4 positions, 8 dimensions
    
    # Show frequency analysis
    print("\nFrequency Analysis:")
    print(f"Encoding range: [{np.min(pe_matrix):.3f}, {np.max(pe_matrix):.3f}]")
    print(f"Mean absolute value: {np.mean(np.abs(pe_matrix)):.3f}")
    
    # Analyze RoPE
    rope_example = result.positional_examples[1]
    rope_matrix = rope_example.encoded_positions
    
    print(f"\nRoPE Encoding Matrix ({rope_matrix.shape}):")
    print("Rotation magnitude by position:")
    # Find the input step that contains the original x values
    original_x = None
    for step in rope_example.computation_steps:
        if 'x' in step.input_values:
            original_x = step.input_values['x']
            break
    
    if original_x is not None:
        for i in range(min(4, rope_matrix.shape[0])):
            rotation_mag = np.linalg.norm(rope_matrix[i] - original_x[i])
            print(f"Position {i}: rotation magnitude = {rotation_mag:.3f}")
    else:
        print("Original input not found in computation steps")
    
    print("\n" + "=" * 80)
    print("TUTORIAL DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_attention_tutorial()