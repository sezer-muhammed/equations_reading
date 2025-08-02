# AI Math Tutorial

An interactive educational system for engineering graduates to master the mathematical foundations of artificial intelligence through visual learning.

## Project Structure

```
ai-math-tutorial/
├── src/
│   ├── content/                  # Mathematical content organization
│   │   ├── foundations/          # Linear algebra, probability, optimization
│   │   ├── neural_networks/      # Feedforward networks, regularization
│   │   ├── architectures/        # CNN, RNN architectures
│   │   ├── attention/            # Attention mechanisms, transformers
│   │   └── advanced/             # Generative models, RL, meta-learning
│   ├── computation/              # Computational backends
│   │   ├── matrix_ops/           # Linear algebra computations
│   │   ├── optimization/         # Gradient descent, Adam optimizer
│   │   ├── attention_ops/        # Attention mechanism calculations
│   │   └── generative_ops/       # VAE, GAN, diffusion computations
│   ├── visualization/            # Visualization components
│   │   ├── matrix_viz/           # Color-coded matrix visualizations
│   │   ├── operation_viz/        # Step-by-step operation animations
│   │   ├── graph_viz/            # Function plots and landscapes
│   │   └── interactive/          # Interactive widgets and controls
│   ├── rendering/                # Content assembly and export
│   │   ├── content_assembly/     # Chapter and section assembly
│   │   ├── cross_reference/      # Link management and validation
│   │   └── export/               # HTML, PDF generation
│   ├── core/                     # Core system components
│   │   ├── models.py             # Data models for mathematical concepts
│   │   ├── config.py             # System configuration
│   │   └── utils.py              # Utility functions
│   └── main.py                   # Main entry point
├── tests/                        # Test suite
│   ├── test_core_models.py       # Tests for core data models
│   └── __init__.py
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

Run the main application:
```bash
python src/main.py --verbose
```

Or use the installed console script:
```bash
ai-math-tutorial --verbose
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Core Features

### Mathematical Content Engine
- Equation definitions with LaTeX expressions
- Variable definitions with type information
- Step-by-step derivations
- Prerequisite tracking

### Computational Backend
- Numerical examples with realistic parameters
- Automatic validation against mathematical properties
- Support for NumPy, PyTorch, and SymPy

### Visualization System
- Color-coded matrix representations
- Step-by-step operation animations
- Interactive parameter manipulation
- Consistent visual themes

### Content Rendering
- Chapter assembly with cross-references
- Multiple export formats (HTML, PDF)
- Progressive learning paths
- Accessibility features

## Development Guidelines

- **File Size Limit**: Keep all files under 200 lines for maintainability
- **Modular Design**: Use hierarchical folder structure instead of monolithic files
- **Mathematical Accuracy**: All computations must be verified against established libraries
- **Visual Consistency**: Use consistent color schemes across related concepts

## Requirements Coverage

This implementation addresses the following requirements:
- **6.1**: Python libraries (NumPy, PyTorch, SymPy) for computational verification
- **6.3**: Automatic validation of mathematical properties and results

## Next Steps

The project structure is now ready for implementing specific mathematical content and visualization components. Each module can be developed independently while maintaining the overall system architecture.