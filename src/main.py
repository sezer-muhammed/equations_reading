"""
Main entry point for the AI Math Tutorial system.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_config
from core.models import MathematicalConcept


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="AI Math Tutorial - Interactive Educational System"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="AI Math Tutorial 0.1.0"
    )
    parser.add_argument(
        "--config", 
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    if args.verbose:
        print(f"AI Math Tutorial v0.1.0")
        print(f"Project root: {config.project_root}")
        print(f"Configuration loaded successfully")
    
    # Basic system check
    try:
        import numpy as np
        import torch
        import sympy
        import matplotlib.pyplot as plt
        
        if args.verbose:
            print("✓ All required libraries are available")
            print(f"  - NumPy: {np.__version__}")
            print(f"  - PyTorch: {torch.__version__}")
            print(f"  - SymPy: {sympy.__version__}")
            print(f"  - Matplotlib: {plt.matplotlib.__version__}")
        
        # Test basic functionality
        test_concept = MathematicalConcept(
            concept_id="test",
            title="Test Concept",
            prerequisites=[],
            equations=[],
            explanations=[],
            examples=[],
            visualizations=[],
            difficulty_level=1
        )
        
        if args.verbose:
            print("✓ Core data models working correctly")
            print(f"  - Created test concept: {test_concept.title}")
        
        print("AI Math Tutorial system initialized successfully!")
        
    except ImportError as e:
        print(f"Error: Missing required library - {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error during initialization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()