"""
Configuration settings for the AI Math Tutorial system.
"""

from dataclasses import dataclass
from typing import Dict, List
import os


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    default_figure_size: tuple = (10, 8)
    default_dpi: int = 100
    color_palette: str = "viridis"
    font_size: int = 12
    animation_fps: int = 30
    max_matrix_display_size: int = 20


@dataclass
class ComputationConfig:
    """Configuration for computational components."""
    numerical_precision: str = "float64"
    tolerance: float = 1e-10
    max_iterations: int = 10000
    random_seed: int = 42
    use_gpu: bool = False


@dataclass
class ContentConfig:
    """Configuration for content generation."""
    max_file_lines: int = 200
    latex_engine: str = "mathjax"
    output_format: str = "html"
    include_source_code: bool = True
    difficulty_levels: List[str] = None
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = [
                "Beginner", "Intermediate", "Advanced", 
                "Expert", "Research"
            ]


@dataclass
class SystemConfig:
    """Main system configuration."""
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = "data"
    output_dir: str = "output"
    cache_dir: str = "cache"
    log_level: str = "INFO"
    
    # Component configurations
    visualization: VisualizationConfig = None
    computation: ComputationConfig = None
    content: ContentConfig = None
    
    def __post_init__(self):
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.computation is None:
            self.computation = ComputationConfig()
        if self.content is None:
            self.content = ContentConfig()


# Global configuration instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration parameters."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")


# Color schemes for consistent visualization
COLOR_SCHEMES = {
    "default": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "accent": "#2ca02c",
        "highlight": "#d62728",
        "background": "#f8f9fa",
        "text": "#212529"
    },
    "matrix": {
        "positive": "#2ca02c",
        "negative": "#d62728", 
        "zero": "#ffffff",
        "attention": "#ff7f0e",
        "gradient": ["#440154", "#31688e", "#35b779", "#fde725"]
    },
    "operation": {
        "input": "#1f77b4",
        "intermediate": "#ff7f0e",
        "output": "#2ca02c",
        "highlight": "#d62728"
    }
}


def get_color_scheme(scheme_name: str = "default") -> Dict[str, str]:
    """Get a color scheme by name."""
    return COLOR_SCHEMES.get(scheme_name, COLOR_SCHEMES["default"])