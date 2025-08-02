"""
Content template system for consistent formatting across tutorial chapters.
Provides templates for equations, explanations, visualizations, and complete chapters.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from jinja2 import Template, Environment, BaseLoader
from src.core.models import (
    MathematicalConcept, Equation, Explanation, Visualization, 
    NumericalExample, TutorialChapter
)


@dataclass
class ContentTemplate:
    """Base template for content rendering."""
    template_id: str
    template_content: str
    required_variables: List[str]
    optional_variables: List[str] = None
    
    def __post_init__(self):
        if self.optional_variables is None:
            self.optional_variables = []


class TemplateSystem:
    """Manages content templates for consistent formatting."""
    
    def __init__(self):
        self.templates: Dict[str, ContentTemplate] = {}
        self.jinja_env = Environment(loader=BaseLoader())
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default templates for common content types."""
        
        # Equation template
        equation_template = ContentTemplate(
            template_id="equation",
            template_content="""
<div class="equation-container" id="eq-{{ equation.equation_id }}">
    <div class="equation-header">
        <h3>{{ equation.equation_id }}</h3>
    </div>
    <div class="equation-content">
        <div class="latex-equation">
            $${{ equation.latex_expression }}$$
        </div>
        {% if equation.variables %}
        <div class="variable-definitions">
            <h4>Variables:</h4>
            <ul>
            {% for var_name, var_def in equation.variables.items() %}
                <li>
                    <span class="variable-name" style="color: {{ var_def.color_code }}">
                        {{ var_name }}
                    </span>: {{ var_def.description }}
                    {% if var_def.constraints %}
                        <em>({{ var_def.constraints }})</em>
                    {% endif %}
                </li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
        {% if equation.mathematical_properties %}
        <div class="mathematical-properties">
            <h4>Mathematical Properties:</h4>
            <ul>
            {% for prop in equation.mathematical_properties %}
                <li>{{ prop }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
</div>
            """,
            required_variables=["equation"]
        )
        self.templates["equation"] = equation_template
        
        # Explanation template
        explanation_template = ContentTemplate(
            template_id="explanation",
            template_content="""
<div class="explanation-container explanation-{{ explanation.explanation_type }}">
    <div class="explanation-content">
        {{ explanation.content }}
    </div>
    {% if explanation.prerequisites %}
    <div class="prerequisites">
        <strong>Prerequisites:</strong>
        {% for prereq in explanation.prerequisites %}
            <a href="#concept-{{ prereq }}" class="prerequisite-link">{{ prereq }}</a>
            {% if not loop.last %}, {% endif %}
        {% endfor %}
    </div>
    {% endif %}
</div>
            """,
            required_variables=["explanation"]
        )
        self.templates["explanation"] = explanation_template
        
        # Numerical example template
        example_template = ContentTemplate(
            template_id="numerical_example",
            template_content="""
<div class="numerical-example" id="example-{{ example.example_id }}">
    <div class="example-header">
        <h4>Example: {{ example.description }}</h4>
    </div>
    <div class="example-content">
        {% if example.input_values %}
        <div class="input-values">
            <h5>Input Values:</h5>
            <ul>
            {% for var_name, value in example.input_values.items() %}
                <li><strong>{{ var_name }}</strong>: {{ value }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        {% if example.computation_steps %}
        <div class="computation-steps">
            <h5>Computation Steps:</h5>
            <ol>
            {% for step in example.computation_steps %}
                <li>
                    <strong>{{ step.operation_name }}</strong>: {{ step.operation_description }}
                    {% if step.output_values %}
                    <div class="step-output">
                        Result: 
                        {% for var_name, value in step.output_values.items() %}
                            {{ var_name }} = {{ value }}
                            {% if not loop.last %}, {% endif %}
                        {% endfor %}
                    </div>
                    {% endif %}
                </li>
            {% endfor %}
            </ol>
        </div>
        {% endif %}
        
        {% if example.educational_notes %}
        <div class="educational-notes">
            <h5>Notes:</h5>
            <ul>
            {% for note in example.educational_notes %}
                <li>{{ note }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
</div>
            """,
            required_variables=["example"]
        )
        self.templates["numerical_example"] = example_template
        
        # Visualization template
        visualization_template = ContentTemplate(
            template_id="visualization",
            template_content="""
<div class="visualization-container" id="viz-{{ visualization.visualization_id }}">
    <div class="visualization-header">
        <h4>{{ visualization.title }}</h4>
        <p class="visualization-description">{{ visualization.description }}</p>
    </div>
    <div class="visualization-content">
        <div id="viz-{{ visualization.visualization_id }}-content" 
             class="visualization-display"
             data-viz-type="{{ visualization.visualization_type }}"
             data-interactive="{{ visualization.interactive }}">
            <!-- Visualization content will be injected here -->
        </div>
    </div>
    {% if visualization.prerequisites %}
    <div class="visualization-prerequisites">
        <small>Prerequisites: 
        {% for prereq in visualization.prerequisites %}
            <a href="#concept-{{ prereq }}">{{ prereq }}</a>
            {% if not loop.last %}, {% endif %}
        {% endfor %}
        </small>
    </div>
    {% endif %}
</div>
            """,
            required_variables=["visualization"]
        )
        self.templates["visualization"] = visualization_template
        
        # Mathematical concept template
        concept_template = ContentTemplate(
            template_id="mathematical_concept",
            template_content="""
<section class="mathematical-concept" id="concept-{{ concept.concept_id }}">
    <header class="concept-header">
        <h2>{{ concept.title }}</h2>
        <div class="concept-metadata">
            <span class="difficulty-level">Difficulty: {{ concept.difficulty_level }}/5</span>
            {% if concept.prerequisites %}
            <div class="concept-prerequisites">
                <strong>Prerequisites:</strong>
                {% for prereq in concept.prerequisites %}
                    <a href="#concept-{{ prereq }}" class="prerequisite-link">{{ prereq }}</a>
                    {% if not loop.last %}, {% endif %}
                {% endfor %}
            </div>
            {% endif %}
        </div>
        {% if concept.learning_objectives %}
        <div class="learning-objectives">
            <h3>Learning Objectives</h3>
            <ul>
            {% for objective in concept.learning_objectives %}
                <li>{{ objective }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
    </header>
    
    <div class="concept-content">
        {% for explanation in concept.explanations %}
            {{ render_explanation(explanation) }}
        {% endfor %}
        
        {% for equation in concept.equations %}
            {{ render_equation(equation) }}
        {% endfor %}
        
        {% for example in concept.examples %}
            {{ render_numerical_example(example) }}
        {% endfor %}
        
        {% for visualization in concept.visualizations %}
            {{ render_visualization(visualization) }}
        {% endfor %}
    </div>
</section>
            """,
            required_variables=["concept"],
            optional_variables=["render_explanation", "render_equation", 
                              "render_numerical_example", "render_visualization"]
        )
        self.templates["mathematical_concept"] = concept_template
        
        # Chapter template
        chapter_template = ContentTemplate(
            template_id="tutorial_chapter",
            template_content="""
<article class="tutorial-chapter" id="chapter-{{ chapter.chapter_number }}">
    <header class="chapter-header">
        <h1>Chapter {{ chapter.chapter_number }}: {{ chapter.title }}</h1>
        <div class="chapter-metadata">
            <span class="estimated-time">Estimated time: {{ chapter.estimated_time_minutes }} minutes</span>
        </div>
    </header>
    
    <div class="chapter-introduction">
        {{ chapter.introduction }}
    </div>
    
    <div class="chapter-content">
        {% for concept in chapter.concepts %}
            {{ render_concept(concept) }}
        {% endfor %}
    </div>
    
    <div class="chapter-summary">
        <h2>Chapter Summary</h2>
        {{ chapter.summary }}
    </div>
</article>
            """,
            required_variables=["chapter"],
            optional_variables=["render_concept"]
        )
        self.templates["tutorial_chapter"] = chapter_template
    
    def register_template(self, template: ContentTemplate):
        """Register a new template."""
        self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[ContentTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def render_template(self, template_id: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context."""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")
        
        # Validate required variables
        missing_vars = [var for var in template.required_variables if var not in context]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Create Jinja2 template and render
        jinja_template = self.jinja_env.from_string(template.template_content)
        return jinja_template.render(**context)
    
    def list_templates(self) -> List[str]:
        """List all available template IDs."""
        return list(self.templates.keys())