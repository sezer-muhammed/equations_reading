"""
Content integration system for combining equations, explanations, and visualizations.
Handles the assembly of mathematical concepts with proper formatting and cross-references.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from src.core.models import (
    MathematicalConcept, Equation, Explanation, Visualization, 
    NumericalExample, TutorialChapter
)
from src.rendering.content_assembly.template_system import TemplateSystem


@dataclass
class ContentSection:
    """Represents a section of integrated content."""
    section_id: str
    section_type: str  # "equation", "explanation", "example", "visualization"
    content_html: str
    dependencies: List[str]
    cross_references: List[str]


@dataclass
class IntegrationRule:
    """Rule for integrating different content types."""
    rule_id: str
    content_types: List[str]
    ordering_priority: int
    integration_method: str  # "sequential", "interleaved", "grouped"


class ContentIntegrator:
    """Integrates equations, explanations, and visualizations into cohesive content."""
    
    def __init__(self, template_system: TemplateSystem):
        self.template_system = template_system
        self.integration_rules: Dict[str, IntegrationRule] = {}
        self.content_cache: Dict[str, str] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default integration rules."""
        
        # Rule for mathematical concepts: explanation -> equation -> example -> visualization
        concept_rule = IntegrationRule(
            rule_id="mathematical_concept_standard",
            content_types=["explanation", "equation", "numerical_example", "visualization"],
            ordering_priority=1,
            integration_method="sequential"
        )
        self.integration_rules["mathematical_concept"] = concept_rule
        
        # Rule for equation-focused content: equation -> explanation -> example
        equation_focused_rule = IntegrationRule(
            rule_id="equation_focused",
            content_types=["equation", "explanation", "numerical_example"],
            ordering_priority=2,
            integration_method="interleaved"
        )
        self.integration_rules["equation_focused"] = equation_focused_rule
    
    def integrate_concept(self, concept: MathematicalConcept, 
                         integration_style: str = "mathematical_concept") -> str:
        """Integrate all components of a mathematical concept."""
        
        # Check cache first
        cache_key = f"concept_{concept.concept_id}_{integration_style}"
        if cache_key in self.content_cache:
            return self.content_cache[cache_key]
        
        # Get integration rule
        rule = self.integration_rules.get(integration_style)
        if not rule:
            rule = self.integration_rules["mathematical_concept"]
        
        # Prepare rendering functions for the template
        render_functions = {
            'render_explanation': self._render_explanation,
            'render_equation': self._render_equation,
            'render_numerical_example': self._render_numerical_example,
            'render_visualization': self._render_visualization
        }
        
        # Render the concept using template
        context = {
            'concept': concept,
            **render_functions
        }
        
        integrated_content = self.template_system.render_template(
            "mathematical_concept", context
        )
        
        # Cache the result
        self.content_cache[cache_key] = integrated_content
        return integrated_content
    
    def integrate_chapter(self, chapter: TutorialChapter) -> str:
        """Integrate all concepts in a chapter."""
        
        cache_key = f"chapter_{chapter.chapter_id}"
        if cache_key in self.content_cache:
            return self.content_cache[cache_key]
        
        # Prepare rendering function for concepts
        def render_concept(concept):
            return self.integrate_concept(concept)
        
        context = {
            'chapter': chapter,
            'render_concept': render_concept
        }
        
        integrated_content = self.template_system.render_template(
            "tutorial_chapter", context
        )
        
        self.content_cache[cache_key] = integrated_content
        return integrated_content
    
    def _render_explanation(self, explanation: Explanation) -> str:
        """Render an explanation component."""
        context = {'explanation': explanation}
        return self.template_system.render_template("explanation", context)
    
    def _render_equation(self, equation: Equation) -> str:
        """Render an equation component."""
        context = {'equation': equation}
        return self.template_system.render_template("equation", context)
    
    def _render_numerical_example(self, example: NumericalExample) -> str:
        """Render a numerical example component."""
        context = {'example': example}
        return self.template_system.render_template("numerical_example", context)
    
    def _render_visualization(self, visualization: Visualization) -> str:
        """Render a visualization component."""
        context = {'visualization': visualization}
        return self.template_system.render_template("visualization", context)
    
    def extract_cross_references(self, content: str) -> List[str]:
        """Extract cross-references from rendered content."""
        import re
        
        # Find all links to concepts, equations, examples, and visualizations
        concept_refs = re.findall(r'href="#concept-([^"]+)"', content)
        equation_refs = re.findall(r'href="#eq-([^"]+)"', content)
        example_refs = re.findall(r'href="#example-([^"]+)"', content)
        viz_refs = re.findall(r'href="#viz-([^"]+)"', content)
        
        all_refs = concept_refs + equation_refs + example_refs + viz_refs
        return list(set(all_refs))  # Remove duplicates
    
    def validate_integration(self, concept: MathematicalConcept) -> Tuple[bool, List[str]]:
        """Validate that a concept can be properly integrated."""
        errors = []
        
        # Check that all equations have proper variable definitions
        for equation in concept.equations:
            if not equation.variables:
                errors.append(f"Equation {equation.equation_id} has no variable definitions")
        
        # Check that examples reference valid equations
        for example in concept.examples:
            if not example.computation_steps:
                errors.append(f"Example {example.example_id} has no computation steps")
        
        # Check that visualizations have proper data
        for viz in concept.visualizations:
            if not viz.data:
                errors.append(f"Visualization {viz.visualization_id} has no data")
        
        return len(errors) == 0, errors
    
    def generate_content_outline(self, concept: MathematicalConcept) -> Dict[str, Any]:
        """Generate an outline of how content will be integrated."""
        outline = {
            'concept_id': concept.concept_id,
            'title': concept.title,
            'sections': [],
            'cross_references': [],
            'estimated_length': 0
        }
        
        # Add sections based on available content
        if concept.explanations:
            outline['sections'].append({
                'type': 'explanations',
                'count': len(concept.explanations),
                'items': [exp.explanation_type for exp in concept.explanations]
            })
        
        if concept.equations:
            outline['sections'].append({
                'type': 'equations',
                'count': len(concept.equations),
                'items': [eq.equation_id for eq in concept.equations]
            })
        
        if concept.examples:
            outline['sections'].append({
                'type': 'examples',
                'count': len(concept.examples),
                'items': [ex.example_id for ex in concept.examples]
            })
        
        if concept.visualizations:
            outline['sections'].append({
                'type': 'visualizations',
                'count': len(concept.visualizations),
                'items': [viz.visualization_id for viz in concept.visualizations]
            })
        
        # Estimate content length (rough approximation)
        outline['estimated_length'] = (
            len(concept.explanations) * 200 +  # ~200 words per explanation
            len(concept.equations) * 100 +     # ~100 words per equation
            len(concept.examples) * 150 +      # ~150 words per example
            len(concept.visualizations) * 50   # ~50 words per visualization
        )
        
        return outline
    
    def clear_cache(self):
        """Clear the content cache."""
        self.content_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the content cache."""
        return {
            'cached_items': len(self.content_cache),
            'total_size_chars': sum(len(content) for content in self.content_cache.values()),
            'cache_keys': list(self.content_cache.keys())
        }