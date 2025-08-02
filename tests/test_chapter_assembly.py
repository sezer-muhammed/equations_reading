"""
Tests for the chapter assembly engine.
"""

import pytest
import numpy as np
from src.core.models import (
    MathematicalConcept, Equation, Explanation, Visualization, 
    NumericalExample, TutorialChapter, VariableDefinition,
    ComputationStep, VisualizationData
)
from src.rendering.content_assembly import (
    ChapterAssembler, AssemblyConfiguration, TemplateSystem
)


def create_sample_concept():
    """Create a sample mathematical concept for testing."""
    
    # Create variable definitions
    var_x = VariableDefinition(
        name="x",
        description="Input variable",
        data_type="scalar",
        color_code="#FF6B6B"
    )
    
    var_y = VariableDefinition(
        name="y",
        description="Output variable", 
        data_type="scalar",
        color_code="#4ECDC4"
    )
    
    # Create equation
    equation = Equation(
        equation_id="linear_eq",
        latex_expression="y = mx + b",
        variables={"x": var_x, "y": var_y},
        mathematical_properties=["linear"],
        applications=["regression", "modeling"]
    )
    
    # Create explanation
    explanation = Explanation(
        explanation_type="intuitive",
        content="A linear equation represents a straight line relationship between variables.",
        mathematical_level=2
    )
    
    # Create numerical example
    example = NumericalExample(
        example_id="linear_example_1",
        description="Simple linear relationship",
        input_values={"x": np.array([1, 2, 3])},
        computation_steps=[
            ComputationStep(
                step_number=1,
                operation_name="linear_transform",
                input_values={"x": np.array([1, 2, 3])},
                operation_description="Apply linear transformation y = 2x + 1",
                output_values={"y": np.array([3, 5, 7])}
            )
        ],
        output_values={"y": np.array([3, 5, 7])}
    )
    
    # Create visualization
    visualization = Visualization(
        visualization_id="linear_plot",
        visualization_type="line_plot",
        title="Linear Relationship Visualization",
        description="Plot showing the linear relationship",
        data=VisualizationData(
            visualization_type="line_plot",
            data={"x": [1, 2, 3], "y": [3, 5, 7]},
            color_mappings={"line": "#FF6B6B"}
        )
    )
    
    # Create concept
    concept = MathematicalConcept(
        concept_id="linear_equations",
        title="Linear Equations",
        prerequisites=["basic_algebra"],
        equations=[equation],
        explanations=[explanation],
        examples=[example],
        visualizations=[visualization],
        difficulty_level=2,
        learning_objectives=["Understand linear relationships", "Apply linear equations"]
    )
    
    return concept


def create_sample_chapter():
    """Create a sample chapter for testing."""
    concept = create_sample_concept()
    
    chapter = TutorialChapter(
        chapter_id="intro_linear",
        title="Introduction to Linear Equations",
        concepts=[concept],
        introduction="This chapter introduces linear equations and their applications.",
        summary="We covered the basics of linear equations and their graphical representation.",
        chapter_number=1,
        estimated_time_minutes=45
    )
    
    return chapter


class TestTemplateSystem:
    """Test the template system."""
    
    def test_template_initialization(self):
        """Test that templates are properly initialized."""
        template_system = TemplateSystem()
        
        # Check that default templates exist
        assert "equation" in template_system.templates
        assert "explanation" in template_system.templates
        assert "numerical_example" in template_system.templates
        assert "visualization" in template_system.templates
        assert "mathematical_concept" in template_system.templates
        assert "tutorial_chapter" in template_system.templates
    
    def test_equation_template_rendering(self):
        """Test rendering of equation template."""
        template_system = TemplateSystem()
        concept = create_sample_concept()
        equation = concept.equations[0]
        
        context = {"equation": equation}
        rendered = template_system.render_template("equation", context)
        
        assert "y = mx + b" in rendered
        assert "linear_eq" in rendered
        assert "Input variable" in rendered
        assert "Output variable" in rendered


class TestChapterAssembler:
    """Test the chapter assembler."""
    
    def test_basic_assembly(self):
        """Test basic chapter assembly."""
        config = AssemblyConfiguration(validate_links=False)  # Disable link validation for test
        assembler = ChapterAssembler(config)
        chapter = create_sample_chapter()
        
        result = assembler.assemble_chapter(chapter)
        
        assert result.success
        assert len(result.chapter_html) > 0
        assert "Linear Equations" in result.chapter_html
        assert "y = mx + b" in result.chapter_html
    
    def test_assembly_with_navigation(self):
        """Test assembly with navigation enabled."""
        config = AssemblyConfiguration(
            include_navigation=True,
            generate_toc=True,
            validate_links=False  # Disable link validation for test
        )
        assembler = ChapterAssembler(config)
        chapter = create_sample_chapter()
        
        result = assembler.assemble_chapter(chapter)
        
        assert result.success
        assert "table-of-contents" in result.chapter_html
        assert "chapter-navigation" in result.chapter_html
    
    def test_assembly_metadata(self):
        """Test that assembly generates proper metadata."""
        config = AssemblyConfiguration(validate_links=False)  # Disable link validation for test
        assembler = ChapterAssembler(config)
        chapter = create_sample_chapter()
        
        result = assembler.assemble_chapter(chapter)
        
        assert result.success
        assert "content_stats" in result.metadata
        assert result.metadata["content_stats"]["total_concepts"] == 1
        assert result.metadata["content_stats"]["total_equations"] == 1
        assert result.metadata["content_stats"]["total_examples"] == 1
        assert result.metadata["content_stats"]["total_visualizations"] == 1
    
    def test_multiple_chapter_assembly(self):
        """Test assembling multiple chapters."""
        config = AssemblyConfiguration(validate_links=False)  # Disable link validation for test
        assembler = ChapterAssembler(config)
        
        # Create two chapters
        chapter1 = create_sample_chapter()
        chapter2 = create_sample_chapter()
        chapter2.chapter_id = "advanced_linear"
        chapter2.title = "Advanced Linear Equations"
        chapter2.chapter_number = 2
        
        results = assembler.assemble_multiple_chapters([chapter1, chapter2])
        
        assert len(results) == 2
        assert all(result.success for result in results)
    
    def test_assembly_caching(self):
        """Test that assembly results are properly cached."""
        config = AssemblyConfiguration(cache_content=True, validate_links=False)  # Disable link validation for test
        assembler = ChapterAssembler(config)
        chapter = create_sample_chapter()
        
        # First assembly
        result1 = assembler.assemble_chapter(chapter)
        assert result1.success
        
        # Second assembly should use cache
        result2 = assembler.assemble_chapter(chapter)
        assert result2.success
        assert result1.chapter_html == result2.chapter_html
        
        # Check cache stats
        assert len(assembler.assembly_cache) == 1


if __name__ == "__main__":
    # Run a simple test
    concept = create_sample_concept()
    chapter = create_sample_chapter()
    
    assembler = ChapterAssembler()
    result = assembler.assemble_chapter(chapter)
    
    print(f"Assembly successful: {result.success}")
    print(f"Content length: {len(result.chapter_html)} characters")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")