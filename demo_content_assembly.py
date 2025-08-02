"""
Demonstration of the content rendering and assembly system.
Shows how to create chapters, assemble content, and generate table of contents.
"""

import numpy as np
from src.core.models import (
    MathematicalConcept, Equation, Explanation, Visualization, 
    NumericalExample, TutorialChapter, VariableDefinition,
    ComputationStep, VisualizationData
)
from src.rendering.content_assembly import (
    ChapterAssembler, AssemblyConfiguration, TableOfContentsGenerator, MasteryLevel
)
from src.rendering.cross_reference import LinkManager


def create_demo_content():
    """Create demonstration content for the assembly system."""
    
    # Create variable definitions
    var_x = VariableDefinition(
        name="x",
        description="Input variable representing the independent variable",
        data_type="scalar",
        color_code="#FF6B6B"
    )
    
    var_y = VariableDefinition(
        name="y", 
        description="Output variable representing the dependent variable",
        data_type="scalar",
        color_code="#4ECDC4"
    )
    
    var_m = VariableDefinition(
        name="m",
        description="Slope parameter determining the rate of change",
        data_type="scalar",
        color_code="#45B7D1"
    )
    
    var_b = VariableDefinition(
        name="b",
        description="Y-intercept parameter representing the baseline value",
        data_type="scalar", 
        color_code="#96CEB4"
    )
    
    # Create equations
    linear_equation = Equation(
        equation_id="linear_function",
        latex_expression="y = mx + b",
        variables={"x": var_x, "y": var_y, "m": var_m, "b": var_b},
        mathematical_properties=["linear", "continuous", "differentiable"],
        applications=["regression", "modeling", "prediction"],
        complexity_level=2
    )
    
    quadratic_equation = Equation(
        equation_id="quadratic_function",
        latex_expression="y = ax^2 + bx + c",
        variables={"x": var_x, "y": var_y},
        mathematical_properties=["quadratic", "continuous", "differentiable"],
        applications=["optimization", "physics", "engineering"],
        complexity_level=3
    )
    
    # Create explanations
    linear_explanation = Explanation(
        explanation_type="intuitive",
        content="""
        A linear equation represents the simplest form of relationship between two variables.
        The equation y = mx + b describes a straight line where:
        - m controls how steep the line is (slope)
        - b determines where the line crosses the y-axis (y-intercept)
        
        This fundamental relationship appears everywhere in mathematics and science,
        from simple proportional relationships to complex machine learning models.
        """,
        mathematical_level=2,
        prerequisites=["basic_algebra"]
    )
    
    quadratic_explanation = Explanation(
        explanation_type="formal",
        content="""
        A quadratic function is a polynomial function of degree 2. The general form
        y = ax² + bx + c represents a parabola that opens upward (a > 0) or downward (a < 0).
        
        Key properties:
        - Vertex at x = -b/(2a)
        - Axis of symmetry at x = -b/(2a)  
        - Discriminant Δ = b² - 4ac determines the number of real roots
        """,
        mathematical_level=3,
        prerequisites=["linear_equations", "basic_algebra"]
    )
    
    # Create numerical examples
    linear_example = NumericalExample(
        example_id="linear_example_1",
        description="Computing linear function values",
        input_values={
            "x": np.array([0, 1, 2, 3, 4]),
            "m": np.array([2.0]),
            "b": np.array([1.0])
        },
        computation_steps=[
            ComputationStep(
                step_number=1,
                operation_name="linear_transformation",
                input_values={"x": np.array([0, 1, 2, 3, 4]), "m": np.array([2.0]), "b": np.array([1.0])},
                operation_description="Apply linear transformation y = mx + b with m=2, b=1",
                output_values={"y": np.array([1, 3, 5, 7, 9])}
            )
        ],
        output_values={"y": np.array([1, 3, 5, 7, 9])},
        educational_notes=[
            "Notice how each unit increase in x results in a 2-unit increase in y (the slope)",
            "The y-intercept is 1, which is where the line crosses the y-axis when x=0"
        ]
    )
    
    quadratic_example = NumericalExample(
        example_id="quadratic_example_1", 
        description="Computing quadratic function values",
        input_values={
            "x": np.array([-2, -1, 0, 1, 2]),
            "a": np.array([1.0]),
            "b": np.array([0.0]),
            "c": np.array([0.0])
        },
        computation_steps=[
            ComputationStep(
                step_number=1,
                operation_name="quadratic_transformation",
                input_values={"x": np.array([-2, -1, 0, 1, 2])},
                operation_description="Apply quadratic transformation y = x² (a=1, b=0, c=0)",
                output_values={"y": np.array([4, 1, 0, 1, 4])}
            )
        ],
        output_values={"y": np.array([4, 1, 0, 1, 4])},
        educational_notes=[
            "This is the simplest quadratic function y = x²",
            "Notice the symmetric parabola shape with vertex at (0,0)"
        ]
    )
    
    # Create visualizations
    linear_viz = Visualization(
        visualization_id="linear_plot",
        visualization_type="line_plot",
        title="Linear Function Visualization",
        description="Interactive plot showing the linear relationship y = mx + b",
        data=VisualizationData(
            visualization_type="line_plot",
            data={
                "x_values": [0, 1, 2, 3, 4],
                "y_values": [1, 3, 5, 7, 9],
                "slope": 2.0,
                "intercept": 1.0
            },
            color_mappings={
                "line": "#FF6B6B",
                "points": "#4ECDC4"
            },
            interactive_elements=["slope_slider", "intercept_slider"]
        ),
        interactive=True
    )
    
    quadratic_viz = Visualization(
        visualization_id="quadratic_plot",
        visualization_type="curve_plot", 
        title="Quadratic Function Visualization",
        description="Interactive plot showing the parabolic relationship y = ax² + bx + c",
        data=VisualizationData(
            visualization_type="curve_plot",
            data={
                "x_values": [-2, -1, 0, 1, 2],
                "y_values": [4, 1, 0, 1, 4],
                "coefficients": {"a": 1.0, "b": 0.0, "c": 0.0}
            },
            color_mappings={
                "curve": "#45B7D1",
                "vertex": "#96CEB4"
            },
            interactive_elements=["coefficient_sliders", "vertex_marker"]
        ),
        interactive=True,
        prerequisites=["linear_equations"]
    )
    
    # Create mathematical concepts
    linear_concept = MathematicalConcept(
        concept_id="linear_equations",
        title="Linear Equations and Functions",
        prerequisites=["basic_algebra"],
        equations=[linear_equation],
        explanations=[linear_explanation],
        examples=[linear_example],
        visualizations=[linear_viz],
        difficulty_level=2,
        learning_objectives=[
            "Understand the structure of linear equations",
            "Interpret slope and y-intercept parameters",
            "Apply linear functions to real-world problems"
        ]
    )
    
    quadratic_concept = MathematicalConcept(
        concept_id="quadratic_equations",
        title="Quadratic Equations and Functions", 
        prerequisites=["linear_equations"],
        equations=[quadratic_equation],
        explanations=[quadratic_explanation],
        examples=[quadratic_example],
        visualizations=[quadratic_viz],
        difficulty_level=3,
        learning_objectives=[
            "Understand quadratic function properties",
            "Analyze parabolic relationships",
            "Solve quadratic optimization problems"
        ]
    )
    
    # Create chapters
    foundations_chapter = TutorialChapter(
        chapter_id="mathematical_foundations",
        title="Mathematical Foundations",
        concepts=[linear_concept],
        introduction="""
        Welcome to the Mathematical Foundations chapter. Here we'll explore the fundamental
        building blocks of mathematical relationships, starting with linear equations.
        These concepts form the foundation for more advanced topics in mathematics and
        machine learning.
        """,
        summary="""
        In this chapter, we covered linear equations and their properties. You learned how
        to interpret slope and y-intercept, work with linear transformations, and visualize
        linear relationships. These skills are essential for understanding more complex
        mathematical concepts.
        """,
        chapter_number=1,
        estimated_time_minutes=45
    )
    
    advanced_chapter = TutorialChapter(
        chapter_id="advanced_functions",
        title="Advanced Function Types",
        concepts=[quadratic_concept],
        introduction="""
        Building on linear functions, we now explore quadratic functions and their rich
        mathematical properties. Quadratic functions introduce curvature and optimization
        concepts that are fundamental to calculus and machine learning.
        """,
        summary="""
        This chapter introduced quadratic functions, their geometric properties, and
        applications. You learned about parabolas, vertices, and the relationship between
        algebraic and geometric representations of quadratic relationships.
        """,
        chapter_number=2,
        estimated_time_minutes=60
    )
    
    return [foundations_chapter, advanced_chapter]


def demonstrate_content_assembly():
    """Demonstrate the complete content assembly system."""
    
    print("=== AI Math Tutorial Content Assembly Demo ===\n")
    
    # Create demo content
    print("1. Creating demonstration content...")
    chapters = create_demo_content()
    print(f"   Created {len(chapters)} chapters with mathematical concepts\n")
    
    # Set up the assembly system
    print("2. Setting up content assembly system...")
    config = AssemblyConfiguration(
        include_navigation=True,
        include_prerequisites=True,
        include_cross_references=True,
        generate_toc=True,
        validate_links=False,  # Disable for demo since we don't have all prerequisites
        cache_content=True
    )
    assembler = ChapterAssembler(config)
    print("   Assembly system configured\n")
    
    # Assemble individual chapters
    print("3. Assembling individual chapters...")
    results = []
    for chapter in chapters:
        result = assembler.assemble_chapter(chapter)
        results.append(result)
        status = "✓ Success" if result.success else "✗ Failed"
        print(f"   Chapter '{chapter.title}': {status}")
        if result.warnings:
            for warning in result.warnings:
                print(f"     Warning: {warning}")
        if result.errors:
            for error in result.errors:
                print(f"     Error: {error}")
    print()
    
    # Generate table of contents
    print("4. Generating table of contents...")
    link_manager = LinkManager()
    toc_generator = TableOfContentsGenerator(link_manager)
    
    # Add some sample progress
    toc_generator.update_progress("linear_equations", MasteryLevel.COMPLETED, 100.0, 45)
    toc_generator.update_progress("quadratic_equations", MasteryLevel.IN_PROGRESS, 60.0, 25)
    
    toc_entries = toc_generator.generate_toc(chapters, include_progress=True)
    print(f"   Generated TOC with {len(toc_entries)} chapters")
    
    # Validate prerequisite chains
    is_valid, errors = toc_generator.validate_prerequisite_chains(toc_entries)
    print(f"   Prerequisite validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"     {error}")
    print()
    
    # Generate learning paths
    print("5. Generating learning paths...")
    learning_paths = toc_generator.generate_learning_paths(toc_entries, ["quadratic_equations"])
    print(f"   Generated {len(learning_paths)} learning paths")
    for path in learning_paths:
        print(f"   Path to {path.path_id.replace('path_to_', '')}: {' → '.join(path.concept_sequence)}")
        print(f"     Estimated time: {path.estimated_time_minutes} minutes")
        print(f"     Difficulty progression: {path.difficulty_progression}")
    print()
    
    # Generate progress report
    print("6. Generating progress report...")
    progress_report = toc_generator.generate_progress_report()
    print(f"   Concepts started: {progress_report['overview']['concepts_started']}")
    print(f"   Concepts completed: {progress_report['overview']['concepts_completed']}")
    print(f"   Total time spent: {progress_report['overview']['total_time_spent_minutes']} minutes")
    print(f"   Completion rate: {progress_report['overview']['completion_rate']:.1%}")
    print(f"   Next recommendations: {', '.join(progress_report['next_recommendations'])}")
    print()
    
    # Show content statistics
    print("7. Content assembly statistics...")
    for i, result in enumerate(results):
        chapter = chapters[i]
        stats = result.metadata['content_stats']
        print(f"   Chapter {chapter.chapter_number}: {chapter.title}")
        print(f"     Content length: {len(result.chapter_html):,} characters")
        print(f"     Concepts: {stats['total_concepts']}")
        print(f"     Equations: {stats['total_equations']}")
        print(f"     Examples: {stats['total_examples']}")
        print(f"     Visualizations: {stats['total_visualizations']}")
        print(f"     Explanations: {stats['total_explanations']}")
    print()
    
    # Generate and show sample HTML output
    print("8. Sample HTML output...")
    if results[0].success:
        html_sample = results[0].chapter_html[:500] + "..." if len(results[0].chapter_html) > 500 else results[0].chapter_html
        print("   First 500 characters of assembled HTML:")
        print(f"   {html_sample}")
    print()
    
    # Generate TOC HTML
    print("9. Table of contents HTML...")
    toc_html = toc_generator.render_toc_html(toc_entries, include_progress=True)
    toc_sample = toc_html[:400] + "..." if len(toc_html) > 400 else toc_html
    print("   First 400 characters of TOC HTML:")
    print(f"   {toc_sample}")
    print()
    
    print("=== Demo Complete ===")
    print("The content rendering and assembly system successfully:")
    print("✓ Created mathematical concepts with equations, explanations, examples, and visualizations")
    print("✓ Assembled chapters with integrated content and navigation")
    print("✓ Generated table of contents with prerequisite validation")
    print("✓ Created learning paths and progress tracking")
    print("✓ Rendered HTML output with cross-references and styling")


if __name__ == "__main__":
    demonstrate_content_assembly()