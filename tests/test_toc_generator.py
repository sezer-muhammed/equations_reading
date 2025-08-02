"""
Tests for the table of contents generator.
"""

import pytest
import numpy as np
from src.core.models import (
    MathematicalConcept, Equation, Explanation, Visualization, 
    NumericalExample, TutorialChapter, VariableDefinition,
    ComputationStep, VisualizationData
)
from src.rendering.content_assembly.toc_generator import (
    TableOfContentsGenerator, MasteryLevel, ConceptProgress, TOCEntry
)
from src.rendering.cross_reference.link_manager import LinkManager


def create_sample_concepts():
    """Create sample mathematical concepts for testing."""
    
    # Basic algebra concept (prerequisite)
    basic_algebra = MathematicalConcept(
        concept_id="basic_algebra",
        title="Basic Algebra",
        prerequisites=[],
        equations=[],
        explanations=[],
        examples=[],
        visualizations=[],
        difficulty_level=1,
        learning_objectives=["Understand algebraic operations"]
    )
    
    # Linear equations concept
    linear_equations = MathematicalConcept(
        concept_id="linear_equations",
        title="Linear Equations",
        prerequisites=["basic_algebra"],
        equations=[
            Equation(
                equation_id="linear_eq",
                latex_expression="y = mx + b",
                variables={},
                mathematical_properties=["linear"],
                complexity_level=2
            )
        ],
        explanations=[
            Explanation(
                explanation_type="intuitive",
                content="Linear equations represent straight lines.",
                mathematical_level=2
            )
        ],
        examples=[
            NumericalExample(
                example_id="linear_example",
                description="Simple linear example",
                input_values={"x": np.array([1, 2, 3])},
                computation_steps=[],
                output_values={"y": np.array([3, 5, 7])}
            )
        ],
        visualizations=[
            Visualization(
                visualization_id="linear_plot",
                visualization_type="line_plot",
                title="Linear Plot",
                description="Plot of linear equation",
                data=VisualizationData(
                    visualization_type="line_plot",
                    data={},
                    color_mappings={}
                )
            )
        ],
        difficulty_level=2,
        learning_objectives=["Understand linear relationships"]
    )
    
    # Advanced calculus concept
    advanced_calculus = MathematicalConcept(
        concept_id="advanced_calculus",
        title="Advanced Calculus",
        prerequisites=["linear_equations"],
        equations=[],
        explanations=[],
        examples=[],
        visualizations=[],
        difficulty_level=5,
        learning_objectives=["Master advanced calculus concepts"]
    )
    
    return [basic_algebra, linear_equations, advanced_calculus]


def create_sample_chapters():
    """Create sample chapters for testing."""
    concepts = create_sample_concepts()
    
    chapter1 = TutorialChapter(
        chapter_id="foundations",
        title="Mathematical Foundations",
        concepts=[concepts[0], concepts[1]],  # basic_algebra, linear_equations
        introduction="Introduction to mathematical foundations",
        summary="We covered basic algebra and linear equations",
        chapter_number=1,
        estimated_time_minutes=60
    )
    
    chapter2 = TutorialChapter(
        chapter_id="advanced_topics",
        title="Advanced Topics",
        concepts=[concepts[2]],  # advanced_calculus
        introduction="Advanced mathematical concepts",
        summary="We covered advanced calculus",
        chapter_number=2,
        estimated_time_minutes=90
    )
    
    return [chapter1, chapter2]


class TestTableOfContentsGenerator:
    """Test the table of contents generator."""
    
    def test_basic_toc_generation(self):
        """Test basic TOC generation."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        toc_entries = toc_generator.generate_toc(chapters, include_progress=False)
        
        assert len(toc_entries) == 2  # Two chapters
        assert toc_entries[0].title == "Mathematical Foundations"
        assert toc_entries[1].title == "Advanced Topics"
        
        # Check chapter 1 has 2 concepts
        assert len(toc_entries[0].children) == 2
        assert toc_entries[0].children[0].entry_id == "basic_algebra"
        assert toc_entries[0].children[1].entry_id == "linear_equations"
        
        # Check chapter 2 has 1 concept
        assert len(toc_entries[1].children) == 1
        assert toc_entries[1].children[0].entry_id == "advanced_calculus"
    
    def test_concept_children_generation(self):
        """Test that concept children (equations, examples, visualizations) are generated."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        toc_entries = toc_generator.generate_toc(chapters, include_progress=False)
        
        # Check linear_equations concept has children
        linear_concept = toc_entries[0].children[1]  # linear_equations
        assert len(linear_concept.children) == 3  # 1 equation + 1 example + 1 visualization
        
        # Check child types
        child_types = [child.entry_type for child in linear_concept.children]
        assert "equation" in child_types
        assert "example" in child_types
        assert "visualization" in child_types
    
    def test_prerequisite_chain_validation(self):
        """Test prerequisite chain validation."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        toc_entries = toc_generator.generate_toc(chapters, include_progress=False)
        is_valid, errors = toc_generator.validate_prerequisite_chains(toc_entries)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_prerequisite_chain_validation_with_missing_prereq(self):
        """Test prerequisite chain validation with missing prerequisite."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        
        # Create concept with missing prerequisite
        concept_with_missing_prereq = MathematicalConcept(
            concept_id="test_concept",
            title="Test Concept",
            prerequisites=["missing_concept"],  # This doesn't exist
            equations=[],
            explanations=[],
            examples=[],
            visualizations=[],
            difficulty_level=3,
            learning_objectives=[]
        )
        
        chapter = TutorialChapter(
            chapter_id="test_chapter",
            title="Test Chapter",
            concepts=[concept_with_missing_prereq],
            introduction="Test",
            summary="Test",
            chapter_number=1,
            estimated_time_minutes=30
        )
        
        toc_entries = toc_generator.generate_toc([chapter], include_progress=False)
        is_valid, errors = toc_generator.validate_prerequisite_chains(toc_entries)
        
        assert not is_valid
        assert len(errors) > 0
        assert "missing_concept" in errors[0]
    
    def test_learning_path_generation(self):
        """Test learning path generation."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        toc_entries = toc_generator.generate_toc(chapters, include_progress=False)
        learning_paths = toc_generator.generate_learning_paths(toc_entries, ["advanced_calculus"])
        
        assert len(learning_paths) > 0
        path = learning_paths[0]
        assert path.path_id == "path_to_advanced_calculus"
        assert "basic_algebra" in path.concept_sequence
        assert "linear_equations" in path.concept_sequence
        assert "advanced_calculus" in path.concept_sequence
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        
        # Update progress for a concept
        toc_generator.update_progress("basic_algebra", MasteryLevel.COMPLETED, 100.0, 30)
        
        assert "basic_algebra" in toc_generator.progress_tracker
        progress = toc_generator.progress_tracker["basic_algebra"]
        assert progress.mastery_level == MasteryLevel.COMPLETED
        assert progress.completion_percentage == 100.0
        assert progress.time_spent_minutes == 30
        assert progress.attempts == 1
    
    def test_next_recommendations(self):
        """Test next concept recommendations."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        # Generate TOC to register concepts
        toc_generator.generate_toc(chapters, include_progress=False)
        
        # Mark basic_algebra as completed
        toc_generator.update_progress("basic_algebra", MasteryLevel.COMPLETED, 100.0, 30)
        
        recommendations = toc_generator.get_next_recommended_concepts()
        
        # Should recommend linear_equations since basic_algebra is completed
        assert "linear_equations" in recommendations
    
    def test_progress_report_generation(self):
        """Test progress report generation."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        # Generate TOC to register concepts
        toc_generator.generate_toc(chapters, include_progress=False)
        
        # Add some progress
        toc_generator.update_progress("basic_algebra", MasteryLevel.COMPLETED, 100.0, 30)
        toc_generator.update_progress("linear_equations", MasteryLevel.IN_PROGRESS, 50.0, 20)
        
        report = toc_generator.generate_progress_report()
        
        assert "overview" in report
        assert "mastery_distribution" in report
        assert "next_recommendations" in report
        assert "detailed_progress" in report
        
        assert report["overview"]["concepts_started"] == 2
        assert report["overview"]["concepts_completed"] == 1
        assert report["overview"]["total_time_spent_minutes"] == 50
    
    def test_toc_html_rendering(self):
        """Test HTML rendering of TOC."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        toc_entries = toc_generator.generate_toc(chapters, include_progress=False)
        html = toc_generator.render_toc_html(toc_entries, include_progress=False)
        
        assert "table-of-contents" in html
        assert "Mathematical Foundations" in html
        assert "Advanced Topics" in html
        assert "Basic Algebra" in html
        assert "Linear Equations" in html
        assert "Advanced Calculus" in html
    
    def test_toc_html_rendering_with_progress(self):
        """Test HTML rendering of TOC with progress indicators."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        # Add progress
        toc_generator.update_progress("basic_algebra", MasteryLevel.COMPLETED, 100.0, 30)
        
        toc_entries = toc_generator.generate_toc(chapters, include_progress=True)
        html = toc_generator.render_toc_html(toc_entries, include_progress=True)
        
        assert "progress-indicator" in html
        assert "progress-completed" in html
        assert "100%" in html
    
    def test_caching(self):
        """Test TOC caching functionality."""
        link_manager = LinkManager()
        toc_generator = TableOfContentsGenerator(link_manager)
        chapters = create_sample_chapters()
        
        # First generation
        toc_entries1 = toc_generator.generate_toc(chapters, include_progress=False)
        
        # Second generation should use cache
        toc_entries2 = toc_generator.generate_toc(chapters, include_progress=False)
        
        assert len(toc_generator.toc_cache) == 1
        assert toc_entries1 is toc_entries2  # Should be the same object from cache


if __name__ == "__main__":
    # Run a simple test
    link_manager = LinkManager()
    toc_generator = TableOfContentsGenerator(link_manager)
    chapters = create_sample_chapters()
    
    toc_entries = toc_generator.generate_toc(chapters, include_progress=False)
    
    print(f"Generated TOC with {len(toc_entries)} chapters")
    for entry in toc_entries:
        print(f"  Chapter: {entry.title} ({len(entry.children)} concepts)")
        for concept in entry.children:
            print(f"    Concept: {concept.title} (difficulty: {concept.difficulty_level})")
    
    # Test prerequisite validation
    is_valid, errors = toc_generator.validate_prerequisite_chains(toc_entries)
    print(f"Prerequisite validation: {'Valid' if is_valid else 'Invalid'}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")
    
    # Test learning paths
    learning_paths = toc_generator.generate_learning_paths(toc_entries, ["advanced_calculus"])
    print(f"Generated {len(learning_paths)} learning paths")
    for path in learning_paths:
        print(f"  Path: {' -> '.join(path.concept_sequence)} ({path.estimated_time_minutes} min)")