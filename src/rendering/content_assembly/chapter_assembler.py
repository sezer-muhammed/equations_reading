"""
Chapter assembly engine for creating complete tutorial chapters.
Combines content templates, integration, and cross-reference linking.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from src.core.models import MathematicalConcept, TutorialChapter
from src.rendering.content_assembly.template_system import TemplateSystem
from src.rendering.content_assembly.content_integrator import ContentIntegrator
from src.rendering.cross_reference.link_manager import LinkManager


@dataclass
class AssemblyConfiguration:
    """Configuration for chapter assembly."""
    include_navigation: bool = True
    include_prerequisites: bool = True
    include_cross_references: bool = True
    generate_toc: bool = True
    validate_links: bool = True
    cache_content: bool = True


@dataclass
class AssemblyResult:
    """Result of chapter assembly process."""
    success: bool
    chapter_html: str
    metadata: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


class ChapterAssembler:
    """Main engine for assembling complete tutorial chapters."""
    
    def __init__(self, config: Optional[AssemblyConfiguration] = None):
        self.config = config or AssemblyConfiguration()
        self.template_system = TemplateSystem()
        self.content_integrator = ContentIntegrator(self.template_system)
        self.link_manager = LinkManager()
        self.assembly_cache: Dict[str, AssemblyResult] = {}
    
    def assemble_chapter(self, chapter: TutorialChapter) -> AssemblyResult:
        """Assemble a complete chapter with all content integrated."""
        
        # Check cache first
        cache_key = f"chapter_{chapter.chapter_id}"
        if self.config.cache_content and cache_key in self.assembly_cache:
            return self.assembly_cache[cache_key]
        
        warnings = []
        errors = []
        
        try:
            # Register all concepts with the link manager
            for concept in chapter.concepts:
                self.link_manager.register_concept(concept)
            
            # Validate links if requested
            if self.config.validate_links:
                validation_result = self.link_manager.validate_links()
                if not validation_result.is_valid:
                    errors.extend([f"Broken link: {link}" for link in validation_result.broken_links])
                    errors.extend([f"Circular dependency: {' -> '.join(cycle)}" 
                                 for cycle in validation_result.circular_dependencies])
                    warnings.extend([f"Missing prerequisite: {prereq}" 
                                   for prereq in validation_result.missing_prerequisites])
            
            # Generate chapter content
            chapter_html = self._generate_chapter_html(chapter)
            
            # Add navigation if requested
            if self.config.include_navigation:
                chapter_html = self._add_navigation(chapter_html, chapter)
            
            # Generate metadata
            metadata = self._generate_metadata(chapter)
            
            result = AssemblyResult(
                success=len(errors) == 0,
                chapter_html=chapter_html,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )
            
            # Cache the result
            if self.config.cache_content:
                self.assembly_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            errors.append(f"Assembly failed: {str(e)}")
            return AssemblyResult(
                success=False,
                chapter_html="",
                metadata={},
                warnings=warnings,
                errors=errors
            )
    
    def _generate_chapter_html(self, chapter: TutorialChapter) -> str:
        """Generate the main HTML content for the chapter."""
        
        # Start with chapter template
        chapter_html = self.content_integrator.integrate_chapter(chapter)
        
        # Add CSS classes and styling
        chapter_html = self._add_styling(chapter_html)
        
        # Process cross-references
        if self.config.include_cross_references:
            chapter_html = self._process_cross_references(chapter_html)
        
        return chapter_html
    
    def _add_navigation(self, chapter_html: str, chapter: TutorialChapter) -> str:
        """Add navigation elements to the chapter."""
        
        # Generate table of contents
        toc_html = ""
        if self.config.generate_toc:
            toc_html = self._generate_table_of_contents(chapter)
        
        # Generate concept navigation
        concept_nav_html = self._generate_concept_navigation(chapter)
        
        # Wrap chapter with navigation
        navigation_template = """
<div class="chapter-container">
    <nav class="chapter-navigation">
        {toc_html}
        {concept_nav_html}
    </nav>
    <main class="chapter-main">
        {chapter_html}
    </main>
</div>
        """
        
        return navigation_template.format(
            toc_html=toc_html,
            concept_nav_html=concept_nav_html,
            chapter_html=chapter_html
        )
    
    def _generate_table_of_contents(self, chapter: TutorialChapter) -> str:
        """Generate table of contents for the chapter."""
        
        toc_items = []
        for i, concept in enumerate(chapter.concepts, 1):
            toc_items.append(f"""
                <li class="toc-concept">
                    <a href="#concept-{concept.concept_id}" class="toc-link">
                        {i}. {concept.title}
                    </a>
                    <span class="difficulty-indicator" data-level="{concept.difficulty_level}">
                        {'★' * concept.difficulty_level}
                    </span>
                </li>
            """)
        
        toc_html = f"""
        <div class="table-of-contents">
            <h3>Chapter Contents</h3>
            <ol class="toc-list">
                {''.join(toc_items)}
            </ol>
        </div>
        """
        
        return toc_html
    
    def _generate_concept_navigation(self, chapter: TutorialChapter) -> str:
        """Generate navigation between concepts."""
        
        nav_items = []
        for concept in chapter.concepts:
            navigation = self.link_manager.generate_navigation_links(concept.concept_id)
            
            prereq_links = []
            if navigation['prerequisites']:
                prereq_links = [f'<a href="{item["link"]}" class="prereq-nav-link">{item["title"]}</a>' 
                               for item in navigation['prerequisites']]
            
            nav_items.append(f"""
                <div class="concept-nav-item" data-concept="{concept.concept_id}">
                    <h4>{concept.title}</h4>
                    {f'<div class="prereq-nav">Prerequisites: {", ".join(prereq_links)}</div>' if prereq_links else ''}
                </div>
            """)
        
        nav_html = f"""
        <div class="concept-navigation">
            <h3>Concept Navigation</h3>
            <div class="concept-nav-list">
                {''.join(nav_items)}
            </div>
        </div>
        """
        
        return nav_html
    
    def _process_cross_references(self, html_content: str) -> str:
        """Process and enhance cross-references in the content."""
        
        # This would typically involve:
        # 1. Finding reference patterns in the HTML
        # 2. Validating that targets exist
        # 3. Adding appropriate CSS classes and attributes
        # 4. Generating hover tooltips or previews
        
        # For now, we'll add some basic enhancements
        import re
        
        # Add tooltips to concept links
        def add_concept_tooltip(match):
            concept_id = match.group(1)
            concept = self.link_manager.concepts.get(concept_id)
            if concept:
                tooltip = f'title="Difficulty: {concept.difficulty_level}/5"'
                return match.group(0).replace('class="prerequisite-link"', 
                                            f'class="prerequisite-link" {tooltip}')
            return match.group(0)
        
        html_content = re.sub(r'href="#concept-([^"]+)"[^>]*class="prerequisite-link"[^>]*>', 
                             add_concept_tooltip, html_content)
        
        return html_content
    
    def _add_styling(self, html_content: str) -> str:
        """Add CSS classes and styling to the content."""
        
        # Add responsive classes
        html_content = html_content.replace('<div class="equation-container"', 
                                          '<div class="equation-container responsive-equation"')
        
        html_content = html_content.replace('<div class="visualization-container"', 
                                          '<div class="visualization-container responsive-viz"')
        
        return html_content
    
    def _generate_metadata(self, chapter: TutorialChapter) -> Dict[str, Any]:
        """Generate metadata for the assembled chapter."""
        
        # Count different content types
        total_equations = sum(len(concept.equations) for concept in chapter.concepts)
        total_examples = sum(len(concept.examples) for concept in chapter.concepts)
        total_visualizations = sum(len(concept.visualizations) for concept in chapter.concepts)
        total_explanations = sum(len(concept.explanations) for concept in chapter.concepts)
        
        # Calculate difficulty distribution
        difficulty_counts = {}
        for concept in chapter.concepts:
            level = concept.difficulty_level
            difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
        
        # Generate prerequisite information
        all_prerequisites = set()
        for concept in chapter.concepts:
            all_prerequisites.update(concept.prerequisites)
        
        metadata = {
            'chapter_id': chapter.chapter_id,
            'title': chapter.title,
            'chapter_number': chapter.chapter_number,
            'estimated_time_minutes': chapter.estimated_time_minutes,
            'content_stats': {
                'total_concepts': len(chapter.concepts),
                'total_equations': total_equations,
                'total_examples': total_examples,
                'total_visualizations': total_visualizations,
                'total_explanations': total_explanations
            },
            'difficulty_distribution': difficulty_counts,
            'prerequisites': list(all_prerequisites),
            'learning_objectives': [obj for concept in chapter.concepts 
                                  for obj in concept.learning_objectives],
            'generated_at': str(Path(__file__).stat().st_mtime),
            'assembly_config': {
                'include_navigation': self.config.include_navigation,
                'include_prerequisites': self.config.include_prerequisites,
                'include_cross_references': self.config.include_cross_references,
                'generate_toc': self.config.generate_toc
            }
        }
        
        return metadata
    
    def assemble_multiple_chapters(self, chapters: List[TutorialChapter]) -> List[AssemblyResult]:
        """Assemble multiple chapters with cross-chapter linking."""
        
        results = []
        
        # First pass: register all concepts across all chapters
        for chapter in chapters:
            for concept in chapter.concepts:
                self.link_manager.register_concept(concept)
        
        # Second pass: assemble each chapter
        for chapter in chapters:
            result = self.assemble_chapter(chapter)
            results.append(result)
        
        return results
    
    def export_assembly_report(self, results: List[AssemblyResult]) -> Dict[str, Any]:
        """Export a comprehensive report of the assembly process."""
        
        total_warnings = sum(len(result.warnings) for result in results)
        total_errors = sum(len(result.errors) for result in results)
        successful_assemblies = sum(1 for result in results if result.success)
        
        report = {
            'summary': {
                'total_chapters': len(results),
                'successful_assemblies': successful_assemblies,
                'total_warnings': total_warnings,
                'total_errors': total_errors,
                'success_rate': successful_assemblies / len(results) if results else 0
            },
            'chapter_details': [],
            'link_validation': self.link_manager.validate_links().__dict__,
            'cache_stats': {
                'cached_chapters': len(self.assembly_cache),
                'content_integrator_cache': self.content_integrator.get_cache_stats()
            }
        }
        
        for i, result in enumerate(results):
            report['chapter_details'].append({
                'chapter_index': i,
                'success': result.success,
                'warnings_count': len(result.warnings),
                'errors_count': len(result.errors),
                'content_length': len(result.chapter_html),
                'metadata': result.metadata
            })
        
        return report
    
    def clear_cache(self):
        """Clear all caches."""
        self.assembly_cache.clear()
        self.content_integrator.clear_cache()