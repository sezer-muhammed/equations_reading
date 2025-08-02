"""
Table of contents generator with prerequisite chain validation and learning path navigation.
Provides progress tracking and concept mastery indicators.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from src.core.models import MathematicalConcept, TutorialChapter
from src.rendering.cross_reference.link_manager import LinkManager


class MasteryLevel(Enum):
    """Levels of concept mastery."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    MASTERED = "mastered"


@dataclass
class ConceptProgress:
    """Progress tracking for a mathematical concept."""
    concept_id: str
    mastery_level: MasteryLevel
    completion_percentage: float = 0.0
    time_spent_minutes: int = 0
    last_accessed: Optional[str] = None
    attempts: int = 0
    success_rate: float = 0.0


@dataclass
class LearningPath:
    """Represents an optimal learning path through concepts."""
    path_id: str
    concept_sequence: List[str]
    estimated_time_minutes: int
    difficulty_progression: List[int]
    prerequisites_satisfied: bool
    description: str


@dataclass
class TOCEntry:
    """Entry in the table of contents."""
    entry_id: str
    title: str
    entry_type: str  # "chapter", "concept", "section"
    level: int  # Hierarchical level (0=chapter, 1=concept, 2=section)
    url_fragment: str
    prerequisites: List[str]
    difficulty_level: int
    estimated_time_minutes: int
    progress: Optional[ConceptProgress] = None
    children: List['TOCEntry'] = field(default_factory=list)


class TableOfContentsGenerator:
    """Generates comprehensive table of contents with learning path navigation."""
    
    def __init__(self, link_manager: LinkManager):
        self.link_manager = link_manager
        self.progress_tracker: Dict[str, ConceptProgress] = {}
        self.learning_paths: Dict[str, LearningPath] = {}
        self.toc_cache: Dict[str, List[TOCEntry]] = {}
    
    def generate_toc(self, chapters: List[TutorialChapter], 
                    include_progress: bool = True) -> List[TOCEntry]:
        """Generate complete table of contents for multiple chapters."""
        
        cache_key = f"toc_{'_'.join([ch.chapter_id for ch in chapters])}"
        if cache_key in self.toc_cache:
            return self.toc_cache[cache_key]
        
        toc_entries = []
        
        for chapter in chapters:
            # Register all concepts with link manager
            for concept in chapter.concepts:
                self.link_manager.register_concept(concept)
            
            # Create chapter entry
            chapter_entry = TOCEntry(
                entry_id=chapter.chapter_id,
                title=chapter.title,
                entry_type="chapter",
                level=0,
                url_fragment=f"#chapter-{chapter.chapter_number}",
                prerequisites=[],
                difficulty_level=self._calculate_chapter_difficulty(chapter),
                estimated_time_minutes=chapter.estimated_time_minutes
            )
            
            # Add concept entries as children
            for concept in chapter.concepts:
                concept_entry = self._create_concept_entry(concept, include_progress)
                chapter_entry.children.append(concept_entry)
            
            toc_entries.append(chapter_entry)
        
        # Cache the result
        self.toc_cache[cache_key] = toc_entries
        return toc_entries
    
    def _create_concept_entry(self, concept: MathematicalConcept, 
                            include_progress: bool) -> TOCEntry:
        """Create a TOC entry for a mathematical concept."""
        
        progress = None
        if include_progress and concept.concept_id in self.progress_tracker:
            progress = self.progress_tracker[concept.concept_id]
        
        concept_entry = TOCEntry(
            entry_id=concept.concept_id,
            title=concept.title,
            entry_type="concept",
            level=1,
            url_fragment=f"#concept-{concept.concept_id}",
            prerequisites=concept.prerequisites,
            difficulty_level=concept.difficulty_level,
            estimated_time_minutes=self._estimate_concept_time(concept),
            progress=progress
        )
        
        # Add sub-sections for equations, examples, visualizations
        if concept.equations:
            for equation in concept.equations:
                eq_entry = TOCEntry(
                    entry_id=equation.equation_id,
                    title=f"Equation: {equation.equation_id}",
                    entry_type="equation",
                    level=2,
                    url_fragment=f"#eq-{equation.equation_id}",
                    prerequisites=[],
                    difficulty_level=equation.complexity_level,
                    estimated_time_minutes=5
                )
                concept_entry.children.append(eq_entry)
        
        if concept.examples:
            for example in concept.examples:
                ex_entry = TOCEntry(
                    entry_id=example.example_id,
                    title=f"Example: {example.description}",
                    entry_type="example",
                    level=2,
                    url_fragment=f"#example-{example.example_id}",
                    prerequisites=[],
                    difficulty_level=concept.difficulty_level,
                    estimated_time_minutes=10
                )
                concept_entry.children.append(ex_entry)
        
        if concept.visualizations:
            for viz in concept.visualizations:
                viz_entry = TOCEntry(
                    entry_id=viz.visualization_id,
                    title=f"Visualization: {viz.title}",
                    entry_type="visualization",
                    level=2,
                    url_fragment=f"#viz-{viz.visualization_id}",
                    prerequisites=viz.prerequisites,
                    difficulty_level=concept.difficulty_level,
                    estimated_time_minutes=8
                )
                concept_entry.children.append(viz_entry)
        
        return concept_entry
    
    def validate_prerequisite_chains(self, toc_entries: List[TOCEntry]) -> Tuple[bool, List[str]]:
        """Validate that all prerequisite chains are complete and valid."""
        errors = []
        all_concept_ids = set()
        
        # Collect all concept IDs
        def collect_concept_ids(entries):
            for entry in entries:
                if entry.entry_type == "concept":
                    all_concept_ids.add(entry.entry_id)
                collect_concept_ids(entry.children)
        
        collect_concept_ids(toc_entries)
        
        # Check prerequisites
        def check_prerequisites(entries):
            for entry in entries:
                if entry.entry_type == "concept":
                    for prereq in entry.prerequisites:
                        if prereq not in all_concept_ids:
                            errors.append(f"Concept '{entry.entry_id}' requires missing prerequisite '{prereq}'")
                check_prerequisites(entry.children)
        
        check_prerequisites(toc_entries)
        
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies(toc_entries)
        if circular_deps:
            for cycle in circular_deps:
                errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        return len(errors) == 0, errors
    
    def _find_circular_dependencies(self, toc_entries: List[TOCEntry]) -> List[List[str]]:
        """Find circular dependencies in prerequisite chains."""
        # Build prerequisite graph
        graph = defaultdict(set)
        
        def build_graph(entries):
            for entry in entries:
                if entry.entry_type == "concept":
                    for prereq in entry.prerequisites:
                        graph[entry.entry_id].add(prereq)
                build_graph(entry.children)
        
        build_graph(toc_entries)
        
        # Find cycles using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for concept_id in graph:
            if concept_id not in visited:
                dfs(concept_id, [])
        
        return cycles
    
    def generate_learning_paths(self, toc_entries: List[TOCEntry], 
                               target_concepts: Optional[List[str]] = None) -> List[LearningPath]:
        """Generate optimal learning paths through the content."""
        
        if target_concepts is None:
            # Generate paths to all advanced concepts
            target_concepts = self._find_advanced_concepts(toc_entries)
        
        learning_paths = []
        
        for target in target_concepts:
            path = self._generate_path_to_concept(target, toc_entries)
            if path:
                learning_paths.append(path)
        
        return learning_paths
    
    def _generate_path_to_concept(self, target_concept: str, 
                                 toc_entries: List[TOCEntry]) -> Optional[LearningPath]:
        """Generate learning path to reach a specific concept."""
        
        # Get prerequisite chain
        prereq_chain = self.link_manager.get_learning_path(target_concept)
        
        if not prereq_chain:
            return None
        
        # Calculate path metrics
        total_time = 0
        difficulty_progression = []
        
        for concept_id in prereq_chain:
            concept_entry = self._find_concept_entry(concept_id, toc_entries)
            if concept_entry:
                total_time += concept_entry.estimated_time_minutes
                difficulty_progression.append(concept_entry.difficulty_level)
        
        # Check if prerequisites are satisfied
        prerequisites_satisfied = self._check_prerequisites_satisfied(prereq_chain)
        
        path = LearningPath(
            path_id=f"path_to_{target_concept}",
            concept_sequence=prereq_chain,
            estimated_time_minutes=total_time,
            difficulty_progression=difficulty_progression,
            prerequisites_satisfied=prerequisites_satisfied,
            description=f"Learning path to master {target_concept}"
        )
        
        return path
    
    def _find_advanced_concepts(self, toc_entries: List[TOCEntry]) -> List[str]:
        """Find concepts with high difficulty levels (4-5)."""
        advanced_concepts = []
        
        def find_advanced(entries):
            for entry in entries:
                if entry.entry_type == "concept" and entry.difficulty_level >= 4:
                    advanced_concepts.append(entry.entry_id)
                find_advanced(entry.children)
        
        find_advanced(toc_entries)
        return advanced_concepts
    
    def _find_concept_entry(self, concept_id: str, 
                           toc_entries: List[TOCEntry]) -> Optional[TOCEntry]:
        """Find a concept entry by ID."""
        
        def search_entries(entries):
            for entry in entries:
                if entry.entry_type == "concept" and entry.entry_id == concept_id:
                    return entry
                result = search_entries(entry.children)
                if result:
                    return result
            return None
        
        return search_entries(toc_entries)
    
    def _check_prerequisites_satisfied(self, concept_sequence: List[str]) -> bool:
        """Check if all prerequisites in a sequence are satisfied."""
        completed_concepts = set()
        
        for concept_id in concept_sequence:
            progress = self.progress_tracker.get(concept_id)
            if progress and progress.mastery_level in [MasteryLevel.COMPLETED, MasteryLevel.MASTERED]:
                completed_concepts.add(concept_id)
        
        # Check if each concept's prerequisites are in the completed set
        for concept_id in concept_sequence:
            concept = self.link_manager.concepts.get(concept_id)
            if concept:
                for prereq in concept.prerequisites:
                    if prereq not in completed_concepts and prereq in concept_sequence:
                        return False
        
        return True
    
    def update_progress(self, concept_id: str, mastery_level: MasteryLevel, 
                       completion_percentage: float = 0.0, 
                       time_spent_minutes: int = 0):
        """Update progress for a concept."""
        
        if concept_id not in self.progress_tracker:
            self.progress_tracker[concept_id] = ConceptProgress(
                concept_id=concept_id,
                mastery_level=mastery_level
            )
        
        progress = self.progress_tracker[concept_id]
        progress.mastery_level = mastery_level
        progress.completion_percentage = completion_percentage
        progress.time_spent_minutes += time_spent_minutes
        progress.attempts += 1
        
        # Update success rate based on mastery level
        if mastery_level in [MasteryLevel.COMPLETED, MasteryLevel.MASTERED]:
            progress.success_rate = min(1.0, progress.success_rate + 0.1)
        else:
            progress.success_rate = max(0.0, progress.success_rate - 0.05)
    
    def get_next_recommended_concepts(self, max_recommendations: int = 5) -> List[str]:
        """Get next recommended concepts based on current progress."""
        
        completed_concepts = set()
        for concept_id, progress in self.progress_tracker.items():
            if progress.mastery_level in [MasteryLevel.COMPLETED, MasteryLevel.MASTERED]:
                completed_concepts.add(concept_id)
        
        # Use link manager to suggest next concepts
        suggestions = self.link_manager.suggest_next_concepts(completed_concepts)
        
        # Sort by difficulty and return top recommendations
        concept_difficulties = []
        for concept_id in suggestions:
            concept = self.link_manager.concepts.get(concept_id)
            if concept:
                concept_difficulties.append((concept_id, concept.difficulty_level))
        
        concept_difficulties.sort(key=lambda x: x[1])
        return [concept_id for concept_id, _ in concept_difficulties[:max_recommendations]]
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """Generate a comprehensive progress report."""
        
        total_concepts = len(self.link_manager.concepts)
        completed_concepts = sum(1 for p in self.progress_tracker.values() 
                               if p.mastery_level in [MasteryLevel.COMPLETED, MasteryLevel.MASTERED])
        
        mastery_distribution = defaultdict(int)
        for progress in self.progress_tracker.values():
            mastery_distribution[progress.mastery_level.value] += 1
        
        total_time_spent = sum(p.time_spent_minutes for p in self.progress_tracker.values())
        average_success_rate = (sum(p.success_rate for p in self.progress_tracker.values()) / 
                              len(self.progress_tracker) if self.progress_tracker else 0)
        
        report = {
            'overview': {
                'total_concepts': total_concepts,
                'concepts_started': len(self.progress_tracker),
                'concepts_completed': completed_concepts,
                'completion_rate': completed_concepts / total_concepts if total_concepts > 0 else 0,
                'total_time_spent_minutes': total_time_spent,
                'average_success_rate': average_success_rate
            },
            'mastery_distribution': dict(mastery_distribution),
            'next_recommendations': self.get_next_recommended_concepts(),
            'learning_paths_available': len(self.learning_paths),
            'detailed_progress': [
                {
                    'concept_id': p.concept_id,
                    'mastery_level': p.mastery_level.value,
                    'completion_percentage': p.completion_percentage,
                    'time_spent_minutes': p.time_spent_minutes,
                    'success_rate': p.success_rate,
                    'attempts': p.attempts
                }
                for p in self.progress_tracker.values()
            ]
        }
        
        return report
    
    def render_toc_html(self, toc_entries: List[TOCEntry], 
                       include_progress: bool = True) -> str:
        """Render table of contents as HTML."""
        
        def render_entry(entry: TOCEntry, level: int = 0) -> str:
            indent = "  " * level
            progress_html = ""
            
            if include_progress and entry.progress:
                progress = entry.progress
                progress_class = f"progress-{progress.mastery_level.value}"
                progress_percentage = progress.completion_percentage
                
                progress_html = f"""
                <div class="progress-indicator {progress_class}">
                    <div class="progress-bar" style="width: {progress_percentage}%"></div>
                    <span class="progress-text">{progress_percentage:.0f}%</span>
                </div>
                """
            
            difficulty_stars = "★" * entry.difficulty_level
            time_estimate = f"{entry.estimated_time_minutes}min"
            
            entry_html = f"""
            {indent}<li class="toc-entry toc-{entry.entry_type}" data-level="{entry.level}">
            {indent}  <a href="{entry.url_fragment}" class="toc-link">
            {indent}    <span class="toc-title">{entry.title}</span>
            {indent}    <span class="toc-metadata">
            {indent}      <span class="difficulty">{difficulty_stars}</span>
            {indent}      <span class="time-estimate">{time_estimate}</span>
            {indent}    </span>
            {indent}  </a>
            {indent}  {progress_html}
            """
            
            if entry.children:
                entry_html += f"\n{indent}  <ul class='toc-children'>\n"
                for child in entry.children:
                    entry_html += render_entry(child, level + 2)
                entry_html += f"\n{indent}  </ul>\n"
            
            entry_html += f"\n{indent}</li>\n"
            return entry_html
        
        html = "<div class='table-of-contents'>\n"
        html += "  <h2>Table of Contents</h2>\n"
        html += "  <ul class='toc-root'>\n"
        
        for entry in toc_entries:
            html += render_entry(entry, 1)
        
        html += "  </ul>\n"
        html += "</div>"
        
        return html
    
    def _calculate_chapter_difficulty(self, chapter: TutorialChapter) -> int:
        """Calculate average difficulty level for a chapter."""
        if not chapter.concepts:
            return 1
        
        total_difficulty = sum(concept.difficulty_level for concept in chapter.concepts)
        return round(total_difficulty / len(chapter.concepts))
    
    def _estimate_concept_time(self, concept: MathematicalConcept) -> int:
        """Estimate time needed to complete a concept."""
        base_time = 15  # Base time per concept
        equation_time = len(concept.equations) * 10
        example_time = len(concept.examples) * 15
        viz_time = len(concept.visualizations) * 8
        explanation_time = len(concept.explanations) * 5
        
        total_time = base_time + equation_time + example_time + viz_time + explanation_time
        
        # Adjust for difficulty
        difficulty_multiplier = 1 + (concept.difficulty_level - 1) * 0.3
        return int(total_time * difficulty_multiplier)
    
    def clear_cache(self):
        """Clear the TOC cache."""
        self.toc_cache.clear()