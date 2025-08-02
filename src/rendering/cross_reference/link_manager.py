"""
Cross-reference linking system for connecting mathematical concepts.
Manages links between concepts, equations, examples, and prerequisites.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import re
from collections import defaultdict, deque
from src.core.models import MathematicalConcept, TutorialChapter


@dataclass
class CrossReference:
    """Represents a cross-reference link between content elements."""
    source_id: str
    target_id: str
    reference_type: str  # "prerequisite", "related", "example", "equation", "visualization"
    link_text: str
    context: Optional[str] = None


@dataclass
class LinkValidationResult:
    """Result of link validation."""
    is_valid: bool
    broken_links: List[str] = field(default_factory=list)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    missing_prerequisites: List[str] = field(default_factory=list)


class LinkManager:
    """Manages cross-reference links between mathematical concepts."""
    
    def __init__(self):
        self.concepts: Dict[str, MathematicalConcept] = {}
        self.cross_references: Dict[str, List[CrossReference]] = defaultdict(list)
        self.reverse_references: Dict[str, List[CrossReference]] = defaultdict(list)
        self.prerequisite_graph: Dict[str, Set[str]] = defaultdict(set)
        self.link_templates: Dict[str, str] = {}
        self._initialize_link_templates()
    
    def _initialize_link_templates(self):
        """Initialize templates for different types of links."""
        self.link_templates = {
            'prerequisite': '<a href="#concept-{target_id}" class="prerequisite-link" title="Prerequisite: {link_text}">{link_text}</a>',
            'related': '<a href="#concept-{target_id}" class="related-link" title="Related concept: {link_text}">{link_text}</a>',
            'equation': '<a href="#eq-{target_id}" class="equation-link" title="Equation: {link_text}">{link_text}</a>',
            'example': '<a href="#example-{target_id}" class="example-link" title="Example: {link_text}">{link_text}</a>',
            'visualization': '<a href="#viz-{target_id}" class="visualization-link" title="Visualization: {link_text}">{link_text}</a>',
            'chapter': '<a href="#chapter-{target_id}" class="chapter-link" title="Chapter: {link_text}">{link_text}</a>'
        }
    
    def register_concept(self, concept: MathematicalConcept):
        """Register a concept and build its cross-references."""
        self.concepts[concept.concept_id] = concept
        
        # Build prerequisite relationships
        for prereq in concept.prerequisites:
            self.prerequisite_graph[concept.concept_id].add(prereq)
            
            # Create cross-reference
            cross_ref = CrossReference(
                source_id=concept.concept_id,
                target_id=prereq,
                reference_type="prerequisite",
                link_text=prereq,
                context=f"Prerequisite for {concept.title}"
            )
            self.cross_references[concept.concept_id].append(cross_ref)
            self.reverse_references[prereq].append(cross_ref)
    
    def add_cross_reference(self, source_id: str, target_id: str, 
                           reference_type: str, link_text: str, 
                           context: Optional[str] = None):
        """Add a custom cross-reference between content elements."""
        cross_ref = CrossReference(
            source_id=source_id,
            target_id=target_id,
            reference_type=reference_type,
            link_text=link_text,
            context=context
        )
        
        self.cross_references[source_id].append(cross_ref)
        self.reverse_references[target_id].append(cross_ref)
    
    def generate_link_html(self, cross_ref: CrossReference) -> str:
        """Generate HTML for a cross-reference link."""
        template = self.link_templates.get(cross_ref.reference_type, 
                                         self.link_templates['related'])
        
        return template.format(
            target_id=cross_ref.target_id,
            link_text=cross_ref.link_text
        )
    
    def get_prerequisites_chain(self, concept_id: str) -> List[str]:
        """Get the complete prerequisite chain for a concept."""
        visited = set()
        chain = []
        
        def dfs(current_id):
            if current_id in visited:
                return
            visited.add(current_id)
            
            # Add prerequisites first (depth-first)
            for prereq in self.prerequisite_graph.get(current_id, []):
                dfs(prereq)
            
            chain.append(current_id)
        
        dfs(concept_id)
        return chain[:-1]  # Exclude the concept itself
    
    def get_learning_path(self, target_concept_id: str) -> List[str]:
        """Get the optimal learning path to reach a target concept."""
        # Use topological sort to find the learning order
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        # Build the graph and calculate in-degrees
        all_concepts = set([target_concept_id])
        queue = deque([target_concept_id])
        
        while queue:
            current = queue.popleft()
            for prereq in self.prerequisite_graph.get(current, []):
                if prereq not in all_concepts:
                    all_concepts.add(prereq)
                    queue.append(prereq)
                graph[prereq].append(current)
                in_degree[current] += 1
        
        # Topological sort
        learning_path = []
        zero_in_degree = deque([concept for concept in all_concepts if in_degree[concept] == 0])
        
        while zero_in_degree:
            current = zero_in_degree.popleft()
            learning_path.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)
        
        return learning_path
    
    def validate_links(self) -> LinkValidationResult:
        """Validate all cross-reference links."""
        result = LinkValidationResult(is_valid=True)
        
        # Check for broken links
        for source_id, refs in self.cross_references.items():
            for ref in refs:
                if ref.reference_type == "prerequisite" or ref.reference_type == "related":
                    if ref.target_id not in self.concepts:
                        result.broken_links.append(f"{source_id} -> {ref.target_id}")
                        result.is_valid = False
        
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies()
        if circular_deps:
            result.circular_dependencies = circular_deps
            result.is_valid = False
        
        # Check for missing prerequisites
        for concept_id, concept in self.concepts.items():
            for prereq in concept.prerequisites:
                if prereq not in self.concepts:
                    result.missing_prerequisites.append(f"{concept_id} requires {prereq}")
                    result.is_valid = False
        
        return result
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the prerequisite graph."""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.prerequisite_graph.get(node, []):
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for concept_id in self.concepts:
            if concept_id not in visited:
                dfs(concept_id, [])
        
        return cycles
    
    def generate_navigation_links(self, current_concept_id: str) -> Dict[str, Any]:
        """Generate navigation links for a concept."""
        navigation = {
            'prerequisites': [],
            'dependents': [],
            'related': [],
            'next_suggested': None,
            'previous_suggested': None
        }
        
        # Get prerequisites
        for prereq in self.prerequisite_graph.get(current_concept_id, []):
            if prereq in self.concepts:
                navigation['prerequisites'].append({
                    'id': prereq,
                    'title': self.concepts[prereq].title,
                    'link': f"#concept-{prereq}"
                })
        
        # Get dependents (concepts that depend on this one)
        for concept_id, prereqs in self.prerequisite_graph.items():
            if current_concept_id in prereqs and concept_id in self.concepts:
                navigation['dependents'].append({
                    'id': concept_id,
                    'title': self.concepts[concept_id].title,
                    'link': f"#concept-{concept_id}"
                })
        
        # Get related concepts (same difficulty level, similar topics)
        current_concept = self.concepts.get(current_concept_id)
        if current_concept:
            for concept_id, concept in self.concepts.items():
                if (concept_id != current_concept_id and 
                    concept.difficulty_level == current_concept.difficulty_level):
                    navigation['related'].append({
                        'id': concept_id,
                        'title': concept.title,
                        'link': f"#concept-{concept_id}"
                    })
        
        return navigation
    
    def generate_prerequisite_tree(self, concept_id: str) -> Dict[str, Any]:
        """Generate a tree structure of prerequisites."""
        if concept_id not in self.concepts:
            return {}
        
        def build_tree(node_id, visited=None):
            if visited is None:
                visited = set()
            
            if node_id in visited:
                return {'id': node_id, 'title': 'Circular Reference', 'children': []}
            
            visited.add(node_id)
            concept = self.concepts.get(node_id)
            if not concept:
                return {'id': node_id, 'title': 'Missing Concept', 'children': []}
            
            tree = {
                'id': node_id,
                'title': concept.title,
                'difficulty': concept.difficulty_level,
                'children': []
            }
            
            for prereq in self.prerequisite_graph.get(node_id, []):
                child_tree = build_tree(prereq, visited.copy())
                tree['children'].append(child_tree)
            
            return tree
        
        return build_tree(concept_id)
    
    def suggest_next_concepts(self, completed_concepts: Set[str]) -> List[str]:
        """Suggest next concepts to study based on completed ones."""
        suggestions = []
        
        for concept_id, concept in self.concepts.items():
            if concept_id in completed_concepts:
                continue
            
            # Check if all prerequisites are completed
            prereqs = self.prerequisite_graph.get(concept_id, set())
            if prereqs.issubset(completed_concepts):
                suggestions.append(concept_id)
        
        # Sort by difficulty level
        suggestions.sort(key=lambda x: self.concepts[x].difficulty_level)
        return suggestions
    
    def export_link_graph(self) -> Dict[str, Any]:
        """Export the link graph for visualization or analysis."""
        nodes = []
        edges = []
        
        # Add nodes
        for concept_id, concept in self.concepts.items():
            nodes.append({
                'id': concept_id,
                'title': concept.title,
                'difficulty': concept.difficulty_level,
                'type': 'concept'
            })
        
        # Add edges
        for source_id, refs in self.cross_references.items():
            for ref in refs:
                edges.append({
                    'source': source_id,
                    'target': ref.target_id,
                    'type': ref.reference_type,
                    'label': ref.link_text
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_concepts': len(self.concepts),
                'total_links': sum(len(refs) for refs in self.cross_references.values())
            }
        }