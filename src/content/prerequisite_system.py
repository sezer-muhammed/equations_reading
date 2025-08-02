"""
Prerequisite dependency system for managing concept relationships.
Handles dependency tracking, validation, and learning path generation.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx


@dataclass
class ConceptNode:
    """Represents a mathematical concept in the dependency graph."""
    concept_id: str
    title: str
    difficulty_level: int
    estimated_time_minutes: int = 30
    learning_objectives: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.concept_id)


@dataclass
class PrerequisiteDependency:
    """Represents a prerequisite relationship between concepts."""
    prerequisite_id: str
    dependent_id: str
    dependency_type: str = "required"  # "required", "recommended", "helpful"
    strength: float = 1.0  # 0.0 to 1.0, strength of the dependency
    
    def __hash__(self):
        return hash((self.prerequisite_id, self.dependent_id))


class PrerequisiteGraph:
    """Graph-based system for managing prerequisite dependencies."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concepts: Dict[str, ConceptNode] = {}
        self.dependencies: Set[PrerequisiteDependency] = set()
    
    def add_concept(self, concept: ConceptNode) -> None:
        """Add a concept to the dependency graph."""
        self.concepts[concept.concept_id] = concept
        self.graph.add_node(concept.concept_id, concept=concept)
    
    def add_dependency(self, dependency: PrerequisiteDependency) -> bool:
        """Add a prerequisite dependency between concepts."""
        # Validate that both concepts exist
        if (dependency.prerequisite_id not in self.concepts or 
            dependency.dependent_id not in self.concepts):
            return False
        
        # Check for circular dependencies
        if self._would_create_cycle(dependency.prerequisite_id, dependency.dependent_id):
            return False
        
        self.dependencies.add(dependency)
        self.graph.add_edge(
            dependency.prerequisite_id, 
            dependency.dependent_id,
            dependency=dependency
        )
        return True
    
    def _would_create_cycle(self, prerequisite_id: str, dependent_id: str) -> bool:
        """Check if adding this dependency would create a cycle."""
        # If there's already a path from dependent to prerequisite, 
        # adding prerequisite -> dependent would create a cycle
        try:
            return nx.has_path(self.graph, dependent_id, prerequisite_id)
        except nx.NetworkXError:
            return False
    
    def get_prerequisites(self, concept_id: str, 
                         dependency_type: Optional[str] = None) -> List[str]:
        """Get all prerequisites for a given concept."""
        if concept_id not in self.graph:
            return []
        
        prerequisites = []
        for pred in self.graph.predecessors(concept_id):
            edge_data = self.graph[pred][concept_id]
            dependency = edge_data['dependency']
            
            if dependency_type is None or dependency.dependency_type == dependency_type:
                prerequisites.append(pred)
        
        return prerequisites
    
    def get_dependents(self, concept_id: str) -> List[str]:
        """Get all concepts that depend on the given concept."""
        if concept_id not in self.graph:
            return []
        return list(self.graph.successors(concept_id))
    
    def get_learning_path(self, target_concept_id: str) -> List[str]:
        """Generate an optimal learning path to reach the target concept."""
        if target_concept_id not in self.graph:
            return []
        
        # Use topological sort to get a valid ordering
        try:
            # Get all concepts that are prerequisites (directly or indirectly)
            ancestors = nx.ancestors(self.graph, target_concept_id)
            ancestors.add(target_concept_id)
            
            # Create subgraph with only relevant concepts
            subgraph = self.graph.subgraph(ancestors)
            
            # Get topological ordering
            topo_order = list(nx.topological_sort(subgraph))
            
            return topo_order
        except nx.NetworkXError:
            return []
    
    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """Validate the entire dependency graph for consistency."""
        errors = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            errors.append(f"Circular dependencies detected: {cycles}")
        
        # Check for orphaned concepts (no prerequisites and no dependents)
        orphaned = []
        for concept_id in self.concepts:
            if (self.graph.in_degree(concept_id) == 0 and 
                self.graph.out_degree(concept_id) == 0):
                orphaned.append(concept_id)
        
        if orphaned:
            errors.append(f"Orphaned concepts (no connections): {orphaned}")
        
        # Check for missing concepts in dependencies
        for dependency in self.dependencies:
            if dependency.prerequisite_id not in self.concepts:
                errors.append(f"Prerequisite concept not found: {dependency.prerequisite_id}")
            if dependency.dependent_id not in self.concepts:
                errors.append(f"Dependent concept not found: {dependency.dependent_id}")
        
        return len(errors) == 0, errors
    
    def get_difficulty_progression(self, learning_path: List[str]) -> List[Tuple[str, int]]:
        """Get difficulty levels for concepts in a learning path."""
        progression = []
        for concept_id in learning_path:
            if concept_id in self.concepts:
                difficulty = self.concepts[concept_id].difficulty_level
                progression.append((concept_id, difficulty))
        return progression
    
    def suggest_next_concepts(self, completed_concepts: Set[str]) -> List[str]:
        """Suggest next concepts to learn based on completed prerequisites."""
        available = []
        
        for concept_id, concept in self.concepts.items():
            if concept_id in completed_concepts:
                continue
            
            # Check if all required prerequisites are completed
            required_prereqs = self.get_prerequisites(concept_id, "required")
            if all(prereq in completed_concepts for prereq in required_prereqs):
                available.append(concept_id)
        
        # Sort by difficulty level
        available.sort(key=lambda cid: self.concepts[cid].difficulty_level)
        return available


class PrerequisiteManager:
    """High-level manager for prerequisite dependencies."""
    
    def __init__(self):
        self.graph = PrerequisiteGraph()
        self.concept_categories: Dict[str, List[str]] = defaultdict(list)
    
    def register_concept(self, 
                        concept_id: str,
                        title: str,
                        difficulty_level: int,
                        category: str = "general",
                        estimated_time_minutes: int = 30,
                        learning_objectives: Optional[List[str]] = None) -> None:
        """Register a new mathematical concept."""
        concept = ConceptNode(
            concept_id=concept_id,
            title=title,
            difficulty_level=difficulty_level,
            estimated_time_minutes=estimated_time_minutes,
            learning_objectives=learning_objectives or []
        )
        
        self.graph.add_concept(concept)
        self.concept_categories[category].append(concept_id)
    
    def add_prerequisite(self, 
                        prerequisite_id: str,
                        dependent_id: str,
                        dependency_type: str = "required",
                        strength: float = 1.0) -> bool:
        """Add a prerequisite relationship."""
        dependency = PrerequisiteDependency(
            prerequisite_id=prerequisite_id,
            dependent_id=dependent_id,
            dependency_type=dependency_type,
            strength=strength
        )
        
        return self.graph.add_dependency(dependency)
    
    def create_learning_curriculum(self, 
                                  target_concepts: List[str],
                                  max_difficulty: int = 5) -> Dict[str, Any]:
        """Create a complete learning curriculum for target concepts."""
        curriculum = {
            "concepts": [],
            "learning_paths": {},
            "estimated_total_time": 0,
            "difficulty_distribution": defaultdict(int)
        }
        
        all_required_concepts = set()
        
        # Collect all concepts needed for targets
        for target in target_concepts:
            path = self.graph.get_learning_path(target)
            all_required_concepts.update(path)
            curriculum["learning_paths"][target] = path
        
        # Filter by difficulty if specified
        if max_difficulty < 5:
            filtered_concepts = []
            for concept_id in all_required_concepts:
                concept = self.graph.concepts.get(concept_id)
                if concept and concept.difficulty_level <= max_difficulty:
                    filtered_concepts.append(concept_id)
            all_required_concepts = set(filtered_concepts)
        
        # Create ordered curriculum
        try:
            subgraph = self.graph.graph.subgraph(all_required_concepts)
            ordered_concepts = list(nx.topological_sort(subgraph))
            
            total_time = 0
            for concept_id in ordered_concepts:
                concept = self.graph.concepts[concept_id]
                curriculum["concepts"].append({
                    "id": concept_id,
                    "title": concept.title,
                    "difficulty": concept.difficulty_level,
                    "time_minutes": concept.estimated_time_minutes,
                    "objectives": concept.learning_objectives
                })
                total_time += concept.estimated_time_minutes
                curriculum["difficulty_distribution"][concept.difficulty_level] += 1
            
            curriculum["estimated_total_time"] = total_time
            
        except nx.NetworkXError as e:
            curriculum["error"] = f"Failed to create curriculum: {str(e)}"
        
        return curriculum
    
    def get_concept_info(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a concept."""
        if concept_id not in self.graph.concepts:
            return None
        
        concept = self.graph.concepts[concept_id]
        prerequisites = self.graph.get_prerequisites(concept_id)
        dependents = self.graph.get_dependents(concept_id)
        
        return {
            "id": concept.concept_id,
            "title": concept.title,
            "difficulty_level": concept.difficulty_level,
            "estimated_time_minutes": concept.estimated_time_minutes,
            "learning_objectives": concept.learning_objectives,
            "prerequisites": prerequisites,
            "dependents": dependents,
            "learning_path": self.graph.get_learning_path(concept_id)
        }
    
    def validate_system(self) -> Tuple[bool, List[str]]:
        """Validate the entire prerequisite system."""
        return self.graph.validate_dependencies()
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for visualization or external use."""
        nodes = []
        edges = []
        
        for concept_id, concept in self.graph.concepts.items():
            nodes.append({
                "id": concept_id,
                "title": concept.title,
                "difficulty": concept.difficulty_level,
                "time": concept.estimated_time_minutes
            })
        
        for dependency in self.graph.dependencies:
            edges.append({
                "source": dependency.prerequisite_id,
                "target": dependency.dependent_id,
                "type": dependency.dependency_type,
                "strength": dependency.strength
            })
        
        return {"nodes": nodes, "edges": edges}