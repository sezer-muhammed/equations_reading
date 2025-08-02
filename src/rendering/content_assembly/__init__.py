"""
Content assembly module for the AI Math Tutorial system.
Provides template system, content integration, and chapter assembly functionality.
"""

from .template_system import TemplateSystem, ContentTemplate
from .content_integrator import ContentIntegrator, ContentSection, IntegrationRule
from .chapter_assembler import ChapterAssembler, AssemblyConfiguration, AssemblyResult
from .toc_generator import (
    TableOfContentsGenerator, MasteryLevel, ConceptProgress, 
    LearningPath, TOCEntry
)

__all__ = [
    'TemplateSystem',
    'ContentTemplate', 
    'ContentIntegrator',
    'ContentSection',
    'IntegrationRule',
    'ChapterAssembler',
    'AssemblyConfiguration',
    'AssemblyResult',
    'TableOfContentsGenerator',
    'MasteryLevel',
    'ConceptProgress',
    'LearningPath',
    'TOCEntry'
]