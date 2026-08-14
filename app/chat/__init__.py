"""
Chat Domain.

Acts as the Ultimate Orchestrator. Coordinates Knowledge Base search and
Thesis validation modules to generate safe, relevant, and verified responses.
"""

from .api import router

__all__ = ["router"]
