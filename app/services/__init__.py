"""Service package initialiser.

Avoid importing submodules at package import time to prevent circular
imports (modules should import specific submodules directly where needed).
"""

__all__ = ["model_client", "qdrant_service", "pipeline_service"]
