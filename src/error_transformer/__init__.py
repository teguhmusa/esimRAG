"""
src/indexing/error_transformer
-------------------------------
Public surface of the error_transformer sub-package.

Preserves the original flat-module import contract:
    from src.indexing.error_transformer import (
        ValidatorError,
        RetrievedContext,
        TransformedError,
        ContextRetriever,
        LLMErrorTransformer,
        ErrorTransformPipeline,
    )
"""

from .models import ValidatorError, RetrievedContext, TransformedError
from .retriever import ContextRetriever
from .llm_client import LLMErrorTransformer
from .pipeline import ErrorTransformPipeline

__all__ = [
    "ValidatorError",
    "RetrievedContext",
    "TransformedError",
    "ContextRetriever",
    "LLMErrorTransformer",
    "ErrorTransformPipeline",
]
