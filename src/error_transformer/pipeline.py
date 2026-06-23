"""
src/error_transformer/pipeline.py
-------------------------------------------
End-to-end batch processing pipeline for error transformation.

Orchestrates ContextRetriever and LLMErrorTransformer to process a list
of raw validator error dicts into human-friendly TransformedError objects.
"""

from typing import Optional

from .models import ValidatorError, TransformedError
from .retriever import ContextRetriever
from .llm_client import LLMErrorTransformer


class ErrorTransformPipeline:
    """
    End-to-end pipeline: raw error rows → human-friendly messages.

    Usage:
        pipeline = ErrorTransformPipeline(index, sections_data, requirements_data)
        results  = pipeline.transform_batch(error_dicts)
    """

    def __init__(
        self,
        index,
        sections_data: list[dict],
        requirements_data: list[dict],
        api_key: Optional[str] = None,
    ):
        self.retriever   = ContextRetriever(index, sections_data, requirements_data)
        self.transformer = LLMErrorTransformer(api_key=api_key)

    def transform_one(self, error_dict: dict) -> TransformedError:
        error   = ValidatorError.from_dict(error_dict)
        context = self.retriever.retrieve(error)
        return self.transformer.transform(error, context)

    def transform_batch(self, error_dicts: list[dict]) -> list[TransformedError]:
        results = []
        for i, err_dict in enumerate(error_dicts, 1):
            print(f"  [{i}/{len(error_dicts)}] {err_dict.get('element_path', '')[:60]}")
            try:
                results.append(self.transform_one(err_dict))
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                results.append(self._fallback(err_dict, str(e)))
        return results

    def _fallback(self, err_dict: dict, reason: str) -> TransformedError:
        """Produce a minimal output when the LLM call fails."""
        return TransformedError(
            message    = err_dict.get("description", "Validation error"),
            cause      = f"Value on {err_dict.get('element_path', '')} does not match.",
            correction  = (
                f"Make sure the value is match with the expected value: "
                f"{err_dict.get('expected_value', '')}."
            ),
            reference  = err_dict.get("standard", ""),
            location   = err_dict.get("element_path", ""),
            severity   = err_dict.get("severity", ""),
            section_id = "",
            raw_error  = err_dict,
            context_used = False,
        )
