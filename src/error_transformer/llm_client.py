"""
src/indexing/error_transformer/llm_client.py
---------------------------------------------
LLM client for the error transformation pipeline.

Calls the Claude API to turn a ValidatorError + RetrievedContext into
a structured, human-friendly TransformedError.  All prompt content
lives in .prompts so this module is concerned only with the API call
and JSON parsing logic.
"""

import re
import json
import urllib.request
import urllib.error
from typing import Optional

from .models import ValidatorError, RetrievedContext, TransformedError
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class LLMErrorTransformer:
    """Calls the Claude API to generate human-friendly error messages."""

    MODEL = "claude-opus-4-8"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def transform(
        self,
        error: ValidatorError,
        context: RetrievedContext,
    ) -> TransformedError:
        """Call Claude API and parse the structured JSON output."""
        context_text = context.to_context_text()
        user_prompt  = USER_PROMPT_TEMPLATE.format(
            element_path    = error.element_path,
            validation_rule = error.validation_rule,
            description     = error.description,
            expected_value  = error.expected_value,
            saip_value      = error.saip_value,
            severity        = error.severity,
            standard        = error.standard,
            context_text    = context_text,
        )

        response_text = self._call_api(user_prompt)
        parsed        = self._parse_response(response_text)

        referensi = parsed.get("referensi", error.standard)
        if context.section_id and context.section_id not in referensi:
            referensi = f"{referensi} section {context.section_id}"

        return TransformedError(
            message    = parsed.get("message",    error.description),
            cause      = parsed.get("cause",      ""),
            correction = parsed.get("correction", ""),
            reference  = referensi,
            location   = error.element_path,
            severity   = error.severity,
            section_id = context.section_id,
            raw_error  = {
                "validation_rule": error.validation_rule,
                "expected_value":  error.expected_value,
                "saip_value":      error.saip_value,
            },
            context_used = bool(context.primary_ko),
        )

    def _call_api(self, user_prompt: str) -> str:
        payload = json.dumps({
            "model":      self.MODEL,
            "max_tokens": 512,
            "system":     SYSTEM_PROMPT,
            "messages":   [{"role": "user", "content": user_prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data    = payload,
            headers = {
                "Content-Type":      "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key":         self._api_key,
            },
            method = "POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"API error {e.code}: {body}")

    def _parse_response(self, text: str) -> dict:
        """Extract JSON from the LLM response, stripping markdown fences."""
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$",           "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return {}
