# src/error_transformer.py
"""
Error Transformer — converts raw validator errors to human-friendly messages
using the Hybrid Index for context retrieval and Claude API for generation.

Input (one row from validator CSV):
  {
    "validation_rule": "Profile Package Rule Set.pukCodes.puk_Header.Identification",
    "element_path":    "ProfileElement[3].pukCodes.puk_Header.Identification",
    "description":     "Value is incorrect",
    "expected_value":  "00",
    "saip_value":      "01",
    "standard":        "Profile interoperability technical specification (TCA)",
    "severity":        "Major",
  }

Output:
  {
    "pesan":      "Incorrect PUK header identification",
    "penyebab":   "Value PUK header identification tidak sesuai standar.",
    "perbaikan":  "Expected value untuk PUK header identification adalah 00.",
    "referensi":  "Profile interoperability technical specification (TCA) section 8.5.2",
    "terjadi_di": "ProfileElement[3].pukCodes.puk_Header.Identification",
    "severity":   "Major",
    "section_id": "8.5.2",
    "context":    { ... retrieved KO + requirements ... }
  }
"""

from os import write
import re
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidatorError:
    """One row from the validator output."""
    validation_rule: str
    element_path: str
    description: str
    expected_value: str
    saip_value: str
    standard: str
    severity: str
    index: int = 0
    status: str = "Fail"
    rule_set: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ValidatorError":
        return cls(
            validation_rule = d.get("validation_rule", d.get("Validation Rule", "")),
            element_path    = d.get("element_path",    d.get("Element Path", "")),
            description     = d.get("description",     d.get("Description", "")),
            expected_value  = d.get("expected_value",  d.get("Expected Value", "")),
            saip_value      = d.get("saip_value",      d.get("SAIP Value", "")),
            standard        = d.get("standard",        d.get("Standard Referenced", "")),
            severity        = d.get("severity",        d.get("Severity", "")),
            index           = int(d.get("index", d.get("Index", 0))),
            status          = d.get("status",          d.get("Status", "Fail")),
            rule_set        = d.get("rule_set",        d.get("SAIP Rule Set", "")),
        )


@dataclass
class RetrievedContext:
    """Context retrieved from the Hybrid Index for one error."""
    primary_ko: Optional[dict]           # ValidationRule KO
    field_ko: Optional[dict]             # Field KO (if distinct from VR)
    pe_ko: Optional[dict]                # Parent PE KO
    type_ko: Optional[dict]              # Referenced type KO (e.g. PEHeader)
    requirements: list[dict]             # linked Requirement KOs
    section_body: str                    # raw section body text
    section_id: str
    section_title: str
    found_expected_value: Optional[str]  # from spec (may differ from validator)

    def to_context_text(self) -> str:
        """Build the context string to inject into the LLM prompt."""
        parts = []

        if self.primary_ko:
            parts.append(f"[Validation Rule]\n{self.primary_ko['text_content']}")

        if self.pe_ko:
            parts.append(f"[Parent Profile Element]\n{self.pe_ko['text_content'][:600]}")

        if self.type_ko:
            parts.append(f"[Referenced Type]\n{self.type_ko['text_content'][:400]}")

        if self.section_body:
            parts.append(f"[Section {self.section_id} — {self.section_title}]\n{self.section_body[:600]}")

        if self.requirements:
            req_text = "\n".join(
                f"- [{r['metadata'].get('strength','').upper()}] {r['text_content'][:200]}"
                for r in self.requirements[:5]
            )
            parts.append(f"[Normative Requirements]\n{req_text}")

        return "\n\n".join(parts)


@dataclass
class TransformedError:
    """Human-friendly error output."""
    pesan: str
    penyebab: str
    perbaikan: str
    referensi: str
    terjadi_di: str
    severity: str
    section_id: str
    raw_error: dict
    context_used: bool = True

    def to_dict(self) -> dict:
        return {
            "pesan":      self.pesan,
            "penyebab":   self.penyebab,
            "perbaikan":  self.perbaikan,
            "referensi":  self.referensi,
            "terjadi_di": self.terjadi_di,
            "severity":   self.severity,
            "section_id": self.section_id,
        }

    def format_display(self) -> str:
        return (
            f"Pesan      : {self.pesan}\n"
            f"Penyebab   : {self.penyebab}\n"
            f"Perbaikan  : {self.perbaikan}\n"
            f"Referensi  : {self.referensi}\n"
            f"Terjadi di : {self.terjadi_di}\n"
            f"Severity   : {self.severity}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context Retriever
# ─────────────────────────────────────────────────────────────────────────────

class ContextRetriever:
    """
    Retrieves and assembles context from the Hybrid Index for a given error.
    """

    def __init__(self, index, sections_data: list[dict], requirements_data: list[dict]):
        self._index = index
        self._sec_index = {s["section_id"]: s for s in sections_data}
        self._req_index = {r["req_id"]: r for r in requirements_data}
        # KO index for fast lookup by ko_id
        self._ko_index = {ko["ko_id"]: ko for ko in index._kos}

    def retrieve(self, error: ValidatorError) -> RetrievedContext:
        """
        Retrieve all relevant context for one error.
        Strategy:
          1. Primary: direct path lookup → ValidationRule KO
          2. Graph traverse: BELONGS_TO → PE, HAS_TYPE → type, VALIDATED_BY → requirements
          3. Fallback: hybrid search on description + element_path tokens
          4. Section: body text from primary KO's section
        """
        primary_ko = self._index.lookup_path(error.element_path)

        # Fallback: semantic search if path lookup missed
        if not primary_ko:
            query = f"{error.element_path} {error.description} {error.expected_value}"
            results = self._index.search(query, top_k=3, ko_type="ValidationRule")
            primary_ko = results[0].__dict__ if results else None
            if primary_ko:
                # Convert SearchResult attrs to KO dict shape
                primary_ko = {
                    "ko_id": primary_ko["ko_id"],
                    "ko_type": primary_ko["ko_type"],
                    "primary_label": primary_ko["primary_label"],
                    "text_content": primary_ko["text_content"],
                    "metadata": primary_ko["metadata"],
                }

        rule_id    = primary_ko["metadata"].get("rule_id", "") if primary_ko else ""
        section_id = primary_ko["metadata"].get("section_id", "") if primary_ko else ""
        field_ref  = primary_ko["metadata"].get("field_ref", "") if primary_ko else ""

        # Graph traversal
        pe_ko   = self._resolve_neighbor(rule_id, "BELONGS_TO")
        type_ko = self._resolve_neighbor(rule_id, "HAS_TYPE")

        # Field KO (separate from VR KO)
        field_ko = self._find_field_ko(rule_id)

        # Requirements via VALIDATED_BY edges
        req_ids = [
            e["target_id"]
            for e in self._index.graph_neighbors(rule_id, rel_type="VALIDATED_BY")
        ]
        requirement_kos = [
            self._ko_index.get(f"KO-REQ-{rid}", self._req_to_ko(self._req_index.get(rid)))
            for rid in req_ids[:6]
            if self._req_index.get(rid)
        ]
        requirement_kos = [r for r in requirement_kos if r]

        # Section context
        section = self._sec_index.get(section_id, {})
        section_body  = section.get("body_text", "")
        section_title = section.get("title", "")

        # Expected value: from spec default, fallback to validator's expected
        spec_expected = primary_ko["metadata"].get("expected_value") if primary_ko else None

        return RetrievedContext(
            primary_ko=primary_ko,
            field_ko=field_ko,
            pe_ko=pe_ko,
            type_ko=type_ko,
            requirements=requirement_kos,
            section_body=section_body,
            section_id=section_id,
            section_title=section_title,
            found_expected_value=spec_expected or error.expected_value,
        )

    def _resolve_neighbor(self, entity_id: str, rel_type: str) -> Optional[dict]:
        """Get first neighbor of given rel_type and return its KO."""
        edges = self._index.graph_neighbors(entity_id, rel_type=rel_type)
        if not edges:
            return None
        target_id = edges[0]["target_id"]
        # Try to find as PE KO
        ko_id_pe    = f"KO-PE-{target_id}"
        ko_id_field = f"KO-FIELD-{target_id}"
        return (
            self._ko_index.get(ko_id_pe)
            or self._ko_index.get(ko_id_field)
        )

    def _find_field_ko(self, rule_id: str) -> Optional[dict]:
        """Find the Field KO corresponding to this ValidationRule."""
        ko_id = f"KO-FIELD-{rule_id}"
        return self._ko_index.get(ko_id)

    def _req_to_ko(self, req: Optional[dict]) -> Optional[dict]:
        """Convert a requirement dict to a minimal KO-like dict."""
        if not req:
            return None
        return {
            "ko_id": req["req_id"],
            "ko_type": "Requirement",
            "primary_label": req["req_id"],
            "text_content": req["statement"],
            "metadata": {
                "strength": req["strength"],
                "keyword": req["keyword"],
                "section_id": req["section_id"],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM Client (Claude API)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah asisten teknis spesialis eUICC/eSIM profile validation.
Tugasmu mengubah pesan error teknis dari validator ASN.1/profile menjadi pesan yang mudah dipahami oleh engineer.

Output HARUS dalam format JSON yang valid dengan key persis:
{
  "message": "Judul singkat error (max 10 kata, bahasa Inggris)",
  "cause": "Penjelasan mengapa error terjadi (1-2 kalimat, bahasa Inggris)",
  "correction": "Langkah konkret untuk memperbaiki (1-2 kalimat, bahasa inggris)",
  "reference": "Nama dokumen dan section (misal: TCA Profile Interoperability section 8.5.2)"
}

Gunakan context dari spec yang diberikan. Jangan tambahkan key lain. Output HANYA JSON, tidak ada teks lain."""

USER_PROMPT_TEMPLATE = """Error validator berikut perlu ditransform:

=== ERROR INFO ===
Element Path    : {element_path}
Validation Rule : {validation_rule}
Description     : {description}
Expected Value  : {expected_value}
Actual Value    : {saip_value}
Severity        : {severity}
Standard        : {standard}

=== CONTEXT DARI SPEC ===
{context_text}

Transform error di atas menjadi format JSON yang diminta."""


class LLMErrorTransformer:
    """Calls Claude API to generate human-friendly error messages."""

    MODEL = "claude-opus-4-8"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def transform(
        self,
        error: ValidatorError,
        context: RetrievedContext,
    ) -> TransformedError:
        """Call Claude API and parse structured output."""
    
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

        

        # Build referensi with section if not already included
        referensi = parsed.get("referensi", error.standard)
        if context.section_id and context.section_id not in referensi:
            referensi = f"{referensi} section {context.section_id}"

        return TransformedError(
            pesan      = parsed.get("message", error.description),
            penyebab   = parsed.get("cause", ""),
            perbaikan  = parsed.get("correction", ""),
            referensi  = referensi,
            terjadi_di = error.element_path,
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
        """Extract JSON from LLM response, stripping any markdown fences."""
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Fallback: try to find JSON object within the text
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (batch processing)
# ─────────────────────────────────────────────────────────────────────────────

class ErrorTransformPipeline:
    """
    End-to-end pipeline: raw error rows → transformed human-friendly messages.
    """

    def __init__(self, index, sections_data, requirements_data, api_key: Optional[str] = None):
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
                result = self.transform_one(err_dict)
                results.append(result)
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                # Graceful fallback
                results.append(self._fallback(err_dict, str(e)))
        return results

    def _fallback(self, err_dict: dict, reason: str) -> TransformedError:
        """Produce a minimal output when LLM call fails."""
        return TransformedError(
            pesan      = err_dict.get("description", "Validation error"),
            penyebab   = f"Value pada {err_dict.get('element_path', '')} tidak sesuai.",
            perbaikan  = f"Pastikan value sesuai dengan expected value: {err_dict.get('expected_value', '')}.",
            referensi  = err_dict.get("standard", ""),
            terjadi_di = err_dict.get("element_path", ""),
            severity   = err_dict.get("severity", ""),
            section_id = "",
            raw_error  = err_dict,
            context_used = False,
        )