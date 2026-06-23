"""
src/error_transformer/prompts.py
------------------------------------------
LLM prompt templates for the error transformation pipeline.

Keeping prompts in a dedicated module makes them easy to version,
review, and update independently of the LLM client logic.
"""

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
