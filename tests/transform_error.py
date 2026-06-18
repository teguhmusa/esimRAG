# transform_errors.py
"""
CLI untuk transform validator errors → human-friendly messages.

Usage:
  # Transform single error (JSON):
  python transform_errors.py --error '{"element_path": "...", ...}'

  # Transform CSV file:
  python transform_errors.py --csv errors.csv --output results.json

  # Test mode (no API call, show retrieved context only):
  python transform_errors.py --test --error '{"element_path": "..."}'
"""

import json
import argparse
import sys
from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import settings
from src.indexing import HybridIndex
from src.indexing.error_transformer import (
    ValidatorError, ContextRetriever,
    LLMErrorTransformer, ErrorTransformPipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sample errors for testing (from the spec example)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ERRORS = [
    {
        "index": 6,
        "status": "Fail",
        "severity": "Major",
        "standard": "Profile interoperability technical specification (TCA)",
        "validation_rule": "ProfilePackage.versionDependencyConsistency",
        "element_path": "ProfilePackage.profileHeader.payloadVersion vs ProfilePayload.schemaVersion",
        "description": "Payload schema version is incompatible with declared header version causing structural dependency conflict",
        "expected_value": "payloadVersion and schemaVersion must match supported matrix",
        "saip_value": "Header version=3.0, Payload schema=3.2 (unsupported combination)",
        "rule_set": "IYU_SAIP Validation Ruleset v33_1",
    },
    {
        "index": 2,
        "status": "Fail",
        "severity": "Major",
        "standard": "Profile interoperability technical specification (TCA)",
        "validation_rule": "Profile Package Rule Set.profileHeader.major-version",
        "element_path": "ProfileElement[1].profileHeader.major-version",
        "description": "Value is incorrect",
        "expected_value": "03",
        "saip_value": "02",
        "rule_set": "IYU_SAIP Validation Ruleset v33_1",
    },
    {
        "index": 3,
        "status": "Fail",
        "severity": "Minor",
        "standard": "Profile interoperability technical specification (TCA)",
        "validation_rule": "Profile Package Rule Set.pukCodes.pukCodes.pukValue",
        "element_path": "ProfileElement[3].pukCodes.pukCodes.pukValue",
        "description": "Length is incorrect",
        "expected_value": "8",
        "saip_value": "4",
        "rule_set": "IYU_SAIP Validation Ruleset v33_1",
    },
]


def load_index(index_dir: str = "output/index") -> HybridIndex:
    print(f"🔎 Loading index from {index_dir}...")
    return HybridIndex.load(index_dir)


def test_mode(error_dict: dict, index: HybridIndex, sections_data, requirements_data):
    """Show retrieved context without calling LLM."""
    print("\n" + "=" * 65)
    print("TEST MODE — Context Retrieval (no LLM call)")
    print("=" * 65)

    error = ValidatorError.from_dict(error_dict)
    retriever = ContextRetriever(index, sections_data, requirements_data)
    ctx = retriever.retrieve(error)

    print(f"\nError path    : {error.element_path}")
    print(f"Description   : {error.description}")
    print(f"Expected      : {error.expected_value}")
    print(f"Actual        : {error.saip_value}")
    print()
    print(f"Primary KO    : {ctx.primary_ko['ko_id'] if ctx.primary_ko else 'NOT FOUND'}")
    print(f"Section       : {ctx.section_id} — {ctx.section_title}")
    print(f"Parent PE     : {ctx.pe_ko['primary_label'] if ctx.pe_ko else 'N/A'}")
    print(f"Type ref      : {ctx.type_ko['primary_label'] if ctx.type_ko else 'N/A'}")
    print(f"Requirements  : {len(ctx.requirements)}")
    print(f"Spec expected : {ctx.found_expected_value}")
    print()
    print("--- Context Text ---")
    print(ctx.to_context_text()[:1200])


def transform_and_print(error_dict: dict, pipeline: ErrorTransformPipeline):
    """Transform one error and print formatted output."""
    print("\n" + "=" * 65)
    result = pipeline.transform_one(error_dict)
    print(result.format_display())
    print("=" * 65)
    return result


def main():
    parser = argparse.ArgumentParser(description="eUICC Error Transformer")
    parser.add_argument("--index",   default="output/index", help="Index directory")
    parser.add_argument("--error",   default=None, help="Single error as JSON string")
    parser.add_argument("--csv",     default=None, help="CSV file with multiple errors")
    parser.add_argument("--sample",  action="store_true", help="Use built-in sample errors")
    parser.add_argument("--test",    action="store_true", help="Test mode: show context only")
    parser.add_argument("--api-key", default="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", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--output",  default=None, help="Output JSON file path")
    args = parser.parse_args()

    # Load index
    index = load_index(args.index)

    # Load supporting data
    with open("output/sections.json", encoding="utf-8") as f:
        sections_data = json.load(f)
    with open("output/requirements.json", encoding="utf-8") as f:
        requirements_data = json.load(f)

    # Determine errors to process
    errors = []
    if args.sample:
        errors = SAMPLE_ERRORS
        print(f"Using {len(errors)} sample errors")
    elif args.error:
        errors = [json.loads(args.error)]
    elif args.csv:
        import csv
        with open(args.csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            errors = list(reader)
        print(f"Loaded {len(errors)} errors from {args.csv}")
    else:
        print("No input specified. Using sample errors.")
        errors = SAMPLE_ERRORS

    if args.test:
        for err in errors[:1]:
            test_mode(err, index, sections_data, requirements_data)
        return

    # Transform
    print(f"\nTransforming {len(errors)} error(s)...\n")
    pipeline = ErrorTransformPipeline(index, sections_data, requirements_data, api_key=args.api_key)
    results  = pipeline.transform_batch(errors)

    # Display
    print("\n" + "=" * 65)
    print("HASIL TRANSFORMASI")
    print("=" * 65)
    for r in results:
        print()
        print(r.format_display())
        print("-" * 65)

    # Save output
    if args.output:
        out_data = [r.to_dict() for r in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved to {args.output}")

    return results


if __name__ == "__main__":
    main()