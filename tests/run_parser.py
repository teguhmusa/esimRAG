# run_parser.py
"""
Full pipeline: PDF → Sections → Tables → Requirements → Entities
             → Relationships → Knowledge Objects → Hybrid Index
"""

import json, argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import (
    PDFParser, SectionExtractor, TableExtractor,
    RequirementExtractor,
    EntityExtractor, print_entity_stats_obj,
    RelationshipExtractor, print_relationship_stats,
    KnowledgeObjectBuilder, print_ko_stats,
    HybridIndex,
)


def run_pipeline(pdf_path, output_dir="output", page_start=1, page_end=None):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    print(f"📄 Parsing: {pdf_path}")

    with PDFParser(pdf_path) as parser:
        total_pages = parser.page_count
        page_end = page_end or total_pages
        print(f"   Total pages: {total_pages} | Processing: {page_start}–{page_end}")

        print("\nStep 1: PDF Parser...")
        pages = parser.parse_all(start=page_start, end=page_end)
        print(f"   Parsed {len(pages)} pages")

        print("\nStep 2: Section Extractor...")
        sec_extractor = SectionExtractor(skip_toc_pages=6)
        sections = sec_extractor.extract(pages)
        section_index = sec_extractor.build_index(sections)
        print(f"   Found {len(sections)} sections")

        print("\nStep 3: Table Extractor...")
        tbl_extractor = TableExtractor()
        tables = tbl_extractor.extract(pages)
        tbl_extractor.assign_sections(tables, section_index)
        print(f"   Found {len(tables)} tables")

        print("\nStep 4: Requirement Extractor...")
        sections_data = [s.to_dict() for s in sections]
        req_extractor = RequirementExtractor()
        requirements  = req_extractor.extract_from_sections(sections_data)
        req_data      = [r.to_dict() for r in requirements]
        print(f"   Found {len(req_data)} requirements")

        print("\nStep 5: Entity Extractor...")
        asn1_all = [
            {"type_name": b.type_name, "asn1_type": b.asn1_type,
             "section_id": b.section_id, "page_num": b.page_num, "content": b.content}
            for s in sections for b in s.asn1_blocks
        ]
        tables_data   = [t.to_dict() for t in tables]
        ent_extractor = EntityExtractor()
        entities      = ent_extractor.extract_all(asn1_all, tables_data, req_data)
        print_entity_stats_obj(entities)
        entities_data = {
            "type_defs":        [t.to_dict() for t in entities["type_defs"]],
            "ef_files":         [e.to_dict() for e in entities["ef_files"]],
            "error_codes":      [e.to_dict() for e in entities["error_codes"]],
            "validation_rules": [v.to_dict() for v in entities["validation_rules"]],
        }

        print("\nStep 6: Relationship Extractor...")
        rel_extractor = RelationshipExtractor()
        relationships = rel_extractor.extract_all(entities_data, sections_data, req_data)
        rels_data     = [r.to_dict() for r in relationships]
        print_relationship_stats(relationships)

        print("\nStep 7: Knowledge Object Builder...")
        ko_builder       = KnowledgeObjectBuilder()
        knowledge_objects = ko_builder.build(entities_data, rels_data, sections_data, req_data)
        ko_data          = [ko.to_dict() for ko in knowledge_objects]
        print_ko_stats(knowledge_objects)

        print("\nStep 8: Hybrid Index...")
        index = HybridIndex()
        index.build(ko_data, rels_data)
        index.stats()
        index_dir = out / "index"
        index.save(index_dir)

        # Save JSON outputs
        print("\nSaving outputs...")
        _save(out / "sections.json",          sections_data,  "sections")
        _save(out / "tables.json",            tables_data,    "tables")
        _save(out / "asn1_blocks.json",       asn1_all,       "ASN.1 blocks")
        _save(out / "requirements.json",      req_data,       "requirements")
        _save(out / "entities.json",          entities_data,  "entities")
        _save(out / "relationships.json",     rels_data,      "relationships")
        _save(out / "knowledge_objects.json", ko_data,        "knowledge objects")

        print("\nDone!")
        return index


def _save(path, data, label):
    with open(path, "w", encoding="utf-8") as f:
        total = sum(len(v) for v in data.values()) if isinstance(data, dict) else len(data)
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   {path.name:<32} → {total} records")


def run_diagnostics(pdf_path, page_num=46):
    print(f"\nDiagnostics for page {page_num}\n")
    with PDFParser(pdf_path) as parser:
        page = parser.parse_page(page_num)
    for b in page.text_blocks:
        print(f"  [{b.block_type.upper():8s}] {b.text[:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--output", default="output")
    parser.add_argument("--pages",  default=None)
    parser.add_argument("--diag",   type=int, default=None)
    args = parser.parse_args()

    if args.diag:
        run_diagnostics(args.pdf, args.diag)
    else:
        start, end = 1, None
        if args.pages:
            parts = args.pages.split("-")
            start = int(parts[0])
            end = int(parts[1]) if len(parts) > 1 else None
        run_pipeline(args.pdf, output_dir=args.output, page_start=start, page_end=end)