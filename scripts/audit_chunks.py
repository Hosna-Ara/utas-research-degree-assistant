"""Read-only corpus audit: python scripts/audit_chunks.py."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
import re
import sys

from utas_research_assistant.retrieval.chunking import ChunkRecord

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed"
SEED = 42
BOILERPLATE = re.compile(r"\b(?:Share\s+this|Bluesky|Twitter|Facebook|LinkedIn|Cookies?)\b", re.I)


def suffix_overlap(first: list[str], second: list[str]) -> int:
    """Measure exact trailing/leading word overlap, without assuming its size."""
    for size in range(min(len(first), len(second)), 0, -1):
        if first[-size:] == second[:size]:
            return size
    return 0


def reference(row_number: int, row: dict) -> dict:
    return {"row": row_number, **{key: row[key] for key in (
        "chunk_id", "document_id", "local_filename", "chunk_index",
    )}}


def audit(rows: list[dict]) -> dict:
    # Fail clearly on malformed records, before writing a potentially misleading report.
    for index, row in enumerate(rows, 1):
        try:
            ChunkRecord.model_validate(row)
        except ValueError as exc:
            raise ValueError(f"Invalid chunk at row {index}: {exc}") from exc
    sizes = [len(row["text"].split()) for row in rows]
    documents, texts, ids = defaultdict(list), defaultdict(list), defaultdict(list)
    findings = {key: [] for key in (
        "below_200_words", "above_500_words", "empty_chunks", "boilerplate_matches",
        "standalone_table_separators", "suspicious_beginnings", "suspicious_endings",
        "word_count_mismatches",
    )}
    for index, (row, size) in enumerate(zip(rows, sizes), 1):
        ref = reference(index, row)
        content = row["text"].strip()
        documents[row["document_id"]].append((index, row))
        texts[" ".join(content.split())].append(ref)
        ids[row["chunk_id"]].append(ref)
        for flag, condition in (("below_200_words", size < 200), ("above_500_words", size > 500), ("empty_chunks", not content)):
            if condition:
                findings[flag].append({**ref, "word_count": size})
        if size != row["word_count"]:
            findings["word_count_mismatches"].append({**ref, "stored": row["word_count"], "actual": size})
        matches = sorted({match.group().casefold() for match in BOILERPLATE.finditer(content)})
        if matches:
            findings["boilerplate_matches"].append({**ref, "terms": matches})
        if re.search(r"^\s*\|(?:\s*\|)*\s*$", content, re.M):
            findings["standalone_table_separators"].append(ref)
        if content and (content[0].islower() or content[0] in "|,;:)]}"):
            findings["suspicious_beginnings"].append({**ref, "excerpt": content[:160]})
        if content and not re.search(r"[.!?][\"'’”\)\]]*$", content):
            findings["suspicious_endings"].append({**ref, "excerpt": content[-160:]})

    per_document, adjacent = [], []
    for document_id, items in documents.items():
        items.sort(key=lambda item: item[1]["chunk_index"])
        first, last = items[0][1], items[-1][1]
        per_document.append({
            "document_id": document_id, "title": first["title"],
            "local_filename": first["local_filename"], "number_of_chunks": len(items),
            "first_chunk_word_count": len(first["text"].split()),
            "last_chunk_word_count": len(last["text"].split()),
        })
        for (left_index, left), (right_index, right) in zip(items, items[1:]):
            # Compare neighbours by chunk index only within the same document/page.
            left_words, right_words = left["text"].split(), right["text"].split()
            overlap = suffix_overlap(left_words, right_words)
            ratio = overlap / min(len(left_words), len(right_words)) if left_words and right_words else 0
            similarity = SequenceMatcher(None, left_words, right_words, autojunk=False).ratio()
            adjacent.append({
                "left": reference(left_index, left), "right": reference(right_index, right),
                "overlap_words": overlap, "overlap_fraction_of_shorter_chunk": round(ratio, 4),
                "token_sequence_similarity": round(similarity, 4),
                "very_high": ratio >= 0.5 or similarity >= 0.7,
            })

    duplicate_texts = [refs for content, refs in texts.items() if content and len(refs) > 1]
    duplicate_ids = [refs for refs in ids.values() if len(refs) > 1]
    return {
        "source_file": "data/processed/general_chunks.json",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "word_counts": "Recomputed using whitespace-delimited words.",
            "duplicate_texts": "Case-sensitive text with whitespace collapsed; blank text excluded.",
            "boilerplate": "Whole-word matches anywhere, including meaningful prose; inspect before removing.",
            "boundaries": "Lowercase or continuation punctuation at start; no sentence-ending punctuation at end. Headings, lists, tables, and intentional overlap may trigger false positives.",
            "adjacent": "Within document, ordered by chunk_index; exact suffix/prefix overlap and difflib token sequence similarity (autojunk disabled).",
            "very_high": "Overlap >=50% of the shorter chunk OR token sequence similarity >=0.70. Expected overlap is 70 words; small final chunks may be flagged.",
            "sample_seed": SEED,
        },
        "total_chunks": len(rows),
        "chunks_by_category": dict(Counter(row["category"] for row in rows)),
        "min_word_count": min(sizes, default=0),
        "average_word_count": round(sum(sizes) / len(sizes), 2) if sizes else 0,
        "max_word_count": max(sizes, default=0),
        "finding_counts": {key: len(value) for key, value in findings.items()},
        "duplicate_text_groups": duplicate_texts,
        "duplicate_text_group_count": len(duplicate_texts),
        "duplicate_text_excess_chunks": sum(len(refs) - 1 for refs in duplicate_texts),
        "duplicate_id_groups": duplicate_ids,
        "duplicate_id_group_count": len(duplicate_ids),
        "findings": findings,
        "documents": per_document,
        "adjacent_pairs": adjacent,
        "high_similarity_pairs": [pair for pair in adjacent if pair["very_high"]],
        "high_similarity_pair_count": sum(pair["very_high"] for pair in adjacent),
        "manual_sample_rows": random.Random(SEED).sample(list(range(1, len(rows) + 1)), min(5, len(rows))),
    }


def main() -> int:
    try:
        rows = json.loads((PROCESSED / "general_chunks.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("general_chunks.json must contain an array")
        report = audit(rows)
        (PROCESSED / "chunk_quality_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"Chunk audit failed: {exc}", file=sys.stderr)
        return 1
    keys = (
        "total_chunks", "chunks_by_category", "min_word_count", "average_word_count",
        "max_word_count", "finding_counts", "duplicate_text_group_count",
        "duplicate_text_excess_chunks", "duplicate_id_group_count", "high_similarity_pair_count",
    )
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    print("\nSource documents:")
    for document in report["documents"]:
        print(json.dumps(document, ensure_ascii=False))
    print(f"\nFive complete sample chunks (seed {SEED}):")
    for index in report["manual_sample_rows"]:
        print(json.dumps(rows[index - 1], ensure_ascii=False, indent=2))
    print("\nSaved: data/processed/chunk_quality_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
