"""Token-safe embedding passages; original corpus items remain display records."""

import hashlib
import re

TARGET_WORDS = 200
MAX_WORDS = 240
OVERLAP_WORDS = 40


def token_count(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])


def passage_spans(text, tokenizer, token_limit):
    """Prefer paragraph boundaries near 200 words, subject to a hard token cap."""
    words = list(re.finditer(r"\S+", text))
    boundaries = {i for i in range(1, len(words))
                  if "\n\n" in text[words[i - 1].end():words[i].start()]}
    start = 0
    while start < len(words):
        low, high = start + 1, min(start + MAX_WORDS, len(words))
        end = start
        while low <= high:
            middle = (low + high) // 2
            value = text[words[start].start():words[middle - 1].end()]
            if token_count(value, tokenizer) <= token_limit:
                end, low = middle, middle + 1
            else:
                high = middle - 1
        if end == start:
            raise ValueError("A single word exceeds the embedding model token limit")
        if end < len(words):
            candidates = [b for b in boundaries if start + min(180, end - start) <= b <= end]
            if candidates:
                end = min(candidates, key=lambda b: (abs(b - start - TARGET_WORDS), b))
            elif end - start > 220:
                end = start + TARGET_WORDS
        value = text[words[start].start():words[end - 1].end()]
        # Verify after boundary selection as tokenizer segmentation can vary at edges.
        while token_count(value, tokenizer) > token_limit and end > start + 1:
            end -= 1
            value = text[words[start].start():words[end - 1].end()]
        yield start, end, value
        if end == len(words):
            break
        start = max(start + 1, end - OVERLAP_WORDS)


def embedding_items(corpus, encoder):
    items = []
    tokenizer, limit = encoder.tokenizer, encoder.max_seq_length
    for parent_index, parent in enumerate(corpus.items):
        metadata = parent.metadata
        if parent.item_type == "research_project":
            spans = [(0, len(parent.text.split()), parent.text)]
        else:
            spans = passage_spans(parent.text, tokenizer, limit)
        for passage_index, (start, end, text) in enumerate(spans):
            tokens = token_count(text, tokenizer)
            if tokens > limit:
                raise ValueError(f"Project {metadata.get('project_id')} exceeds the model token limit")
            parent_id = metadata.get("chunk_id", metadata["document_id"])
            items.append({
                "passage_id": hashlib.sha256(f"{parent_id}\0{start}\0{end}\0{text}".encode()).hexdigest(),
                "parent_item_index": parent_index, "parent_chunk_id": metadata.get("chunk_id"),
                "document_id": metadata["document_id"], "title": metadata["title"],
                "category": metadata.get("category"), "source_url": metadata.get("source_url"),
                "local_filename": metadata.get("local_filename"), "page_number": metadata.get("page_number"),
                "item_type": parent.item_type, "text": text, "provenance": parent.provenance,
                "passage_index": passage_index, "start_word": start, "end_word": end,
                "word_count": end - start, "token_count": tokens,
            })
    return items
