"""N-polar streaming sampler for MTG-Jamendo-style records."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Iterator


TAG_FIELDS = ("tags", "tag", "genre", "genres", "mood", "moods")


def normalize_tag(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def extract_record_tags(record: dict) -> set[str]:
    tags: set[str] = set()
    for field in TAG_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            tags.add(normalize_tag(value))
            parts = value.replace(",", " ").replace(";", " ").split()
            tags.update(normalize_tag(part) for part in parts)
        elif isinstance(value, Iterable):
            tags.update(normalize_tag(part) for part in value)
    return {tag for tag in tags if tag}


def match_primary_tag(record: dict, target_tags: Iterable[str]) -> str | None:
    record_tags = extract_record_tags(record)
    for tag in target_tags:
        normalized = normalize_tag(tag)
        if normalized in record_tags:
            return normalized
    return None


def iter_n_polar_samples(
    stream: Iterable[dict],
    target_tags: Iterable[str],
    per_tag_limit: int,
) -> Iterator[tuple[str, dict]]:
    target_tags = [normalize_tag(tag) for tag in target_tags]
    counts: Counter[str] = Counter()

    for record in stream:
        if all(counts[tag] >= per_tag_limit for tag in target_tags):
            break
        primary_tag = match_primary_tag(record, target_tags)
        if primary_tag is None or counts[primary_tag] >= per_tag_limit:
            continue
        counts[primary_tag] += 1
        yield primary_tag, record
