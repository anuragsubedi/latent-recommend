"""N-polar streaming sampler for MTG-Jamendo-style records."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Iterator


TAG_FIELDS = ("tags", "tag", "genre", "genres", "mood", "moods")

DEFAULT_TAG_ALIASES = {
    "ambient_soundscape": {
        "ambient",
        "atmospheric",
        "soundscape",
        "meditative",
        "calm",
        "relaxing",
        "dream",
        "newage",
    },
    "electronic_dance": {
        "electronic",
        "electronica",
        "techno",
        "house",
        "deephouse",
        "dance",
        "club",
        "trance",
        "disco",
        "drummachine",
        "synth",
        "synthesizer",
    },
    "classical_orchestral": {
        "classical",
        "orchestral",
        "symphonic",
        "strings",
        "violin",
        "cello",
        "piano",
        "pianosolo",
        "contemporary",
    },
    "jazz_soul": {
        "jazz",
        "trumpet",
        "saxophone",
        "soul",
        "funk",
        "swing",
        "electricpiano",
    },
    "blues_roots": {
        "blues",
        "roots",
        "country",
        "reggae",
        "world",
    },
    "hiphop_rap": {
        "hiphop",
        "hip_hop",
        "rap",
        "rnb",
        "beat",
        "beats",
    },
    "folk_acoustic": {
        "folk",
        "acoustic",
        "acousticguitar",
        "popfolk",
        "guitar",
    },
    "indie_rock": {
        "indie",
        "alternative",
        "rock",
        "poprock",
        "instrumentalrock",
        "electricguitar",
        "punk",
        "metal",
    },
    "experimental_trip_hop": {
        "experimental",
        "triphop",
        "downtempo",
        "deep",
        "dark",
        "computer",
        "sampler",
    },
    "happy_pop": {
        "pop",
        "instrumentalpop",
        "happy",
        "playful",
        "summer",
        "melodic",
        "easylistening",
        "lounge",
    },
    # Backwards-compatible labels used in earlier extraction experiments.
    "ambient": {"ambient", "atmospheric", "soundscape", "meditative"},
    "electronic": {"electronic", "electronica", "techno", "house", "deephouse"},
    "classical": {"classical", "orchestral", "symphonic"},
    "jazz": {"jazz"},
    "blues": {"blues"},
    "rock": {"rock", "poprock", "instrumentalrock"},
    "acoustic_folk": {"folk", "acoustic", "acousticguitar", "popfolk"},
    "dub": {"dub"},
    "heavy_metal": {"metal"},
}


def normalize_tag(value: object) -> str:
    tag = str(value).strip().lower()
    if "---" in tag:
        tag = tag.split("---", 1)[1]
    return tag.replace(" ", "_").replace("-", "_")


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


def match_primary_tag(
    record: dict,
    target_tags: Iterable[str],
    tag_aliases: dict[str, set[str]] | None = None,
) -> str | None:
    record_tags = extract_record_tags(record)
    aliases = tag_aliases or DEFAULT_TAG_ALIASES
    for tag in target_tags:
        normalized = normalize_tag(tag)
        candidates = {normalized}
        candidates.update(normalize_tag(alias) for alias in aliases.get(normalized, set()))
        if record_tags & candidates:
            return normalized
    return None


def iter_n_polar_samples(
    stream: Iterable[dict],
    target_tags: Iterable[str],
    per_tag_limit: int,
    tag_aliases: dict[str, set[str]] | None = None,
) -> Iterator[tuple[str, dict]]:
    target_tags = [normalize_tag(tag) for tag in target_tags]
    counts: Counter[str] = Counter()

    for record in stream:
        if all(counts[tag] >= per_tag_limit for tag in target_tags):
            break
        primary_tag = match_primary_tag(record, target_tags, tag_aliases=tag_aliases)
        if primary_tag is None or counts[primary_tag] >= per_tag_limit:
            continue
        counts[primary_tag] += 1
        yield primary_tag, record
