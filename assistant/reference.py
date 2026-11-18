"""Search project documentation by heading and return the source section with each result."""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from paths import PROJECT_ROOT

CORPUS = (
    "ERGONOMICS_METHODS.md",
    "README.md",
    "PROJECT_ARCHITECTURE.md",
    "INSTALL.md",
)

_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
_WORD = re.compile(r"[a-z0-9]+")

#: Terms that appear in nearly every section of an ergonomics document carry no
#: discriminating power, and leaving them in makes every query match the longest
#: section rather than the relevant one.
_STOP = frozenset(
    """a an and are as at be by for from how in into is it its of on or that the this to what
    when where which who why with not no does do can could should would will may""".split()
)

#: Dropped from the *query* only, never from the index.
#:
#: These are the words that frame a question rather than name its subject. The
#: cost of leaving them in is not theoretical: "what does coupling mean in REBA"
#: ranked the temporal-aggregation section first, because "mean" is the
#: statistics word and it is rare enough in this corpus (5 sections) to carry a
#: higher idf than "coupling" itself. Rarity is not importance when the rare word
#: is a different sense of a common one. Someone actually asking about the
#: statistic writes "duration weighted mean", and those two words carry it.
_QUERY_STOP = frozenset(
    """mean means meaning explain describe tell show give define definition
    difference between about regarding concerning""".split()
)

#: A section longer than this is truncated in the result. Sections here run to a
#: few hundred words; the cap only guards against a pathological one.
MAX_SECTION_CHARS = 2600


@lru_cache(maxsize=1)
def _sections() -> tuple[tuple[str, str, str, Counter[str]], ...]:
    """(document, heading path, body, term counts) for every heading in the corpus."""

    def entry(
        document: str, heading: str, lines: list[str]
    ) -> tuple[str, str, str, Counter[str]] | None:
        """One section, or None when it is too short to be a useful answer."""

        body = "\n".join(lines).strip()
        if len(body) < 40:
            return None
        terms = Counter(
            word for word in _WORD.findall(f"{heading} {body}".lower()) if word not in _STOP
        )
        return document, heading, body, terms

    found: list[tuple[str, str, str, Counter[str]]] = []
    for name in CORPUS:
        path = PROJECT_ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        stack: list[str] = []
        heading = name
        buffer: list[str] = []

        for line in text.splitlines():
            match = _HEADING.match(line)
            if not match:
                buffer.append(line)
                continue
            section = entry(name, heading, buffer)
            if section is not None:
                found.append(section)
            buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            heading = " › ".join(stack)

        section = entry(name, heading, buffer)
        if section is not None:
            found.append(section)
    return tuple(found)


def search_reference(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Return the best-matching documentation sections with their citations."""

    sections = _sections()
    if not sections:
        return []
    words = _WORD.findall(str(query).lower())
    terms = [word for word in words if word not in _STOP and word not in _QUERY_STOP]
    # A question made entirely of framing words still deserves an answer, so the
    # narrower filter is only applied while it leaves something to search for.
    if not terms:
        terms = [word for word in words if word not in _STOP]
    if not terms:
        return []

    # Plain idf-weighted overlap, length-normalised. A section that mentions
    # "coupling" once out of forty words is a better answer about coupling than
    # one that mentions it once out of four hundred.
    total = len(sections)
    document_frequency: Counter[str] = Counter()
    for _, _, _, counts in sections:
        for term in set(terms):
            if counts.get(term):
                document_frequency[term] += 1

    # Okapi BM25. The earlier length normalisation was ad hoc and it showed:
    # "what does coupling mean in REBA" returned the temporal-aggregation
    # section, because a long section that mentions a term once outscored a short
    # one that is about it. BM25's saturation (k1) and pivoted length norm (b)
    # are the standard fix and they behave correctly on a corpus this small.
    lengths = [max(sum(counts.values()), 1) for _, _, _, counts in sections]
    average_length = sum(lengths) / len(lengths)
    k1, b = 1.5, 0.75

    idfs = {
        term: math.log(
            1 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
        )
        for term in set(terms)
    }

    scored: list[tuple[float, int]] = []
    for index, (_, heading, _, counts) in enumerate(sections):
        length = lengths[index]
        score = 0.0
        for term in terms:
            hits = counts.get(term, 0)
            if not hits:
                continue
            saturated = (hits * (k1 + 1)) / (
                hits + k1 * (1 - b + b * length / average_length)
            )
            score += idfs[term] * saturated
        if score <= 0:
            continue
        # A section whose own heading names the term is almost always the one a
        # reader wanted; "REBA 2000 › Posture and adjustment rules" should beat a
        # passing mention of REBA inside a section about something else.
        heading_words = {word for word in _WORD.findall(heading.lower()) if word not in _STOP}
        overlap = len(heading_words & set(terms))
        scored.append((score * (1.0 + 0.45 * overlap), index))

    scored.sort(key=lambda item: (-item[0], item[1]))
    results = []
    for _, index in scored[: max(1, int(limit))]:
        document, heading, body, _ = sections[index]
        results.append(
            {
                "citation": f"{document} § {heading}",
                "document": document,
                "heading": heading,
                "text": body[:MAX_SECTION_CHARS],
            }
        )
    return results


__all__ = ["CORPUS", "search_reference"]
