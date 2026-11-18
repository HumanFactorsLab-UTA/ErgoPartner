"""Resolve a turn's clickable evidence against measurements read and images delivered."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

_FRAME = re.compile(r"\bframe\s*[#:=]?\s*(\d+)\b", re.I)
_TAG = re.compile(r"\[(computed|observed)\b([^\]]*)\]", re.I)
_CAMERA = re.compile(r"\bcamera\s*[#:=]?\s*(\d+)\b", re.I)
_METHOD = re.compile(r"\b(rula|reba)\b", re.I)


@dataclass(frozen=True, slots=True)
class Citation:
    frame: int | None = None
    sec: float | None = None
    method: str | None = None
    region: str | None = None
    camera: int | None = None
    person: str | None = None
    kind: str | None = None
    view: str | None = None

    def payload(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass
class TurnEvidence:
    sources: list[Citation] = field(default_factory=list)
    measurements: dict[Citation, dict] = field(default_factory=dict)

    def record(self, content: dict, selection: dict) -> None:
        """Only structured tool rows can establish a measured frame."""
        method = str(content.get("method") or selection.get("method") or "").lower()
        rows = [
            content,
            *content.get("rows", []),
            *content.get("episodes", []),
            *content.get("periods", []),
            *content.get("regions", []),
        ]
        for row in rows:
            frame = row.get("frame", row.get("peakFrame"))
            if type(frame) is not int or method not in {"rula", "reba"}:
                continue
            ref = Citation(
                frame=frame,
                sec=row.get("timestampSec", row.get("peakSec")),
                method=method,
                person=row.get("person") or content.get("person") or selection.get("person"),
                kind="computed",
            )
            self.sources.append(ref)
            # Some tools return a subset of the same frame. Keep the full receipt.
            self.measurements.setdefault(ref, {}).update(row)

    def image(self, item: dict) -> None:
        self.sources.append(
            Citation(
                frame=item["frame"],
                sec=item.get("sec"),
                method=item.get("method"),
                region=item.get("region"),
                camera=item["camera"],
                person=item.get("person"),
                kind="observed",
                view=item.get("view"),
            )
        )

    def resolve(self, requested: Citation, selection: dict) -> Citation | None:
        choices = [
            ref
            for ref in self.sources
            if ref.frame == requested.frame
            and all(
                getattr(requested, k) is None or getattr(ref, k) == getattr(requested, k)
                for k in ("kind", "method", "person", "camera", "view")
            )
            and (
                ref.kind != "observed"
                or requested.region is None
                or ref.region is None
                or requested.region == ref.region
            )
        ]
        if not choices:
            return None
        if requested.kind is None:
            # Untyped footers commonly accompany an observed inline tag. Prefer the
            # actual image when it is the only view backing this frame.
            choices = [r for r in choices if r.kind == "observed"] or choices
        for key in ("method", "person"):
            matching = [r for r in choices if getattr(r, key) == selection.get(key)]
            choices = matching or choices
        if len({(r.method, r.camera, r.person) for r in choices}) > 1:
            return None
        # A raw image and a focus crop can coexist. An unqualified link opens the full view.
        chosen = next((r for r in choices if r.view == "raw"), choices[-1])
        return chosen if chosen.kind == "observed" else replace(chosen, region=requested.region)

    def citations(
        self, text: str, footers: list[Citation], selection: dict
    ) -> tuple[list[Citation], list[str]]:
        resolved: list[Citation] = []
        problems: list[str] = []
        for match in _TAG.finditer(text):
            kind, body = match.group(1).lower(), match.group(2)
            frame_match = _FRAME.search(body)
            if not frame_match:
                if kind == "observed":
                    problems.append("Name the delivered image's frame in the observed tag.")
                continue
            frame = int(frame_match[1])
            camera = _CAMERA.search(body)
            method = _METHOD.search(body)
            requested = Citation(
                frame=frame,
                kind=kind,
                method=method[1].lower() if method else None,
                camera=int(camera[1]) if camera else None,
            )
            # An explicit footer can disambiguate one of several images.
            matching = [
                f
                for f in footers
                if f.frame == frame
                and f.kind in (None, kind)
                and (requested.camera is None or f.camera in (None, requested.camera))
                and (requested.method is None or f.method in (None, requested.method))
            ]
            if len(matching) == 1:
                f = matching[0]
                requested = replace(
                    requested,
                    camera=requested.camera if camera else f.camera,
                    method=requested.method or f.method,
                    region=f.region,
                    person=f.person,
                    view=f.view,
                )
            ref = self.resolve(requested, selection)
            if ref:
                resolved.append(ref)
            else:
                problems.append(
                    f"No matching {kind} evidence for frame {frame} and the cited view."
                )
        for f in footers:
            ref = self.resolve(f, selection)
            if ref is None:
                problems.append(f"The citation to frame {f.frame} does not match a tool result.")
            else:
                resolved.append(ref)
        return list(dict.fromkeys(resolved)), list(dict.fromkeys(problems))

    def check_measurements(self, text: str, citations: list[Citation]) -> list[str]:
        """Check explicit frame score and angle statements against their typed records."""
        issues: list[str] = []
        start = 0
        for tag in _TAG.finditer(text):
            segment, start = text[start : tag.start()], tag.end()
            frame = _FRAME.search(tag[2])
            if tag[1].lower() != "computed" or not frame:
                continue
            method = _METHOD.search(tag[2])
            refs = [
                r
                for r in citations
                if r.kind == "computed"
                and r.frame == int(frame[1])
                and (not method or r.method == method[1].lower())
            ]
            statement = _FRAME.sub("", segment.lower()).replace("*", "")
            for ref in refs:
                records = [
                    row
                    for src, row in self.measurements.items()
                    if (src.frame, src.method, src.person) == (ref.frame, ref.method, ref.person)
                ]
                record = next(
                    (r for r in records if "angles" in r),
                    next((r for r in records if "score" in r or "peakScore" in r), {}),
                )
                scores = record.get("sides") or {}
                for match in re.finditer(
                    r"\b(?:(rula|reba)(?:\s+(?:final\s+)?score)?|(?:final\s+)?score)"
                    r"(?:\s+(?:at|of|is|was|equals|reached|totalled)|[\s:=])+"
                    r"(\d+(?:\.\d+)?)\b",
                    statement,
                ):
                    if match[1] and match[1] != ref.method:
                        if not any(other.method == match[1] for other in refs):
                            issues.append(
                                f"Retrieve the {match[1].upper()} record for frame {ref.frame}."
                            )
                        continue
                    preceding = statement[max(0, match.start() - 24) : match.start()]
                    following = statement[match.end() : match.end() + 36]
                    side = re.search(r"\b(left|right)(?:[- ](?:side|sided))?\s*$", preceding)
                    side = side or re.match(r"\s+(?:on|for)\s+(?:the\s+)?(left|right)\b", following)
                    expected = (
                        scores.get(side[1], {}).get("score")
                        if side
                        else record.get("score", record.get("peakScore"))
                    )
                    if float(match[2]) != expected:
                        issues.append(
                            f"Check the {ref.method} score and side for frame {ref.frame}."
                        )
                if ("both sides" in statement or "both the left and right" in statement) and scores:
                    if scores.get("left", {}).get("score") != scores.get("right", {}).get("score"):
                        issues.append(f"The scores for the two sides at frame {ref.frame} differ.")
                for angle in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?\b)", statement):
                    prefix = re.split(r"[.;\n]", statement[: angle.start()])[-1][-90:]
                    side = re.findall(r"\b(left|right)\b", prefix)
                    words = re.findall(r"[a-z]+", prefix.replace("flexed", "flexion"))
                    camera = _CAMERA.search(prefix)
                    angles = (
                        record.get("anglesByCamera", {}).get(camera[1], {})
                        if camera
                        else record.get("angles", {})
                    )
                    candidates = []
                    for name, value in angles.items():
                        terms = name.lower().split("_")
                        if side and terms[0] in {"left", "right"} and terms[0] != side[-1]:
                            continue
                        positions = [i for i, word in enumerate(words) if word in terms]
                        if positions and isinstance(value, (int, float)):
                            candidates.append(((positions[-1], len(positions)), value))
                    target = float(angle[1])
                    decimals = len(angle[1].partition(".")[2])
                    rank = max((r for r, _ in candidates), default=None)
                    if not any(round(v, decimals) == target for r, v in candidates if r == rank):
                        issues.append(
                            f"Look up the named joint, side and angle for frame {ref.frame}."
                        )
        return list(dict.fromkeys(issues))
