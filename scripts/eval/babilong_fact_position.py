"""Locate bAbILong gold-fact sentences in raw context text by character offset.

bAbILong (RMT-team/babilong) embeds task facts as short templated sentences
(e.g. "Mary journeyed to the bathroom.") inside PG19 distractor text. There is
no ground-truth offset in the dataset, so this module recovers it heuristically.
Used purely for post-hoc position bucketing (experiment 1) and multi-fact
distance (experiment 2) -- it does not change what text the model sees.

Three task shapes need different logic:
  qa1 "Where is Mary?"                 -- single person entity, one fact.
  qa2 "Where is the football?"         -- object noun, two-hop: find who last
       interacted with the object, then that actor's last location move.
  qa3 "Where was the apple before X?"  -- object noun + anchor location, three
       facts: find the actor via the object, then the actor's location-move
       immediately before the one matching X.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

_ENTITY_RE = re.compile(r"[A-Z][a-z]+")
_ACTORS = {"Daniel", "John", "Mary", "Sandra"}
_MOVE_RE = re.compile(
    r"(Daniel|John|Mary|Sandra)\b[^.]{0,25}?\bto the ([a-z]+)"
)
_OBJECT_Q_RE = re.compile(r"the ([a-z]+)\?")
_BEFORE_LOC_RE = re.compile(r"before the ([a-z]+)\?")


def _question_entity(question: str) -> Optional[str]:
    """Best-effort extraction of the person entity a qa1 question asks about."""
    q = question.strip().rstrip("?").strip()
    names = _ENTITY_RE.findall(q)
    return names[-1] if names else None


def _question_object(question: str) -> Optional[str]:
    """Extract the object noun a qa2/qa3 question asks about (e.g. "football")."""
    m = _OBJECT_Q_RE.search(question)
    return m.group(1) if m else None


def _all_moves(input_text: str) -> List[Tuple[int, str, str]]:
    """All (offset, actor_name, location) location-move mentions, in text order."""
    return [(m.start(), m.group(1), m.group(2)) for m in _MOVE_RE.finditer(input_text)]


def _object_actor(input_text: str, obj: str) -> Optional[Tuple[int, str]]:
    """Find the actor of the last sentence mentioning `obj`, and that sentence's offset."""
    last = None
    for m in re.finditer(re.escape(obj), input_text, re.IGNORECASE):
        # walk back to the start of the enclosing sentence to find its subject
        start = input_text.rfind(".", 0, m.start()) + 1
        sentence = input_text[start : m.end()]
        names = [n for n in _ENTITY_RE.findall(sentence) if n in _ACTORS]
        if names:
            last = (m.start(), names[0])
    return last


def _object_question_facts(
    input_text: str, question: str
) -> Optional[Tuple[List[Tuple[int, float]], int]]:
    """For qa2/qa3 (object-noun questions): return (all_hit_offsets, gold_index).

    `gold_index` indexes into the returned list and marks which hit is the one
    that actually supports the target answer (as opposed to just being part of
    the supporting chain) -- qa2's last actor-move, or qa3's before-anchor move.
    """
    obj = _question_object(question)
    if not obj:
        return None
    oa = _object_actor(input_text, obj)
    if not oa:
        return None
    obj_offset, actor = oa
    moves = [mv for mv in _all_moves(input_text) if mv[1] == actor]
    if not moves:
        return None
    n = max(1, len(input_text))
    hits: List[Tuple[int, float]] = [(obj_offset, obj_offset / n)]

    before_m = _BEFORE_LOC_RE.search(question)
    if before_m:
        anchor_loc = before_m.group(1)
        anchor_idx = None
        for i in range(len(moves) - 1, -1, -1):
            if moves[i][2] == anchor_loc:
                anchor_idx = i
                break
        if anchor_idx is None or anchor_idx == 0:
            return None
        before_offset = moves[anchor_idx - 1][0]
        anchor_offset = moves[anchor_idx][0]
        hits.append((before_offset, before_offset / n))
        hits.append((anchor_offset, anchor_offset / n))
        gold_offset = before_offset
    else:
        last_offset = moves[-1][0]
        hits.append((last_offset, last_offset / n))
        gold_offset = last_offset

    seen = set()
    out = []
    for off, frac in sorted(hits, key=lambda x: x[0]):
        if off not in seen:
            seen.add(off)
            out.append((off, frac))
    gold_index = next(i for i, (off, _) in enumerate(out) if off == gold_offset)
    return out, gold_index


def locate_gold_fact(
    input_text: str, question: str, target: str
) -> Optional[Tuple[int, float]]:
    """Return (char_offset, frac_position) of the single gold fact sentence, or None.

    qa1: last "<name>...<target>" mention.
    qa2: two-hop via object -> actor -> actor's last location move.
    qa3: three-hop; gold is the actor's move immediately before the anchor.
    """
    result = _object_question_facts(input_text, question)
    if result is not None:
        hits, gold_index = result
        return hits[gold_index]
    if _question_object(question):
        return None  # object question but hop chain failed to resolve

    name = _question_entity(question)
    if not name or not target:
        return None
    pat = re.compile(re.escape(name) + r"[^.]{0,40}" + re.escape(target), re.IGNORECASE)
    last = None
    for m in pat.finditer(input_text):
        last = m  # bAbILong qa1: last matching mention is the answer-supporting one
    if last is None:
        return None
    offset = last.start()
    return offset, offset / max(1, len(input_text))


def locate_all_facts(
    input_text: str, question: str, target: str, max_entities: int = 4
) -> List[Tuple[int, float]]:
    """Return offsets of all plausible supporting-fact sentences for qa2/qa3.

    qa2: (object-mention offset, actor's-last-move offset) -- two hops.
    qa3: (object-mention offset, actor's-move-before-X offset,
          actor's-move-to-X offset) -- three hops.
    Falls back to the qa1 single/multi-entity scan when there's no object noun
    in the question. Best-effort: callers should tolerate an empty list.
    """
    result = _object_question_facts(input_text, question)
    if result is not None:
        return result[0]
    if _question_object(question):
        return []

    q_entities = _ENTITY_RE.findall(question)[:max_entities]
    if not q_entities:
        return []
    hits: List[Tuple[int, float]] = []
    seen_offsets = set()
    for name in q_entities:
        pat = re.compile(re.escape(name) + r"[^.]{0,60}", re.IGNORECASE)
        for m in pat.finditer(input_text):
            if target and target.lower() in m.group().lower():
                off = m.start()
                if off not in seen_offsets:
                    seen_offsets.add(off)
                    hits.append((off, off / max(1, len(input_text))))
    hits.sort(key=lambda x: x[0])
    return hits


def position_bucket(frac: float) -> str:
    if frac < 1 / 3:
        return "begin"
    if frac < 2 / 3:
        return "middle"
    return "end"


def fact_span(offsets: List[Tuple[int, float]]) -> Optional[float]:
    """Fractional distance between the first and last located fact (qa2/qa3)."""
    if len(offsets) < 2:
        return None
    fracs = [f for _, f in offsets]
    return max(fracs) - min(fracs)


def extract_evidence_sentences(input_text: str, offsets: List[Tuple[int, float]]) -> str:
    """Concatenate the sentences containing each located fact, in text order, deduped.

    Used for experiment 2's short-evidence-only control: same supporting facts,
    without the surrounding distractor text.
    """
    spans = []
    for off, _ in offsets:
        start = input_text.rfind(".", 0, off) + 1
        end = input_text.find(".", off)
        end = end + 1 if end != -1 else len(input_text)
        spans.append((start, end))
    merged: List[Tuple[int, int]] = []
    for s, e in sorted(set(spans)):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return " ".join(input_text[s:e].strip() for s, e in merged)
