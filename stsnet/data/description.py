"""
SSLL description parser for STS-Net.

Parses Swedish Sign Language Lexicon description strings into per-phase tuples
of (shape, attitude, hand_type, contact_loc, contact_type, motion, nondom_shape, nondom_att).
"""

import csv
import re
from pathlib import Path

from stsnet.data.contact import parse_contact


# ---------------------------------------------------------------------------
# Handshape vocabulary
# ---------------------------------------------------------------------------

def load_handshape_vocab(vocab_file: str | Path) -> list[str]:
    """Read sts_handformer.txt — one handshape name per line."""
    lines = Path(vocab_file).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


_IRREGULAR_PLURAL = {
    "pekfingrar":   "Pekfingret",
    "krokfingrar":  "Krokfingret",
    "lillfingrar":  "Lillfingret",
    "dubbelkrokar": "Dubbelkroken",
    "knuten hand":  "Knutna handen",
    "sprethand":    "Sprethanden",   # indefinite form used in "förändras till sprethand"
}

_SKIP_PREFIXES = ("h. h.",)

# Bokstaveras (fingerspelling) pattern: "Bokstaveras: A-B-C" or "Bokstaveras: 3-D"
# Tokens are letters (A-Z, ÅÄÖ) or digits (0-9).
_BOKSTAVERAS_TOKEN = r'[A-ZÅÄÖ0-9]'
_BOKSTAVERAS_RE = re.compile(
    r'^\s*Bokstaveras:\s*(' + _BOKSTAVERAS_TOKEN + r'(?:\s*-\s*' + _BOKSTAVERAS_TOKEN + r')*)\s*$',
    re.IGNORECASE,
)


def _normalise_shape(raw: str) -> str | None:
    s = raw.strip()
    if not s:
        return None
    sl = s.lower()
    for prefix in _SKIP_PREFIXES:
        if sl.startswith(prefix):
            return None
    if sl in _IRREGULAR_PLURAL:
        return _IRREGULAR_PLURAL[sl]
    s = re.sub(r'händer\b', 'handen', s, flags=re.IGNORECASE)
    # "Xhand" / "X-hand" → "Xhanden" / "X-handen" (indefinite → definite)
    # "hand\b" matches at word end; doesn't fire inside "handen" (ends in -en)
    s = re.sub(r'hand\b', 'handen', s, flags=re.IGNORECASE)
    return s


# Fingerspelling letter → canonical SSLL handshape name (original 42-item vocab).
# Letters already of the form "X-handen" need no entry here.
_LETTER_TO_SHAPE: dict[str, str] = {
    # Letters whose STS handshape matches an existing named shape
    "B": "Tumhanden",
    "C": "S-handen",
    "F": "Stora hållhanden",
    "G": "Knutna handen",
    "H": "Stora nyphanden",
    "I": "Lillfingret",
    "J": "Flata handen",
    "P": "Stora nyphanden",
    "Q": "Tumhanden",
    "R": "Långfingret",
    "Y": "Vinkelhanden",
    "Z": "Pekfingret",
    "Å": "A-handen",
    "Ä": "A-handen",
    "Ö": "O-handen",
    # Digits (STS number handshapes)
    "1": "Pekfingret",
    "2": "V-handen",
    "3": "W-handen",
    "4": "4-handen",
    "5": "Sprethanden",
    "6": "Tumhanden",
    "7": "L-handen",
    "8": "Tupphanden",
    "9": "Knutna handen",
}
# Letters that map directly to "X-handen" (in original vocab):
# A D E K L M N O S T U V W X


def _parse_bokstaveras(desc: str) -> list[tuple] | None:
    """Parse 'Bokstaveras: X-Y-Z' into one phase per letter.

    Maps each letter to its canonical SSLL handshape name using the original
    42-item vocab.  Letters not in _LETTER_TO_SHAPE default to "X-handen"
    (e.g. A→A-handen, D→D-handen).  Unrecognised tokens (digits etc.) are
    skipped — callers drop phases where the shape isn't in vocab.

    Returns a list of (shape, None, "one", None, None, None, None, None)
    tuples, or None if the description is not a Bokstaveras entry.
    """
    m = _BOKSTAVERAS_RE.match(desc.strip())
    if not m:
        return None
    letters = [l.strip().upper() for l in m.group(1).split('-')]
    phases = []
    for letter in letters:
        shape = _LETTER_TO_SHAPE.get(letter, f"{letter}-handen")
        if phases and phases[-1][0] == shape:
            continue  # collapse double letters (e.g. G-A-L-L → G-A-L)
        phases.append((shape, None, "one", None, None, None, None, None))
    return phases


def _is_plural(raw: str) -> bool:
    sl = raw.lower()
    return bool(re.search(r'händer\b', sl)) or sl in _IRREGULAR_PLURAL


# ---------------------------------------------------------------------------
# Attitude parsing
# ---------------------------------------------------------------------------

_DIRS = ['uppåt', 'nedåt', 'vänster', 'höger', 'framåt', 'inåt']

# Right-hand interpretation of symmetric two-hand descriptions:
#   "riktade mot varandra" → right hand points left  → "vänsterriktade"
#   "vända mot varandra"   → right hand faces left   → "vänstervända"
_sub_riktat = re.compile(r'rikta[dt][e]?\s+mot varandra', re.IGNORECASE)
_sub_vanda  = re.compile(r'vän[dt]a?\s+mot varandra',     re.IGNORECASE)

_pat_att = re.compile(
    r'(' + '|'.join(_DIRS) + r')rikta[dt][e]?\s+och\s+(' + '|'.join(_DIRS) + r')vän[dt]a?',
    re.IGNORECASE,
)

# Non-dominant hand: shape + optional attitude after a spatial preposition.
# Matches: "ovanpå flata handen, högerriktad och uppåtvänd"
#          "framför knutna handen, framåtriktad och nedåtvänd"
_NONDOM_PREPS = r'(?:ovanpå|framför|bredvid|under|bakom|längs|mot)'
_NONDOM_RE = re.compile(
    r'(?:kontakt\s+)?' + _NONDOM_PREPS + r'\s+'
    r'((?:[a-zåäö]+[-\s])*[a-zåäö]*handen?)',
    re.IGNORECASE,
)


def parse_nondom(phase: str) -> tuple[str | None, str | None]:
    """
    Extract non-dominant hand shape and attitude from a phase string.

    Returns (canonical_shape | None, attitude_slug | None).
    """
    m = _NONDOM_RE.search(phase)
    if not m:
        return None, None

    raw_shape = m.group(1).strip()
    canonical = _normalise_shape(raw_shape)

    # Look for attitude in the text after the matched shape
    after = phase[m.end():]
    att_m = _pat_att.search(after)
    att = f"{att_m.group(1).lower()}riktad-{att_m.group(2).lower()}vänd" if att_m else None

    return canonical, att


def _normalise_phase(phase: str) -> str:
    """Apply mot-varandra substitutions so the main regex can match."""
    phase = _sub_riktat.sub('vänsterriktade', phase)
    phase = _sub_vanda.sub('vänstervända',    phase)
    return phase


def _parse_attitude(phase_norm: str) -> str | None:
    """Extract attitude slug from a (pre-normalised) phase string, or None."""
    m = _pat_att.search(phase_norm)
    if m:
        return f"{m.group(1).lower()}riktad-{m.group(2).lower()}vänd"
    return None


_MOT_VARANDRA_RE = re.compile(r'mot\s+varandra', re.IGNORECASE)


def _mirror_lr(att: str | None) -> str | None:
    """Swap vänster↔höger in an attitude slug (for non-dominant hand of plural sign)."""
    if att is None:
        return None
    return (att
            .replace('vänster', '\x00')
            .replace('höger',   'vänster')
            .replace('\x00',    'höger'))


# ---------------------------------------------------------------------------
# Motion-direction extraction
# ---------------------------------------------------------------------------

_MOTION_DIRS = [
    'uppåt', 'nedåt', 'inåt', 'framåt', 'bakåt',
    r'åt\s+höger', r'åt\s+vänster',
    r'mot\s+varandra', r'från\s+varandra',
]
_MOTION_VERBS = r'(?:förs|föres|slås|vrids|drags|drages|lyftes|böjs|fälls)'
_MOTION_RE = re.compile(
    _MOTION_VERBS + r'\s+(?:kort\s+)?(' + '|'.join(_MOTION_DIRS) + r')',
    re.IGNORECASE,
)

# Dominant-hand remapping: symmetric two-hand directions → single-hand equivalents
_MOTION_REMAP: dict[str, str] = {
    'mot_varandra':   'åt_vänster',   # dominant right hand moves left toward nondominant
    'från_varandra':  'åt_höger',     # dominant right hand moves right away from nondominant
}

# Canonical direction vocabulary (7 classes)
MOTION_DIRECTIONS: list[str] = ['nedåt', 'uppåt', 'framåt', 'bakåt', 'åt_höger', 'åt_vänster', 'inåt']


def parse_motion_dir(phase: str) -> str | None:
    """Return the dominant-hand movement direction from a phase string, or None."""
    m = _MOTION_RE.search(phase)
    if not m:
        return None
    raw = re.sub(r'\s+', '_', m.group(1).lower())
    return _MOTION_REMAP.get(raw, raw) if raw in MOTION_DIRECTIONS or raw in _MOTION_REMAP else None


# ---------------------------------------------------------------------------
# Description parser
# ---------------------------------------------------------------------------

# "förändras till X", also with an adverbial before "till" ("förändras först / sedan /
# slutligen / ett par gånger / framför munnen till X") and the "til" typo.
_FORANDRAS_RE = re.compile(r'förändras\s+(?:(?!till?\b)[^,/\s]+\s+){0,4}?till?\s+([^,]+)', re.IGNORECASE)
# Words that end a shape name after "till" (contact / motion / manner follows).
_SHAPE_END_RE = re.compile(r'\s+(?:med|i|mot|och|framför|upprepas|som|samtidigt|ovanför|under|ovanpå|'
                           r'till|flera|ett|sedan|därefter|slutligen|bredvid|vid|bakom|nedanför)\b', re.IGNORECASE)
# Chained changes after the first: "… till X och sedan / därefter / slutligen till Y".
_FORANDRAS_NEXT_RE = re.compile(r'\b(?:och\s+)?(?:sedan|därefter|slutligen)\s+till\s+([^,]+)', re.IGNORECASE)

# Optional module-level LLM-fallback cache: maps description string →
# list of (shape, att, plural, cloc, ctype) tuples (same format as the
# deterministic parser).  Populated by batch_llm_parse.py and loaded via
# load_llm_parse_cache().
_LLM_PARSE_CACHE: dict[str, list] = {}


def load_llm_parse_cache(path: str) -> None:
    """Load a JSON cache produced by batch_llm_parse.py into the module-level
    cache so that _parse_description can use it as a fallback."""
    import json
    global _LLM_PARSE_CACHE
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # raw is {desc: [[shape, att, plural, cloc, ctype], ...]}
    # Old cache files have 5 elements; map plural→hand_type and pad to 9.
    def _upgrade(ph):
        t = list(ph)
        if len(t) == 5:  # old format: shape, att, plural, cloc, ctype
            plural = t.pop(2)
            hand_type = "two" if plural else "one"
            t.insert(2, hand_type)      # → shape, att, hand_type, cloc, ctype
            t.insert(5, None)           # motion
            t += [None, None]           # nondom_shape, nondom_att
        return tuple(t) + (None,) * (8 - len(t))
    _LLM_PARSE_CACHE = {
        desc: [_upgrade(ph) for ph in phases]
        for desc, phases in raw.items()
    }


def _parse_description(desc: str) -> list[tuple]:
    """
    Parse a description string into per-phase tuples:
        (raw_canonical_shape, attitude_slug | None,
         hand_type,           # "one" | "two"
         contact_location | None, contact_type | None,
         motion | None,       # canonical direction string or None
         nondom_shape | None, nondom_attitude | None)

    Parsing order:
    1. "Bokstaveras: X-Y-Z"  → one phase per letter
    2. Regular "Shape, attitude, ..." with //-phase splitting
       — last phase "förändras till X" appends an extra end-shape phase
    3. If result is still empty: check _LLM_PARSE_CACHE (populated by
       batch_llm_parse.py / load_llm_parse_cache()).

    Phases whose shape cannot be normalised (H. H. descriptions, etc.) are
    dropped; callers must also filter via _match_vocab if needed.
    """
    if not desc or not isinstance(desc, str):
        return []

    # ── 0. Old-style descriptions (Österberg 1916, "H. H." notation) ───────
    if re.match(r'H\.\s*H\.', desc.strip(), re.IGNORECASE):
        return []

    # ── 1. Fingerspelling ──────────────────────────────────────────────────
    boks = _parse_bokstaveras(desc)
    if boks is not None:
        return boks

    # ── 2. Regular deterministic parsing ──────────────────────────────────
    raw_phases = desc.split("//")
    result = []
    for phase in raw_phases:
        # A phase may itself be a Bokstaveras entry
        boks_phase = _parse_bokstaveras(phase.strip())
        if boks_phase is not None:
            result.extend(boks_phase)
            continue
        phase_norm = _normalise_phase(phase)
        raw_shape  = phase_norm.strip().split(",")[0].strip()
        hand_type  = "two" if _is_plural(raw_shape) else "one"
        canonical  = _normalise_shape(raw_shape)
        if canonical is None:
            continue
        att     = _parse_attitude(phase_norm)
        contact = parse_contact(phase_norm)
        contact_loc  = contact[0] if contact else None
        contact_type = contact[1] if contact else None
        motion       = parse_motion_dir(phase_norm)

        if hand_type == "two":
            nondom_shape = canonical
            if _MOT_VARANDRA_RE.search(phase):
                nondom_att = _mirror_lr(att)
            else:
                nondom_att = att
        else:
            nondom_shape, nondom_att = parse_nondom(phase_norm)

        result.append((canonical, att, hand_type, contact_loc, contact_type, motion, nondom_shape, nondom_att))

        # "förändras till X" anywhere in a phase → extra end-shape entry
        # ("förändras först till X och sedan till Y" → one entry per shape)
        m = _FORANDRAS_RE.search(phase_norm)
        if m:
            raws = [m.group(1)] + [n.group(1) for n in _FORANDRAS_NEXT_RE.finditer(phase_norm, m.start(1), m.end(1))]
            prev = canonical
            for end_raw in raws:
                end_raw = _SHAPE_END_RE.split(end_raw.strip())[0]
                end_canonical = _normalise_shape(end_raw.strip())
                if end_canonical and end_canonical != prev:
                    result.append((end_canonical, None, hand_type, contact_loc, contact_type, None, nondom_shape, nondom_att))
                    prev = end_canonical

    if result:
        return result

    # ── 3. LLM cache fallback ──────────────────────────────────────────────
    if desc in _LLM_PARSE_CACHE:
        return list(_LLM_PARSE_CACHE[desc])

    return []
