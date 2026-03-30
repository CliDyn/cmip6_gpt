#!/usr/bin/env python3
"""
Section-Aware Chunker V6 for CMIP6 RAG Pipeline
================================================
Reads Docling .docling.json files (raw JSON, no Pydantic) and produces
semanticly coherent chunks with rich metadata.

V2: Major quality overhaul based on Gemini review of 6,938 chunks.
    Fixes: HTML noise, URN placeholders, figure-axis gibberish, author
    fragments, excessive overlap, URL-only captions.
V5: Data cleaning overhaul based on manual review of 4,567 chunks.
    Fixes: OCR 'ris' stripping restoration, stronger reference-list
    filtering, caption merging, content-hash dedup, section path validation.
V6: Sanitizer based on Gemini 3.1 Pro Preview audit of 4,060 chunks.
    Fixes: context-dependent k/e/Pa OCR fixes, +→C PDF glyph fix,
    digit-density garbage filter, strengthened ref/boilerplate filtering,
    expanded section path validation.

Output: rag/chunks.jsonl  (one JSON object per line)
"""

import json
import re
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import tiktoken

# ── Configuration ──────────────────────────────────────────────────────────────
PARSED_DIR = Path(__file__).parent / "parsed"
PARSED_LEVANTE_DIR = Path(__file__).parent / "parsed_levante"
OUTPUT_FILE = Path(__file__).parent / "chunks.jsonl"
OUTPUT_LEVANTE_FILE = Path(__file__).parent / "chunks_levante.jsonl"

MAX_TOKENS = 1000       # target cap per chunk
TABLE_MAX_TOKENS = 1200 # hard cap for table chunks (larger allowed)
MIN_TOKENS = 80         # merge if below this
MIN_QUALITY_TOKENS = 30 # absolute minimum to emit a chunk
OVERLAP_RATIO = 0.05    # 5% overlap (reduced from 10% — review showed excessive)
ABSTRACT_SOLO = True    # abstract always gets its own chunk

# Sections to EXCLUDE from RAG (noise for retrieval)
EXCLUDE_SECTIONS = {
    # Academic boilerplate
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "author information", "authors and affiliations", "contributions",
    "author contributions", "corresponding author", "corresponding authors",
    "ethics declarations", "competing interests", "conflict of interest",
    "additional information", "supplementary information", "supplementary material",
    "supplementary materials", "supplementary data",
    "rights and permissions", "about this article", "cite this article",
    "this article is cited by", "data availability", "data availability statement",
    "code availability", "code and data availability", "declarations", "funding",
    "financial support", "review statement", "disclaimer",
    "change history", "ethics and inclusion statement",
    "information & authors", "submission history", "notes",
    "peer review information", "peer review", "publisher's note",
    # V3: Front/back matter from Gemini review
    "lead authors", "contributing authors", "how to cite",
    "table of contents", "contents", "orcidids", "orcids",
    "correspondence", "correspondence to", "edited by", "reviewed by",
    "copyright", "open access", "license",
    "author affiliations", "affiliations",
    "create a new account",  # web UI bleed
    # V6: Boilerplate / publisher noise
    "general rights", "take down policy", "preamble",
    "citation for published version", "citation for published version:",
    "fair data use statement",
    "resources",  # PMC sidebar
    # Journal navigation (Nature, Wiley, AGU, PNAS, etc.)
    "explore content", "about the journal", "publish with us",
    "search", "quick links", "nature.com sitemap",
    "about nature portfolio", "discover content", "publishing policies",
    "author & researcher services", "libraries & institutions",
    "advertising & partnerships", "professional development",
    "regional websites", "similar content being viewed by others",
    "access options", "additional access options:", "subjects",
    "access through your institution", "buy or subscribe",
    # Wiley-specific
    "citing literature", "article metrics", "related articles",
    "connect with wiley", "change password", "forgot your password?",
    "request username", "login", "login / register",
    # PNAS-specific
    "sign up for pnas alerts", "metrics",
    # PMC-specific
    "links to ncbi databases", "cited by other articles",
    # IOP-specific
    "you may also like",
    # General
    "privacy preference center", "manage consent preferences",
    "essential cookies", "performance cookies", "functional cookies",
    "cookie list", "reviewpaper",
}

# Text patterns that indicate HTML navigation noise
NOISE_PATTERNS = [
    r"^skip to (?:main )?content$",
    r"^skip to article$",
    r"^thank you for visiting nature\.com",
    r"^you are using a browser version",
    r"^the best experience",
    r"^internet explorer\b",
    r"^this site uses cookies",
    r"^we use cookies",
    r"^accept all cookies",
    r"^jump to content$",
    r"^page not found$",
    r"^accessibility links$",
    r"^advertisement$",
    r"^log in$",
    r"^sign up$",
    r"^sign in$",
    r"^subscribe$",
    r"^view all",
    # Wiley / AGU / journal navigation
    r"^privacy policy$",
    r"^terms of use$",
    r"^about cookies$",
    r"^manage cookies$",
    r"^accessibility$",
    r"^wiley online library$",
    r"^publication (?:award|policies|ethics)",
    r"^submit a paper$",
    r"^usage statistics$",
    r"^scientific ethics$",
    r"^copyright ©",
    r"^© \d{4}",
    r"^volume \d+.*issue \d+",
    r"^\d+ pages?$",
    r"^open access$",
    r"^full access$",
    r"^free access$",
    r"^download pdf$",
    r"^share$",
    r"^cite$",
    r"^figures?$",
    r"^tables?$",
    r"^related$",
    r"^information$",
    r"^metrics$",
    r"^\s*doi:\s*$",
    # V2: Additional patterns from Gemini review
    r"^download (?:xlsx|pdf|csv|print version)$",
    r"^open in figure viewer$",
    r"^check for updates",
    r"^verify currency and authenticity",
    r"^crossmark$",
    r"^scite metrics$",
    r"^share qr code$",
    r"^altmetric",
    r"^article has an altmetric score",
    r"^forgot your password",
    r"^request username$",
    r"^change password$",
    r"^congrats!$",
    r"^your password must have",
    r"^a lower case character",
    r"^an upper case character",
    r"^a special character",
    r"^or a digit$",
    r"^send email$",
    r"^recipient\(s\) will receive",
    r"^article activity alert$",
    r"^\d+ publications? \d+ supporting",
    r"^this article also appears in",
    r"^multiple terms:",
    r"^we are sorry, but your search",
    r"^turn mathjax on$",
    r"^creative commons attribution",
    r"^data protection$",
    r"^full-text xml$",
    r"^bibtex$",
    r"^ris$",
    r"^endnote$",
    r"^sciencedirect$",
    r"^journals & books$",
    r"^my account$",
    r"^dieses dialogfeld schlie",
    r"^datenschutzrichtlinie",
    r"^geoscientific model development",
    r"^pnas nexus$",
    # URN placeholders (Wiley media refs)
    r"^urn:x-wiley:",
    # Standalone URLs
    r"^https?://",
    # Journal/publisher names as standalone text
    r"^nature climate change$",
    r"^nature communications$",
    r"^nature food$",
    r"^nature reviews",
    r"^scientific data$",
    r"^communications earth",
]
_noise_re = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

# Additional exact-match noise strings (lowercased)
_noise_exact = {
    "pdf", "xml", "ris", "abstract", "figures", "references", "related",
    "information", "metrics", "share", "cite", "tables", "home",
    "about", "help", "contact", "feedback", "search",
    "open menu", "close menu", "sections", "tools",
    "download", "save", "print", "export", "alerts",
    "sign in", "register", "my account", "cart",
    "back to top", "next", "previous", "show more",
    "supplementary data", "supplementary materials",
    "view article", "crossref", "google scholar",
    "pubmed", "web of science",
}

# Labels to always skip
SKIP_LABELS = {"page_header", "page_footer", "footnote"}

# ── V5: OCR "ris" Restoration ─────────────────────────────────────────────────
# Docling PDF parser strips the ligature "ris" from words. This dictionary
# maps corrupted forms → correct forms using strict word boundaries.
# Only SAFE corrections with \b boundaries are included.

_OCR_RIS_CORRECTIONS = {
    # ── High-frequency scientific vocabulary ──
    r"\bcompaon\b": "comparison",
    r"\bcompaons\b": "comparisons",
    r"\bcharactetics\b": "characteristics",
    r"\bcharactetic\b": "characteristic",
    r"\bcharacteing\b": "characterising",
    r"\bcharacteed\b": "characterised",
    r"\bcharactee\b": "characterise",
    r"\bcharactees\b": "characterises",
    r"\bcharacteze\b": "characterize",
    r"\bcharactezed\b": "characterized",
    r"\bcharactezes\b": "characterizes",
    r"\bcharactezing\b": "characterizing",
    r"\bcharactezation\b": "characterization",
    r"\bparameteation\b": "parameterisation",
    r"\bparameteations\b": "parameterisations",
    r"\bparameteize\b": "parameterize",
    r"\bparameteized\b": "parameterized",
    r"\bparameteizes\b": "parameterizes",
    r"\bparameteizing\b": "parameterizing",
    r"\bparameteization\b": "parameterization",
    r"\bparameteizations\b": "parameterizations",
    r"\bcompe\b": "comprise",
    r"\bcomped\b": "comprised",
    r"\bcompes\b": "comprises",
    r"\bcomping\b": "comprising",
    r"\bsurpe\b": "surprise",
    r"\bsurped\b": "surprised",
    r"\bsurpes\b": "surprises",
    r"\bsurping\b": "surprising",
    r"\bsurpingly\b": "surprisingly",
    r"\bheutic\b": "heuristic",
    r"\bheutics\b": "heuristics",
    r"\bptine\b": "pristine",
    r"\btoum\b": "tourism",
    r"\btout\b": "tourist",
    r"\btouts\b": "tourists",
    r"\bnouhment\b": "nourishment",
    r"\bdeb\b": "debris",
    r"\ben\b": "risen",
    r"\baes\b": "arises",
    r"\baing\b": "arising",
    r"\bsummaed\b": "summarised",
    r"\bsummaes\b": "summarises",
    r"\bsummae\b": "summarise",
    r"\bregulaed\b": "regularised",
    r"\bregulae\b": "regularise",
    r"\bautherise\b": "authorise",
    r"\bautheed\b": "authorised",
    # ── Author names (case-sensitive) ──
    r"\bChtian\b": "Christian",
    r"\bChtians\b": "Christians",
    r"\bChtianity\b": "Christianity",
    r"\bChtopher\b": "Christopher",
    r"\bChtensen\b": "Christensen",
    r"\bChtoffersen\b": "Christoffersen",
    r"\bChtoph\b": "Christoph",
    r"\bChtophe\b": "Christophe",
    r"\bBbane\b": "Brisbane",
    r"\bKtie\b": "Katie",
    r"\bKtjansson\b": "Kristjansson",
    r"\bKtensen\b": "Kristensen",
    # ── Multi-word phrase fixes (safe context-dependent) ──
    r"\bgives e to\b": "gives rise to",
    r"\bgave e to\b": "gave rise to",
    r"\bgiven e to\b": "given rise to",
    r"\bgiving e to\b": "giving rise to",
    r"\bat k\b": "at risk",
    r"\bat k of\b": "at risk of",
    r"\bthe e of\b": "the rise of",
    r"\bthe e in\b": "the rise in",
    r"\bon the e\b": "on the rise",
    r"\btemperature e\b": "temperature rise",
    r"\bsea level e\b": "sea level rise",
    r"\bsea-level e\b": "sea-level rise",
    r"\bhigh k\b": "high risk",
    r"\blow k\b": "low risk",
    r"\bk assessment\b": "risk assessment",
    r"\bk assessments\b": "risk assessments",
    r"\bk factor\b": "risk factor",
    r"\bk factors\b": "risk factors",
    r"\bk management\b": "risk management",
    r"\bk of\b": "risk of",
    # V6: Expanded context-dependent "k" = risk
    r"\bk reduction\b": "risk reduction",
    r"\bk tolerance\b": "risk tolerance",
    r"\bk Information\b": "risk Information",
    r"\bflood k\b": "flood risk",
    r"\bfire k\b": "fire risk",
    r"\bmortality k\b": "mortality risk",
    r"\benvironmental k\b": "environmental risk",
    r"\bdisaster k\b": "disaster risk",
    r"\bclimate k\b": "climate risk",
    r"\bcompound ks\b": "compound risks",
    r"\bincreasing ks\b": "increasing risks",
    r"\bmeteo-hydrological ks\b": "meteo-hydrological risks",
    r"\bks ae\b": "risks arise",
    r"\bks from\b": "risks from",
    r"\bks and\b": "risks and",
    r"\bks to\b": "risks to",
    r"\bks of\b": "risks of",
    # V6: Expanded context-dependent "e" = rise
    r"\bGMSL e\b": "GMSL rise",
    r"\bing sea level\b": "rising sea level",
    r"\bing temperatures\b": "rising temperatures",
    r"\bing CO2\b": "rising CO2",
    r"\bing food prices\b": "rising food prices",
    r"\bmarked e\b": "marked rise",
    r"\brapid e\b": "rapid rise",
    r"\bglobal e\b": "global rise",
    # V6: "Pa" = Paris
    r"\bPa Agreement\b": "Paris Agreement",
    r"\bglobal climate cis\b": "global climate crisis",
    # V6: "+" → "C" PDF glyph fix
    r"\bHistorical C SSP\b": "Historical + SSP",
}

# Substring-based corrections — these are SAFE because the corrupted forms
# NEVER appear as valid English words/substrings. Used for compound words
# where \b word boundaries fail (e.g., "intercompaon", "compaons").
_OCR_SUBSTR_CORRECTIONS = {
    "compaon": "comparison",  # intercompaon, compaons, etc.
    "charactetic": "characteristic",  # charactetics, charactetically
    "characteing": "characterising",
    "characteed": "characterised",
    "characteze": "characterize",
    "charactezed": "characterized",
    "charactezing": "characterizing",
    "charactezation": "characterization",
    "parameteation": "parameterisation",
    "parameteization": "parameterization",
    "parameteize": "parameterize",
    "parameteized": "parameterized",
    # V6: Additional compound-word substrate fixes
    "enterpe": "enterprise",     # enterprise → enterpe
    "polaing": "polarising",     # polarising → polaing
    "Ctofanelli": "Cristofanelli",  # author name
    "Pco": "Prisco",              # author name
    "Kmer": "Krismer",             # author name
    " fi ": " fi",  # ligature: "signi fi cant" → "significant"
    " fl ": " fl",  # ligature: "in fl uence" → "influence"
    " fi\n": " fi\n",  # fi at line end — SKIP, leave as-is
}

# Build safe substring corrections: the fi/fl ligature fix needs special care.
# "signi fi cant" → rejoin then the word is correct: "significant"
# Strategy: remove the space around fi/fl to rejoin the word.
_OCR_LIGATURE_PATTERNS = [
    (re.compile(r'\b(\w+)\s+fi\s+(\w+)\b'), r'\1fi\2'),   # "signi fi cant" → "significant"
    (re.compile(r'\b(\w+)\s+fl\s+(\w+)\b'), r'\1fl\2'),   # "in fl uence" → "influence"
    (re.compile(r'\b(\w+)\s+ff\s+(\w+)\b'), r'\1ff\2'),   # "e ff ect" → "effect"
]

# Pre-compile word-boundary patterns for speed
_ocr_ris_compiled = [(re.compile(pat), repl) for pat, repl in _OCR_RIS_CORRECTIONS.items()]

# Pre-compile substring patterns (just plain string replace, no regex needed)
_ocr_substr_safe = [
    ("compaon", "comparison"),
    ("charactetic", "characteristic"),
    ("characteing", "characterising"),
    ("characteed", "characterised"),
    ("characteze", "characterize"),
    ("charactezed", "characterized"),
    ("charactezing", "characterizing"),
    ("charactezation", "characterization"),
    ("parameteation", "parameterisation"),
    ("parameteization", "parameterization"),
    ("parameteize", "parameterize"),
    ("parameteized", "parameterized"),
]


def fix_ocr_ris_stripping(text: str) -> str:
    """Fix systematic OCR corruption where 'ris' ligature is stripped from words.
    
    Three-pass approach:
    1. Word-boundary regex for isolated corrupted words
    2. Substring replacement for compound words (intercompaon, etc.)
    3. Ligature rejoining for split fi/fl/ff ligatures
    
    Safe: will never turn a valid word into something wrong.
    """
    # Pass 1: word-boundary corrections
    for pattern, replacement in _ocr_ris_compiled:
        text = pattern.sub(replacement, text)
    
    # Pass 2: substring corrections for compound words
    for old, new in _ocr_substr_safe:
        if old in text:
            text = text.replace(old, new)
    
    # Pass 3: rejoin split fi/fl/ff ligatures
    for pattern, replacement in _OCR_LIGATURE_PATTERNS:
        text = pattern.sub(replacement, text)
    
    return text


# ── V5: Garbage Section Path Detection ────────────────────────────────────────

_JOURNAL_NAME_SECTIONS = {
    "journal of advances in modeling earth systems",
    "reviews of geophysics",
    "geoscientific model development",
    "earth system dynamics",
    "earth system science data",
    "nature climate change",
    "nature communications",
    "nature geoscience",
    "nature food",
    "scientific data",
    "communications earth & environment",
    "environmental research letters",
    "global change biology",
    "journal of climate",
    "pnas",
    "pnas nexus",
    "science advances",
    "science",
}

_garbage_section_re = re.compile(
    r"^-?\d+\.?\d*\s+\d+\.?\d*[°ºÅ]?$"  # coordinate-like: "-25 60", "0 180°"
    r"|^\d{1,4}$"                         # bare numbers: "691"
    r"|^[°ºÅ\d\s.,-]+$"                   # pure numeric/degree strings
    r"|^\w{1,3}$"                          # single tiny word: "net"
    # V6: More garbage patterns
    r"|^PUBLISHED$"                        # Wiley boilerplate
    r"|^Data:?$"                           # floating label
    r"|^[A-Z]{2,4}\s+[A-Z]{2,4}$"         # chart legend: "WCA ECA", "CAU EAU"
    r"|^\d+\|"                             # axis tick prefix: "210|"
    r"|^Figures?\s*\d"                     # "Figures 414", "Figure 3"
    r"|^Supplementary\s+Table\b"           # "Supplementary Table 1..."
    r"|^NPP:\s*"                           # "NPP: MODIS" axis label
)


def is_garbage_section_path(section_name: str) -> bool:
    """Detect section paths that are garbled coordinates, journal names, or gibberish."""
    s = section_name.strip()
    if not s:
        return True
    sl = s.lower()
    if sl in _JOURNAL_NAME_SECTIONS:
        return True
    if _garbage_section_re.match(s):
        return True
    return False

# ── V2: Advanced Noise Detectors ──────────────────────────────────────────────

_urn_re = re.compile(r"urn:x-wiley:", re.IGNORECASE)
_url_only_re = re.compile(r"^\s*https?://\S+\s*$")
_affiliation_re = re.compile(
    r"(?:department|school|university|institute|laboratory|center|centre|faculty)"
    r"|(?:@[a-z0-9.-]+\.[a-z]{2,})"
    r"|(?:orcid\.org)"
    r"|(?:\d{4}-\d{4}-\d{4}-\d{3}[0-9X])",
    re.IGNORECASE
)
_degree_coord_re = re.compile(r"[°ºÅ][NSEW]", re.IGNORECASE)
_ui_button_re = re.compile(
    r"(?:Download|Open in figure viewer|PowerPoint|Print Version|Download XLSX"
    r"|Download PDF|Download CSV|Full-text XML|BibTeX|EndNote|RIS)",
    re.IGNORECASE
)


def is_urn_placeholder(text: str) -> bool:
    """Check if text is mostly Wiley URN placeholders."""
    lines = text.strip().split("\n")
    urn_lines = sum(1 for l in lines if _urn_re.search(l))
    return urn_lines > len(lines) * 0.5


def is_url_only(text: str) -> bool:
    """Check if text is just a URL."""
    return bool(_url_only_re.match(text.strip()))


def is_affiliation_fragment(text: str) -> bool:
    """Check if text is a standalone author affiliation chunk."""
    t = text.strip()
    # Short text with affiliation markers
    if len(t) > 500:
        return False
    matches = len(_affiliation_re.findall(t))
    words = len(t.split())
    if words < 3:
        return False
    # High density of affiliation markers = likely affiliation
    return matches >= 2 and matches / max(words, 1) > 0.05


def is_figure_axis_gibberish(text: str) -> bool:
    """Detect text extracted from figure axes/legends (word-salad)."""
    t = text.strip()
    if len(t) < 30:
        return False
    
    # Heuristic: many degree/coordinate symbols
    coord_matches = len(_degree_coord_re.findall(t))
    if coord_matches > 5:
        return True
    
    # Heuristic: very fragmented text (many very short lines)
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    if len(lines) > 5:
        short_lines = sum(1 for l in lines if len(l.split()) <= 3)
        if short_lines / len(lines) > 0.7:
            return True
    
    # Heuristic: repetitive short fragments separated by spaces
    words = t.split()
    if len(words) > 20:
        # Count 1-2 char "words" (likely axis ticks)
        tiny = sum(1 for w in words if len(w) <= 2 and not w.isalpha())
        if tiny / len(words) > 0.3:
            return True
    
    # V3: Short verb-less lines = axis labels (e.g. "RCP8.5", "Buenos Aires")
    if len(lines) > 8:
        # Check if most lines lack verbs (proxy: no words ending in common verb suffixes)
        verbless = 0
        for line in lines:
            wds = line.split()
            has_verb_like = any(
                w.lower().endswith(('ing', 'tion', 'ted', 'tes', 'ses', 'ize', 'ise', 'ate'))
                for w in wds if len(w) > 3
            )
            if not has_verb_like and len(wds) <= 5:
                verbless += 1
        if verbless / len(lines) > 0.6:
            return True
    
    return False


def is_digit_heavy_garbage(text: str, threshold: float = 0.30) -> bool:
    """V6: Detect chunks that are mostly numbers/symbols (axis data, coordinates).
    
    If >30% of characters are digits and special symbols, this is likely
    visual garbage from chart axes, coordinate grids, or figure annotations.
    """
    t = text.strip()
    if len(t) < 50:
        return False
    symbols = sum(1 for c in t if c.isdigit() or c in "°±+-.,|<>[]{}()=×÷")
    return symbols / len(t) > threshold


def is_boilerplate_noise(text: str) -> bool:
    """V6: Detect publisher boilerplate, copyright text, and preprint disclaimers."""
    t = text.strip().lower()
    boilerplate_markers = [
        "copyright and moral rights",
        "research square preprints are preliminary",
        "in the format provided by the authors and unedited",
        "take down policy",
        "general rights",
        "citation for published version",
        "verify currency and authenticity",
        "version of record",
    ]
    matches = sum(1 for m in boilerplate_markers if m in t)
    return matches >= 2 or (matches >= 1 and len(t) < 300)


# V3/V5: Reference-like text detector (strengthened in V5)
_ref_pattern = re.compile(
    r"(?:[A-Z][a-z]+(?:,\s*[A-Z]\.?)+\s*(?:,|&|and)\s*){2,}"
    r"|(?:\(\d{4}[a-z]?\))"
    r"|(?:et\s+al\.?,\s*\d{4})"
    r"|(?:doi:\s*10\.\d{4,})",
    re.IGNORECASE
)
_doi_re_v5 = re.compile(r"10\.\d{4,9}/\S+")
_year_bracket_re = re.compile(r"\(\d{4}[a-z]?\)")

def is_reference_block(text: str) -> bool:
    """Detect if text is mostly bibliographic references.
    
    V5: Added DOI density and year-bracket density checks.
    V6: Lowered thresholds (DOI 5→3, years 8→6), added "et al." density.
    """
    t = text.strip()
    if len(t) < 100:
        return False
    
    # V6: DOI density check — if 3+ DOIs in one chunk, it's a ref block
    doi_count = len(_doi_re_v5.findall(t))
    if doi_count >= 3:
        return True
    
    # V6: Year-bracket density — 6+ year citations like (2019) in one chunk
    year_count = len(_year_bracket_re.findall(t))
    if year_count >= 6:
        return True
    
    # V6: "et al." density — 6+ occurrences = bibliography
    et_al_count = t.lower().count("et al")
    if et_al_count >= 6:
        return True
    
    # V3: Line-based detection
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    ref_lines = sum(1 for l in lines if _ref_pattern.search(l))
    # If >60% of lines look like references
    return ref_lines / len(lines) > 0.6


def has_repeating_loop(text: str, min_repeat: int = 3) -> bool:
    """Detect text with repeating string loops (e.g. 'Coral: Bard...' x10)."""
    t = text.strip()
    if len(t) < 200:
        return False
    # Check for any substring of 20+ chars repeating 3+ times
    for length in (50, 30, 20):
        for start in range(0, min(len(t) - length, 500), 10):
            substr = t[start:start + length]
            if t.count(substr) >= min_repeat:
                return True
    return False


_line_noise_re = re.compile(
    r"^(?:"
    r"urn:x-wiley:|"
    r"https?://\S+$|"
    r"Dieses Dialogfeld|"
    r"Datenschutzrichtlinie|"
    r"This site uses cookies|"
    r"We use cookies|"
    r"Accept all cookies|"
    r"Forgot your password|"
    r"Request Username|"
    r"Change Password|"
    r"Your password must have|"
    r"a lower case character|"
    r"an upper case character|"
    r"a special character|"
    r"or a digit|"
    r"Congrats!|"
    r"Login / Register|"
    r"Scite metrics|"
    r"Share QR Code|"
    r"Access through your institution|"
    r"Buy or subscribe|"
    r"Check for updates|"
    r"Open in figure viewer|"
    r"Wiley Online Library|"
    r"Article Activity Alert|"
    r"Send Email|"
    r"Recipient\(s\) will receive|"
    r"Article has an altmetric score|"
    r"\d+ publications? \d+ supporting|"
    r"Download & links|"
    r"Full-text XML|"
    r"BibTeX|"
    r"Multiple terms:|"
    r"We are sorry, but your search|"
    r"Turn MathJax on|"
    r"Creative Commons|"
    r"Connect with Wiley|"
    r"Privacy Preference Center|"
    r"Manage Consent Preferences|"
    r"Essential cookies|"
    r"Performance cookies|"
    r"Functional cookies|"
    r"Cookie List|"
    # V3: Additional boilerplate patterns
    r"Correspondence to:|"
    r"Received:\s+\d|"
    r"Accepted:\s+\d|"
    r"Published:\s+\d|"
    r"Edited by:|"
    r"Reviewed by:|"
    r"Lead Authors?:|"
    r"Contributing Authors?:|"
    r"How to cite|"
    r"Crown copyright|"
    r"Attribution \d\.\d License|"
    r"An official website of the United States|"
    r"Search PMC|"
    r"Sorry, we could not find|"
    r"Go to Figure|"
    r"Open all in viewer|"
    r"There are no results for"
    r")",
    re.IGNORECASE
)


def clean_ui_from_text(text: str) -> str:
    """Strip embedded UI elements and noise lines from text."""
    # Remove "Open in figure viewer PowerPoint" etc.
    text = _ui_button_re.sub("", text)
    # Remove "Check for updates" boilerplate
    text = re.sub(r"Check for updates\.?\s*Verify currency and authenticity via CrossMark\.?", "", text)

    # V2: Line-level noise removal — strip individual noise lines
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append(line)
            continue
        if _line_noise_re.match(stripped):
            continue  # skip this line
        # Skip pure URN lines
        if stripped.startswith("urn:"):
            continue
        clean_lines.append(line)

    text = "\n".join(clean_lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ── Tokenizer ──────────────────────────────────────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (≈same as Gemini tokenization)."""
    return len(_enc.encode(text))


# ── Noise Detection ────────────────────────────────────────────────────────────

def is_noise_text(text: str) -> bool:
    """Return True if text is HTML navigation / cookie banner noise."""
    t = text.strip()
    if len(t) < 3:
        return True
    tl = t.lower()
    if tl in _noise_exact:
        return True
    for pat in _noise_re:
        if pat.search(t):
            return True
    # V2: URN placeholders
    if is_urn_placeholder(t):
        return True
    # V2: Standalone URLs
    if is_url_only(t):
        return True
    return False


def is_excluded_section(section_name: str) -> bool:
    """Return True if section should be excluded from RAG."""
    return section_name.strip().lower() in EXCLUDE_SECTIONS


# ── Document Tree Walker ───────────────────────────────────────────────────────

class DocNode:
    """A node in the document tree."""
    __slots__ = ("label", "text", "children", "ref", "table_data")

    def __init__(self, label: str, text: str = "", ref: str = "",
                 table_data: Optional[dict] = None):
        self.label = label
        self.text = text
        self.children: list["DocNode"] = []
        self.ref = ref
        self.table_data = table_data


def build_item_index(doc: dict) -> dict:
    """Build a ref -> raw item lookup from the docling document."""
    items = {}
    for collection in ("texts", "groups", "tables", "pictures",
                       "key_value_items", "form_items"):
        for item in doc.get(collection, []):
            ref = item.get("self_ref", "")
            if ref:
                items[ref] = item
    return items


def resolve_ref(child_ptr: dict) -> str:
    """Extract the reference string from a child pointer."""
    return child_ptr.get("cref", child_ptr.get("$ref", ""))


def build_tree(doc: dict) -> DocNode:
    """Build a tree of DocNodes from the docling document body."""
    items = build_item_index(doc)
    root = DocNode(label="root", text="document")

    def _build(children_list: list) -> list[DocNode]:
        nodes = []
        for child_ptr in children_list:
            ref = resolve_ref(child_ptr)
            raw = items.get(ref, {})
            label = raw.get("label", "unknown")
            text = raw.get("text", "")

            # For tables, preserve the full data
            table_data = None
            if label == "table" or "tables" in ref:
                table_data = raw

            node = DocNode(label=label, text=text, ref=ref,
                           table_data=table_data)

            # Recurse into children
            if "children" in raw:
                node.children = _build(raw["children"])

            nodes.append(node)
        return nodes

    body = doc.get("body", {})
    root.children = _build(body.get("children", []))
    return root


# ── Section Collector ──────────────────────────────────────────────────────────

class Section:
    """A logical section of the document."""
    __slots__ = ("name", "path", "paragraphs", "tables", "captions", "level")

    def __init__(self, name: str, path: str, level: int = 0):
        self.name = name
        self.path = path
        self.level = level
        self.paragraphs: list[str] = []  # text blocks
        self.tables: list[dict] = []     # table data
        self.captions: list[str] = []    # figure/table captions


def collect_sections(root: DocNode) -> list[Section]:
    """Walk the document tree and collect sections with their content."""
    sections: list[Section] = []
    current_section = Section(name="Preamble", path="Preamble", level=0)

    def _walk(node: DocNode, section_path_parts: list[str], depth: int):
        nonlocal current_section

        # Skip noise
        if node.label in SKIP_LABELS:
            return
        if node.label in ("text", "list_item") and is_noise_text(node.text):
            return

        # Section header → start new section
        if node.label == "section_header":
            header_text = node.text.strip()
            if not header_text:
                return

            # Check if this section should be excluded
            if is_excluded_section(header_text):
                # Still process children but mark section as excluded
                excluded = Section(name=header_text,
                                   path=" > ".join(section_path_parts + [header_text]),
                                   level=depth)
                excluded.paragraphs.append("__EXCLUDED__")
                sections.append(excluded)
                return

            # Save current section if it has content
            if current_section.paragraphs or current_section.tables or current_section.captions:
                sections.append(current_section)

            new_path = section_path_parts + [header_text]
            current_section = Section(
                name=header_text,
                path=" > ".join(new_path),
                level=depth,
            )
            # Process children under this section header
            for child in node.children:
                _walk(child, new_path, depth + 1)
            return

        # Text content
        if node.label in ("text", "list_item"):
            text = node.text.strip()
            if text and len(text) > 5:
                current_section.paragraphs.append(text)

        # Caption
        elif node.label == "caption":
            text = node.text.strip()
            if text:
                current_section.captions.append(text)

        # Table
        elif node.label == "table" or node.table_data:
            current_section.tables.append(node.table_data or {"text": node.text})

        # Key-value area, list, group, picture, inline → recurse
        for child in node.children:
            _walk(child, section_path_parts, depth)

    for child in root.children:
        _walk(child, [], 0)

    # Don't forget the last section
    if current_section.paragraphs or current_section.tables or current_section.captions:
        sections.append(current_section)

    return sections


# ── Markdown Section Parser (MinerU VLM) ───────────────────────────────────────

_md_header_re = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
_md_table_line_re = re.compile(r'^\|.*\|\s*$')
_md_figure_caption_re = re.compile(
    r'^(?:Figure|Fig\.|Table|Plate|Scheme)\s+\d+',
    re.IGNORECASE
)
_md_image_ref_re = re.compile(r'^!\[.*\]\(.*\)$')


def parse_markdown_sections(md_text: str) -> list["Section"]:
    """Parse MinerU VLM markdown into Section objects using # headers.
    
    Handles:
    - # / ## / ### headers → section boundaries
    - Markdown tables (| col | col |) → table entries
    - Figure/Table captions → caption entries
    - Image references ![...](...) → skipped
    - Everything else → text paragraphs
    """
    sections: list[Section] = []
    current_section = Section(name="Preamble", path="Preamble", level=0)
    section_stack: list[tuple[int, str]] = []  # (level, name) for building paths
    
    lines = md_text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            i += 1
            continue
        
        # Skip image references
        if _md_image_ref_re.match(stripped):
            i += 1
            continue
        
        # Check for header
        header_match = _md_header_re.match(stripped)
        if header_match:
            level = len(header_match.group(1))  # number of #s
            header_text = header_match.group(2).strip()
            
            if not header_text:
                i += 1
                continue
            
            # Save current section if it has content
            if current_section.paragraphs or current_section.tables or current_section.captions:
                sections.append(current_section)
            
            # Update section stack
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, header_text))
            
            # Build section path from stack
            path_parts = [name for _, name in section_stack]
            section_path = " > ".join(path_parts)
            
            # Check if excluded
            if is_excluded_section(header_text):
                current_section = Section(name=header_text, path=section_path, level=level)
                current_section.paragraphs.append("__EXCLUDED__")
                sections.append(current_section)
                # Skip content until next header of same or higher level
                i += 1
                while i < len(lines):
                    next_match = _md_header_re.match(lines[i].strip())
                    if next_match and len(next_match.group(1)) <= level:
                        break
                    i += 1
                current_section = Section(name="[continued]", path="[continued]", level=0)
                continue
            
            current_section = Section(name=header_text, path=section_path, level=level)
            i += 1
            continue
        
        # Check for markdown table block
        if _md_table_line_re.match(stripped):
            table_lines = []
            while i < len(lines) and _md_table_line_re.match(lines[i].strip()):
                table_lines.append(lines[i].strip())
                i += 1
            if table_lines:
                table_text = '\n'.join(table_lines)
                current_section.tables.append({"text": table_text})
            continue
        
        # Check for figure/table caption
        if _md_figure_caption_re.match(stripped):
            # Collect multi-line caption
            caption_lines = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not _md_header_re.match(lines[i].strip()):
                if _md_table_line_re.match(lines[i].strip()):
                    break
                if _md_image_ref_re.match(lines[i].strip()):
                    i += 1
                    continue
                if _md_figure_caption_re.match(lines[i].strip()):
                    break  # new caption starts
                caption_lines.append(lines[i].strip())
                i += 1
            current_section.captions.append(' '.join(caption_lines))
            continue
        
        # Regular text — skip noise
        if is_noise_text(stripped):
            i += 1
            continue
        
        # Collect paragraph (consecutive non-empty, non-header lines)
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            next_stripped = lines[i].strip()
            if not next_stripped:  # blank line = paragraph break
                break
            if _md_header_re.match(next_stripped):
                break
            if _md_table_line_re.match(next_stripped):
                break
            if _md_image_ref_re.match(next_stripped):
                i += 1
                continue
            if is_noise_text(next_stripped):
                i += 1
                continue
            para_lines.append(next_stripped)
            i += 1
        
        para_text = ' '.join(para_lines)
        if len(para_text) > 5:
            current_section.paragraphs.append(para_text)
    
    # Don't forget the last section
    if current_section.paragraphs or current_section.tables or current_section.captions:
        sections.append(current_section)
    
    return sections


def chunk_markdown_document(md_path: Path, paper_id: str) -> list[dict]:
    """Chunk a MinerU VLM markdown file using the same pipeline as docling.
    
    Reuses: all filters, OCR fixes, overlap, token budget, dedup, etc.
    """
    with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
        md_text = f.read()
    
    meta_path = md_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as mf:
            try:
                meta = json.load(mf)
            except Exception:
                pass

    # Extract DOI from paper_id (folder name or meta)
    doi = meta.get("doi")
    if not doi:
        parts = paper_id.split('_', 2)
        if len(parts) >= 3 and parts[0] == '10':
            doi = f"10.{parts[1]}/{parts[2].replace('_', '.')}"
        else:
            doi = paper_id.replace('_', '/')
    
    title = meta.get("title", "")
    year = meta.get("year", "")
    journal = meta.get("journal", "")
    tier = meta.get("tier", "UNKNOWN")
    
    # Parse markdown into sections
    sections = parse_markdown_sections(md_text)
    
    # ── Reuse the EXACT same chunking pipeline ──
    chunks_out = []
    chunk_counter = 0
    seen_hashes = set()
    
    for section in sections:
        if section.paragraphs == ["__EXCLUDED__"]:
            continue
        
        section_path = section.path
        if is_garbage_section_path(section.name):
            section_path = "[section unknown]"
        
        # ── Text chunks ──
        if section.paragraphs:
            full_text = "\n\n".join(section.paragraphs)
            full_text = fix_ocr_ris_stripping(full_text)
            text_tokens = count_tokens(full_text)
            
            is_abstract = section.name.strip().lower() == "abstract"
            chunk_type = "abstract" if is_abstract else "text"
            
            if text_tokens <= MAX_TOKENS:
                raw_chunks = [full_text]
            else:
                raw_chunks = chunk_text_block(full_text, MAX_TOKENS)
            
            if len(raw_chunks) > 1:
                raw_chunks = add_overlap(raw_chunks, OVERLAP_RATIO)
            
            capped = []
            for rc in raw_chunks:
                if count_tokens(rc) > MAX_TOKENS + 50:
                    capped.extend(chunk_text_block(rc, MAX_TOKENS))
                else:
                    capped.append(rc)
            raw_chunks = capped
            
            for i_chunk, chunk_text in enumerate(raw_chunks):
                ct = count_tokens(chunk_text)
                if ct < MIN_QUALITY_TOKENS:
                    continue
                chunk_text = clean_ui_from_text(chunk_text)
                if not chunk_text or count_tokens(chunk_text) < MIN_QUALITY_TOKENS:
                    continue
                if is_figure_axis_gibberish(chunk_text):
                    continue
                if is_digit_heavy_garbage(chunk_text):
                    continue
                if is_boilerplate_noise(chunk_text):
                    continue
                if is_affiliation_fragment(chunk_text):
                    continue
                if is_reference_block(chunk_text):
                    continue
                if has_repeating_loop(chunk_text):
                    continue
                
                content_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
                
                prefix = f'Paper: "{title}"' if title else f'Paper: {paper_id}'
                if year:
                    prefix += f" ({year}"
                    if journal:
                        prefix += f", {journal}"
                    prefix += ")"
                elif journal:
                    prefix += f" ({journal})"
                prefix += f"\nDOI: {doi}"
                prefix += f"\nSection: {section_path}"
                prefix += "\n---\n"
                
                text_with_prefix = prefix + chunk_text
                chunk_id = f"{paper_id}__{content_hash[:12]}"
                
                chunks_out.append({
                    "chunk_id": chunk_id,
                    "paper_id": doi,
                    "doi": doi,
                    "title": title,
                    "year": int(year) if str(year).isdigit() else year,
                    "journal": journal,
                    "tier": tier,
                    "section_path": section_path,
                    "section_name": section.name,
                    "chunk_type": chunk_type,
                    "chunk_index": chunk_counter,
                    "token_count": count_tokens(text_with_prefix),
                    "text_with_prefix": text_with_prefix,
                    "text_raw": chunk_text,
                })
                chunk_counter += 1
        
        # ── Table chunks ──
        for j, tbl in enumerate(section.tables):
            tbl_text = tbl.get("text", "")
            if not tbl_text or count_tokens(tbl_text) < 10:
                continue
            
            caption = ""
            if j < len(section.captions):
                caption = section.captions[j]
            
            context = f"[TABLE in section: {section_path}]"
            if caption:
                context += f"\nCaption: {caption}"
            full_table = context + "\n\n" + tbl_text
            
            prefix = f'Paper: "{title}"' if title else f'Paper: {paper_id}'
            if year:
                prefix += f" ({year}"
                if journal:
                    prefix += f", {journal}"
                prefix += ")"
            prefix += f"\nDOI: {doi}"
            prefix += f"\nSection: {section_path}"
            prefix += "\n---\n"
            
            text_with_prefix = prefix + full_table
            chunk_id_raw = f"{doi}__table__{section_path}__{j}"
            chunk_id = hashlib.md5(chunk_id_raw.encode()).hexdigest()[:12]
            chunk_id = f"{paper_id}__tbl_{chunk_id}"
            
            # Truncate oversized tables
            if count_tokens(tbl_text) > TABLE_MAX_TOKENS:
                tbl_lines = tbl_text.split('\n')
                kept = []
                tok_count = 0
                for tl in tbl_lines:
                    lt = count_tokens(tl)
                    if tok_count + lt > TABLE_MAX_TOKENS - 20:
                        break
                    kept.append(tl)
                    tok_count += lt
                tbl_text = '\n'.join(kept) + f"\n[... TABLE TRUNCATED ...]"
                full_table = context + "\n\n" + tbl_text
                text_with_prefix = prefix + full_table
            
            chunks_out.append({
                "chunk_id": chunk_id,
                "paper_id": doi,
                "doi": doi,
                "title": title,
                "year": int(year) if str(year).isdigit() else year,
                "journal": journal,
                "tier": tier,
                "section_path": section_path,
                "section_name": section.name,
                "chunk_type": "table",
                "chunk_index": chunk_counter,
                "token_count": count_tokens(text_with_prefix),
                "text_with_prefix": text_with_prefix,
                "text_raw": full_table,
            })
            chunk_counter += 1
        
        # ── Standalone captions ──
        unpaired = section.captions[len(section.tables):]
        for k, cap in enumerate(unpaired):
            if count_tokens(cap) < 10:
                continue
            if is_url_only(cap):
                continue
            cap = clean_ui_from_text(cap)
            if not cap or count_tokens(cap) < 10:
                continue
            cap = fix_ocr_ris_stripping(cap)
            
            prefix = f'Paper: "{title}"' if title else f'Paper: {paper_id}'
            if year:
                prefix += f" ({year}"
                if journal:
                    prefix += f", {journal}"
                prefix += ")"
            prefix += f"\nDOI: {doi}"
            prefix += f"\nSection: {section_path}"
            prefix += "\n---\n"
            
            text_with_prefix = prefix + f"[FIGURE CAPTION]\n{cap}"
            chunk_id_raw = f"{doi}__caption__{section_path}__{k}"
            chunk_id = hashlib.md5(chunk_id_raw.encode()).hexdigest()[:12]
            chunk_id = f"{paper_id}__cap_{chunk_id}"
            
            chunks_out.append({
                "chunk_id": chunk_id,
                "paper_id": doi,
                "doi": doi,
                "title": title,
                "year": int(year) if str(year).isdigit() else year,
                "journal": journal,
                "tier": tier,
                "section_path": section_path,
                "section_name": section.name,
                "chunk_type": "caption",
                "chunk_index": chunk_counter,
                "token_count": count_tokens(text_with_prefix),
                "text_with_prefix": text_with_prefix,
                "text_raw": cap,
            })
            chunk_counter += 1
    
    # ── Post-process: merge tiny chunks ──
    MIN_CAPTION_STANDALONE = 150
    merged = []
    ii = 0
    while ii < len(chunks_out):
        chunk = chunks_out[ii]
        if (chunk["token_count"] < MIN_TOKENS and
                chunk["chunk_type"] == "text" and
                ii + 1 < len(chunks_out) and
                chunks_out[ii + 1]["section_path"] == chunk["section_path"] and
                chunks_out[ii + 1]["chunk_type"] == "text"):
            next_chunk = chunks_out[ii + 1]
            merged_text = chunk["text_raw"] + "\n\n" + next_chunk["text_raw"]
            merged_prefix = chunk["text_with_prefix"].split("---\n", 1)[0] + "---\n" + merged_text
            next_chunk["text_raw"] = merged_text
            next_chunk["text_with_prefix"] = merged_prefix
            next_chunk["token_count"] = count_tokens(merged_prefix)
            ii += 1
        elif (chunk["chunk_type"] == "caption" and
              chunk["token_count"] < MIN_CAPTION_STANDALONE and
              merged and
              merged[-1]["section_path"] == chunk["section_path"] and
              merged[-1]["chunk_type"] == "text"):
            prev = merged[-1]
            appended_text = prev["text_raw"] + "\n\n[Figure caption: " + chunk["text_raw"] + "]"
            appended_prefix = prev["text_with_prefix"].split("---\n", 1)[0] + "---\n" + appended_text
            prev["text_raw"] = appended_text
            prev["text_with_prefix"] = appended_prefix
            prev["token_count"] = count_tokens(appended_prefix)
            ii += 1
        else:
            merged.append(chunk)
            ii += 1
    
    for idx, c in enumerate(merged):
        c["chunk_index"] = idx
    
    return merged


# ── Sentence Splitter ──────────────────────────────────────────────────────────

_sent_re = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'    # Split after sentence-ending punct + space + capital
    r'|(?<=[.!?])\s*\n'          # or after punct + newline
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (conservative)."""
    parts = _sent_re.split(text)
    return [s.strip() for s in parts if s.strip()]


# ── Smart Chunking ─────────────────────────────────────────────────────────────

def chunk_text_block(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Split a text block into chunks respecting sentence boundaries."""
    sentences = split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        # If single sentence exceeds max, force-split by words
        if sent_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            words = sent.split()
            buf = []
            buf_tokens = 0
            for w in words:
                wt = count_tokens(w + " ")
                if buf_tokens + wt > max_tokens and buf:
                    chunks.append(" ".join(buf))
                    buf = []
                    buf_tokens = 0
                buf.append(w)
                buf_tokens += wt
            if buf:
                chunks.append(" ".join(buf))
            continue

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def add_overlap(chunks: list[str], ratio: float = OVERLAP_RATIO) -> list[str]:
    """Add sentence-level overlap between consecutive chunks."""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_sents = split_sentences(chunks[i - 1])
        if not prev_sents:
            result.append(chunks[i])
            continue

        # Take last N sentences as overlap
        overlap_tokens_target = int(count_tokens(chunks[i - 1]) * ratio)
        overlap_sents = []
        overlap_tokens = 0
        for s in reversed(prev_sents):
            st = count_tokens(s)
            if overlap_tokens + st > overlap_tokens_target and overlap_sents:
                break
            overlap_sents.insert(0, s)
            overlap_tokens += st

        overlap_text = " ".join(overlap_sents)
        result.append(overlap_text + " " + chunks[i])

    return result


def table_to_text(table_data: dict, max_tokens: int = TABLE_MAX_TOKENS) -> str:
    """Convert a Docling table to text representation, truncating if needed."""
    if not table_data:
        return ""

    raw = ""
    # Try to extract grid data
    grid = table_data.get("data", {}).get("grid", [])
    if grid:
        lines = []
        for row in grid:
            cells = []
            for cell in row:
                text = cell.get("text", "")
                cells.append(text)
            lines.append(" | ".join(cells))
        raw = "\n".join(lines)
    else:
        # Fallback: use text or markdown representation
        raw = table_data.get("text", "")

    if not raw:
        return "[Table content not extractable]"

    # Truncate oversized tables
    if count_tokens(raw) > max_tokens:
        # Keep first N rows that fit within token budget
        lines = raw.split("\n")
        kept = []
        tok_count = 0
        for line in lines:
            lt = count_tokens(line)
            if tok_count + lt > max_tokens - 20:  # leave room for truncation marker
                break
            kept.append(line)
            tok_count += lt
        raw = "\n".join(kept) + f"\n[... TABLE TRUNCATED, {len(lines) - len(kept)} more rows ...]"

    return raw


# ── Main Chunking Pipeline ─────────────────────────────────────────────────────

def chunk_document(docling_path: Path, meta_path: Path) -> list[dict]:
    """Chunk a single document into retrieval-ready pieces."""
    # Load docling JSON (raw, fast)
    with open(docling_path) as f:
        doc = json.load(f)

    # Load metadata
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    doi = meta.get("doi", docling_path.stem.replace("_", "/", 1)
                   .replace("_", ".", 1))
    title = meta.get("title", doc.get("name", "Unknown"))
    year = meta.get("year", "")
    journal = meta.get("journal", "")
    tier = meta.get("tier", "CORE")
    paper_id = doi

    # Build tree and collect sections
    tree = build_tree(doc)
    sections = collect_sections(tree)

    chunks_out = []
    chunk_counter = 0
    seen_hashes = set()  # V5: content-hash deduplication

    for section in sections:
        # Skip excluded sections
        if section.paragraphs == ["__EXCLUDED__"]:
            continue

        section_path = section.path
        # V5: Fix garbage section paths
        if is_garbage_section_path(section.name):
            section_path = f"[section unknown]"

        # ── Text chunks ──
        if section.paragraphs:
            full_text = "\n\n".join(section.paragraphs)
            # V5: Apply OCR "ris" restoration
            full_text = fix_ocr_ris_stripping(full_text)
            text_tokens = count_tokens(full_text)

            # Determine chunk type
            is_abstract = section.name.strip().lower() == "abstract"
            chunk_type = "abstract" if is_abstract else "text"

            if text_tokens <= MAX_TOKENS:
                # Section fits in one chunk
                raw_chunks = [full_text]
            else:
                # Split by sentence boundaries
                raw_chunks = chunk_text_block(full_text, MAX_TOKENS)

            # Add overlap between chunks in same section
            if len(raw_chunks) > 1:
                raw_chunks = add_overlap(raw_chunks, OVERLAP_RATIO)

            # Hard cap: re-split any chunks that exceeded MAX_TOKENS after overlap
            capped = []
            for rc in raw_chunks:
                if count_tokens(rc) > MAX_TOKENS + 50:  # small slack
                    capped.extend(chunk_text_block(rc, MAX_TOKENS))
                else:
                    capped.append(rc)
            raw_chunks = capped

            for i, chunk_text in enumerate(raw_chunks):
                ct = count_tokens(chunk_text)
                if ct < MIN_QUALITY_TOKENS:
                    continue

                # V2: Clean UI remnants from text
                chunk_text = clean_ui_from_text(chunk_text)
                if not chunk_text or count_tokens(chunk_text) < MIN_QUALITY_TOKENS:
                    continue

                # V2: Skip figure axis gibberish
                if is_figure_axis_gibberish(chunk_text):
                    continue

                # V6: Skip digit-heavy garbage (axis data, coordinate grids)
                if is_digit_heavy_garbage(chunk_text):
                    continue

                # V6: Skip publisher boilerplate
                if is_boilerplate_noise(chunk_text):
                    continue

                # V2: Skip affiliation fragments
                if is_affiliation_fragment(chunk_text):
                    continue

                # V3/V6: Skip reference blocks (refs hidden under wrong section headers)
                if is_reference_block(chunk_text):
                    continue

                # V3: Skip repeating string loops
                if has_repeating_loop(chunk_text):
                    continue

                # V5: Content-hash deduplication
                content_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                # Build metadata prefix
                prefix = f'Paper: "{title}"'
                if year:
                    prefix += f" ({year}"
                    if journal:
                        prefix += f", {journal}"
                    prefix += ")"
                elif journal:
                    prefix += f" ({journal})"
                prefix += f"\nDOI: {doi}"
                prefix += f"\nSection: {section_path}"
                prefix += "\n---\n"

                text_with_prefix = prefix + chunk_text

                # V5: Generate content-based chunk ID (stable, dedup-safe)
                chunk_id = f"{doi.replace('/', '_').replace('.', '_')}__{content_hash[:12]}"

                chunks_out.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "doi": doi,
                    "title": title,
                    "year": int(year) if str(year).isdigit() else year,
                    "journal": journal,
                    "tier": tier,
                    "section_path": section_path,
                    "section_name": section.name,
                    "chunk_type": chunk_type,
                    "chunk_index": chunk_counter,
                    "token_count": count_tokens(text_with_prefix),
                    "text_with_prefix": text_with_prefix,
                    "text_raw": chunk_text,
                })
                chunk_counter += 1

        # ── Table chunks (atomic) ──
        for j, tbl in enumerate(section.tables):
            tbl_text = table_to_text(tbl)
            if not tbl_text or count_tokens(tbl_text) < 10:
                continue

            # Include relevant captions
            caption = ""
            if j < len(section.captions):
                caption = section.captions[j]

            context = f"[TABLE in section: {section_path}]"
            if caption:
                context += f"\nCaption: {caption}"
            full_table = context + "\n\n" + tbl_text

            prefix = f'Paper: "{title}"'
            if year:
                prefix += f" ({year}"
                if journal:
                    prefix += f", {journal}"
                prefix += ")"
            prefix += f"\nDOI: {doi}"
            prefix += f"\nSection: {section_path}"
            prefix += "\n---\n"

            text_with_prefix = prefix + full_table
            chunk_id_raw = f"{paper_id}__table__{section_path}__{j}"
            chunk_id = hashlib.md5(chunk_id_raw.encode()).hexdigest()[:12]
            chunk_id = f"{doi.replace('/', '_').replace('.', '_')}__tbl_{chunk_id}"

            chunks_out.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "doi": doi,
                "title": title,
                "year": int(year) if str(year).isdigit() else year,
                "journal": journal,
                "tier": tier,
                "section_path": section_path,
                "section_name": section.name,
                "chunk_type": "table",
                "chunk_index": chunk_counter,
                "token_count": count_tokens(text_with_prefix),
                "text_with_prefix": text_with_prefix,
                "text_raw": full_table,
            })
            chunk_counter += 1

        # ── Standalone captions (not paired with tables) ──
        unpaired_captions = section.captions[len(section.tables):]
        for k, cap in enumerate(unpaired_captions):
            if count_tokens(cap) < 10:
                continue

            # V2: Skip URL-only captions
            if is_url_only(cap):
                continue

            # V2: Clean UI from captions
            cap = clean_ui_from_text(cap)
            if not cap or count_tokens(cap) < 10:
                continue

            # V5: Apply OCR fix to captions
            cap = fix_ocr_ris_stripping(cap)

            prefix = f'Paper: "{title}"'
            if year:
                prefix += f" ({year}"
                if journal:
                    prefix += f", {journal}"
                prefix += ")"
            prefix += f"\nDOI: {doi}"
            prefix += f"\nSection: {section_path}"
            prefix += "\n---\n"

            text_with_prefix = prefix + f"[FIGURE CAPTION]\n{cap}"
            chunk_id_raw = f"{paper_id}__caption__{section_path}__{k}"
            chunk_id = hashlib.md5(chunk_id_raw.encode()).hexdigest()[:12]
            chunk_id = f"{doi.replace('/', '_').replace('.', '_')}__cap_{chunk_id}"

            chunks_out.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "doi": doi,
                "title": title,
                "year": int(year) if str(year).isdigit() else year,
                "journal": journal,
                "tier": tier,
                "section_path": section_path,
                "section_name": section.name,
                "chunk_type": "caption",
                "chunk_index": chunk_counter,
                "token_count": count_tokens(text_with_prefix),
                "text_with_prefix": text_with_prefix,
                "text_raw": cap,
            })
            chunk_counter += 1

    # ── Post-process: merge tiny chunks with neighbors ──
    # V5: Also merge tiny captions into preceding text chunks
    MIN_CAPTION_STANDALONE = 150  # captions below this get merged
    merged = []
    i = 0
    while i < len(chunks_out):
        chunk = chunks_out[i]
        # Merge tiny TEXT chunks into next text chunk in same section
        if (chunk["token_count"] < MIN_TOKENS and
                chunk["chunk_type"] == "text" and
                i + 1 < len(chunks_out) and
                chunks_out[i + 1]["section_path"] == chunk["section_path"] and
                chunks_out[i + 1]["chunk_type"] == "text"):
            next_chunk = chunks_out[i + 1]
            merged_text = chunk["text_raw"] + "\n\n" + next_chunk["text_raw"]
            merged_prefix = chunk["text_with_prefix"].split("---\n", 1)[0] + "---\n" + merged_text
            next_chunk["text_raw"] = merged_text
            next_chunk["text_with_prefix"] = merged_prefix
            next_chunk["token_count"] = count_tokens(merged_prefix)
            i += 1  # skip current, will pick up merged next
        # V5: Merge tiny CAPTION chunks into preceding text chunk
        elif (chunk["chunk_type"] == "caption" and
              chunk["token_count"] < MIN_CAPTION_STANDALONE and
              merged and
              merged[-1]["section_path"] == chunk["section_path"] and
              merged[-1]["chunk_type"] == "text"):
            prev = merged[-1]
            appended_text = prev["text_raw"] + "\n\n[Figure caption: " + chunk["text_raw"] + "]"
            appended_prefix = prev["text_with_prefix"].split("---\n", 1)[0] + "---\n" + appended_text
            prev["text_raw"] = appended_text
            prev["text_with_prefix"] = appended_prefix
            prev["token_count"] = count_tokens(appended_prefix)
            i += 1  # skip caption, it's merged
        else:
            merged.append(chunk)
            i += 1

    # Re-index
    for idx, c in enumerate(merged):
        c["chunk_index"] = idx

    return merged


# ── Main ───────────────────────────────────────────────────────────────────────

def find_markdown_papers(input_dir: Path) -> list[tuple[str, Path]]:
    """Find all MinerU VLM markdown files in parsed_levante or cleaned_levante structure.
    
    Structure 1 (parsed): {input_dir}/{DOI_folder}/vlm/{DOI_folder}.md
    Structure 2 (cleaned): {input_dir}/{DOI_filename}.md
    Returns: list of (paper_id, md_path) tuples
    """
    papers = []
    for item in sorted(input_dir.iterdir()):
        if item.is_dir():
            paper_id = item.name
            # Look for vlm/*.md
            vlm_dir = item / "vlm"
            if vlm_dir.is_dir():
                md_files = list(vlm_dir.glob("*.md"))
                if md_files:
                    papers.append((paper_id, md_files[0]))
            else:
                # Maybe .md directly in the folder
                md_files = list(item.glob("*.md"))
                if md_files:
                    papers.append((paper_id, md_files[0]))
        elif item.is_file() and item.suffix == ".md":
            # Flat directory structure (cleaned_levante)
            paper_id = item.stem
            papers.append((paper_id, item))
    return papers


def print_summary(all_chunks: list, total_papers: int, errors: list, output_path: Path):
    """Print chunking summary statistics."""
    print(f"\n{'='*60}")
    print(f"CHUNKING COMPLETE")
    print(f"{'='*60}")
    print(f"Papers processed: {total_papers - len(errors)}/{total_papers}")
    print(f"Errors: {len(errors)}")
    print(f"Total chunks: {len(all_chunks)}")

    if all_chunks:
        tokens = [c["token_count"] for c in all_chunks]
        print(f"Token range: {min(tokens)}-{max(tokens)}")
        print(f"Mean tokens: {sum(tokens)/len(tokens):.0f}")
        print(f"Median tokens: {sorted(tokens)[len(tokens)//2]}")

        buckets = {"<100": 0, "100-200": 0, "200-500": 0, "500-1000": 0,
                   "1000-1200": 0, ">1200": 0}
        for t in tokens:
            if t < 100: buckets["<100"] += 1
            elif t < 200: buckets["100-200"] += 1
            elif t < 500: buckets["200-500"] += 1
            elif t < 1000: buckets["500-1000"] += 1
            elif t < 1200: buckets["1000-1200"] += 1
            else: buckets[">1200"] += 1
        print(f"\nToken distribution:")
        for k, v in buckets.items():
            pct = v / len(tokens) * 100
            bar = "█" * int(pct / 2)
            print(f"  {k:>10}: {v:5d} ({pct:5.1f}%) {bar}")

        types = {}
        for c in all_chunks:
            ct = c["chunk_type"]
            types[ct] = types.get(ct, 0) + 1
        print(f"\nChunk types:")
        for ct, count in sorted(types.items()):
            print(f"  {ct}: {count}")

        papers = set(c["paper_id"] for c in all_chunks)
        print(f"\nUnique papers: {len(papers)}")

    if errors:
        print(f"\nFailed papers ({len(errors)}):")
        for stem, err in errors[:20]:
            print(f"  {stem}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CMIP6 RAG Chunker V6")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory (auto-detects docling vs markdown)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file path")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Max papers to process (for testing)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else None

    # ── Auto-detect mode ──
    use_markdown = False
    if input_dir:
        # Check if input_dir has VLM markdown structure
        test_dirs = [d for d in input_dir.iterdir() if d.is_dir()][:5]
        has_vlm = any((d / "vlm").is_dir() for d in test_dirs)
        has_md_nested = any(list(d.glob("*.md")) for d in test_dirs if d.is_dir())
        has_md_flat = any(list(input_dir.glob("*.md"))[:1])
        has_docling = any(list(input_dir.glob("*.docling.json"))[:1])

        if has_vlm or (has_md_nested and not has_docling) or (has_md_flat and not has_docling):
            use_markdown = True
    else:
        # Default: try parsed_levante first, then parsed
        if PARSED_LEVANTE_DIR.exists() and any(PARSED_LEVANTE_DIR.iterdir()):
            input_dir = PARSED_LEVANTE_DIR
            use_markdown = True
        else:
            input_dir = PARSED_DIR
            use_markdown = False

    if use_markdown:
        # ── MinerU VLM Markdown Mode ──
        output_path = Path(args.output) if args.output else OUTPUT_LEVANTE_FILE
        papers = find_markdown_papers(input_dir)
        if args.max_papers:
            papers = papers[:args.max_papers]
        print(f"Mode: MinerU VLM Markdown")
        print(f"Found {len(papers)} papers in {input_dir}")
        print(f"Output: {output_path}")
        print()

        if not papers:
            print("ERROR: No markdown files found!")
            return

        all_chunks = []
        errors = []

        for i, (paper_id, md_path) in enumerate(papers):
            try:
                chunks = chunk_markdown_document(md_path, paper_id)
                all_chunks.extend(chunks)
                tokens = [c["token_count"] for c in chunks]
                avg_t = sum(tokens) / len(tokens) if tokens else 0
                if (i + 1) % 100 == 0 or (i + 1) == len(papers) or i < 5:
                    print(f"[{i+1:5d}/{len(papers)}] {paper_id}: "
                          f"{len(chunks)} chunks, "
                          f"avg {avg_t:.0f} tokens")
            except Exception as e:
                print(f"[{i+1:5d}/{len(papers)}] ERROR {paper_id}: {e}")
                errors.append((paper_id, str(e)))

        with open(output_path, "w") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print_summary(all_chunks, len(papers), errors, output_path)

    else:
        # ── Original Docling JSON Mode ──
        output_path = Path(args.output) if args.output else OUTPUT_FILE
        docling_files = sorted(input_dir.glob("*.docling.json"))
        if args.max_papers:
            docling_files = docling_files[:args.max_papers]
        print(f"Mode: Docling JSON")
        print(f"Found {len(docling_files)} docling files in {input_dir}")

        if not docling_files:
            print("ERROR: No .docling.json files found!")
            return

        all_chunks = []
        errors = []

        for i, dp in enumerate(docling_files):
            stem = dp.name.replace(".docling.json", "")
            meta_path = input_dir / f"{stem}.json"

            try:
                chunks = chunk_document(dp, meta_path)
                all_chunks.extend(chunks)
                tokens = [c["token_count"] for c in chunks]
                avg_t = sum(tokens) / len(tokens) if tokens else 0
                print(f"[{i+1:3d}/{len(docling_files)}] {stem}: "
                      f"{len(chunks)} chunks, "
                      f"avg {avg_t:.0f} tokens, "
                      f"range [{min(tokens) if tokens else 0}-{max(tokens) if tokens else 0}]")
            except Exception as e:
                print(f"[{i+1:3d}/{len(docling_files)}] ERROR {stem}: {e}")
                errors.append((stem, str(e)))

        with open(output_path, "w") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print_summary(all_chunks, len(docling_files), errors, output_path)


if __name__ == "__main__":
    main()
