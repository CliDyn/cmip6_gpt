#!/usr/bin/env python3
"""
MinerU VLM Markdown Preprocessor for CMIP6 RAG Pipeline
========================================================
Strips boilerplate, links images to captions, normalises LaTeX,
and fixes OCR artefacts from MinerU VLM-parsed markdown files.

Input:  rag/parsed_levante/{doi}/vlm/{doi}.md
Output: rag/cleaned_levante/{doi}.md  +  rag/cleaned_levante/{doi}.meta.json

Run:
    python preprocess_markdown.py                        # full run
    python preprocess_markdown.py --limit 10             # test on 10 papers
    python preprocess_markdown.py --paper 10_1002_2015gl064738  # single paper
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RAG_DIR = Path(__file__).parent
INPUT_DIR = RAG_DIR / "parsed_levante"
OUTPUT_DIR = RAG_DIR / "cleaned_levante"

# ── Journal / Publisher Header Patterns ───────────────────────────────────────
# These appear at the very top of MinerU markdown and are pure noise.
JOURNAL_HEADERS = {
    "geophysical research letters",
    "journal of geophysical research",
    "journal of geophysical research: atmospheres",
    "journal of geophysical research: oceans",
    "journal of geophysical research: biogeosciences",
    "journal of advances in modeling earth systems",
    "reviews of geophysics",
    "global biogeochemical cycles",
    "water resources research",
    "earth's future",
    "geochemistry, geophysics, geosystems",
    "geoscientific model development",
    "earth system dynamics",
    "earth system science data",
    "atmospheric chemistry and physics",
    "biogeosciences",
    "climate of the past",
    "the cryosphere",
    "hydrology and earth system sciences",
    "ocean science",
    "nature",
    "nature climate change",
    "nature communications",
    "nature geoscience",
    "nature food",
    "nature reviews earth & environment",
    "nature energy",
    "nature sustainability",
    "nature water",
    "scientific data",
    "scientific reports",
    "communications earth & environment",
    "npj climate and atmospheric science",
    "environmental research letters",
    "global change biology",
    "journal of climate",
    "monthly weather review",
    "journal of the atmospheric sciences",
    "journal of physical oceanography",
    "journal of hydrometeorology",
    "bulletin of the american meteorological society",
    "climate dynamics",
    "climatic change",
    "international journal of climatology",
    "pnas",
    "pnas nexus",
    "proceedings of the national academy of sciences",
    "science",
    "science advances",
    "one earth",
    "cell reports sustainability",
    "the lancet planetary health",
    "annual review of environment and resources",
    "annual review of marine science",
    "frontiers in climate",
    "frontiers in earth science",
    "frontiers in marine science",
    "environmental research: climate",
    "weather and climate extremes",
    "global and planetary change",
    "quaternary science reviews",
    "earth-science reviews",
    "progress in oceanography",
    "deep-sea research",
    "tellus",
    "remote sensing of environment",
}

# Article type labels at the top
ARTICLE_TYPE_LABELS = {
    "research letter", "research letters", "research article",
    "original article", "article", "letter", "letters",
    "review article", "review", "brief communication",
    "rapid communication", "commentary", "perspective",
    "technical note", "discussion paper", "editorial",
    "research paper", "report", "analysis",
}

# ── Preamble Noise Patterns ──────────────────────────────────────────────────
# Regex patterns for lines that should be stripped from the preamble
# (before the real title/abstract)
PREAMBLE_NOISE_RE = [
    re.compile(r'^\d{2}\.\d{4}/\S+', re.IGNORECASE),           # DOI: 10.1002/...
    re.compile(r'^doi:\s*\d{2}\.\d{4}', re.IGNORECASE),         # doi: 10.1002/...
    re.compile(r'^https?://doi\.org/', re.IGNORECASE),           # https://doi.org/...
    re.compile(r'^Received\s+\d+', re.IGNORECASE),               # Received 28 MAY 2015
    re.compile(r'^Received:', re.IGNORECASE),                     # Received: 18 Oct 2022
    re.compile(r'^Accepted\s+\d+', re.IGNORECASE),               # Accepted 17 JUL 2015
    re.compile(r'^Accepted:', re.IGNORECASE),                     # Accepted: ...
    re.compile(r'^Published\s+(?:online\s*)?[:\d]', re.IGNORECASE),# Published online 6 AUG
    re.compile(r'^Accepted\s+article\s+online', re.IGNORECASE),  # Accepted article online
    re.compile(r'^First\s+published', re.IGNORECASE),            # First published: ...
    re.compile(r'^Revised:\s', re.IGNORECASE),                   # Revised: ...
    re.compile(r'^©\s*\d{4}', re.IGNORECASE),                    # ©2015. American...
    re.compile(r'^Copyright\s+[©:]', re.IGNORECASE),             # Copyright © 2020
    re.compile(r'^All Rights Reserved', re.IGNORECASE),          # All Rights Reserved.
    re.compile(r'^\*\s*Correspondence', re.IGNORECASE),          # * Correspondence: ...
    re.compile(r'^Correspondence\s+(?:to|and)', re.IGNORECASE),  # Correspondence to:
    re.compile(r'^Edited\s+by:', re.IGNORECASE),                 # Edited by: ...
    re.compile(r'^Reviewed\s+by:', re.IGNORECASE),               # Reviewed by: ...
    re.compile(r'^Editor:', re.IGNORECASE),                      # Editor: ...
    re.compile(r'^Volume\s+\d+', re.IGNORECASE),                 # Volume 42, ...
    re.compile(r'^\d+\s+pages?\b', re.IGNORECASE),               # 9 pages
    re.compile(r'^Open\s+Access', re.IGNORECASE),                # Open Access
    re.compile(r'^Check\s+for\s+updates', re.IGNORECASE),        # Check for updates
    re.compile(r'^CrossMark', re.IGNORECASE),                    # CrossMark
    re.compile(r'^Cite\s+this:', re.IGNORECASE),                 # Cite this: ...
    re.compile(r'^How\s+to\s+cite', re.IGNORECASE),              # How to cite
    re.compile(r'^ISSN\s+\d', re.IGNORECASE),                    # ISSN 1234-5678
    # ── T&F / IOP / Springer specific ──
    re.compile(r'^Submit\s+your\s+article', re.IGNORECASE),      # Submit your article to...
    re.compile(r'^Article\s+views:', re.IGNORECASE),             # Article views: 2807
    re.compile(r'^View\s+(?:related|Crossmark)\s+data', re.IGNORECASE),
    re.compile(r'^Citing\s+articles:', re.IGNORECASE),           # Citing articles: ...
    re.compile(r'^View the article online', re.IGNORECASE),      # View the article online
    re.compile(r'^Full Terms \& Conditions', re.IGNORECASE),     # Full Terms & Conditions
    re.compile(r'^To link to this article', re.IGNORECASE),      # To link to this article
    re.compile(r'^To cite this article', re.IGNORECASE),         # To cite this article
    re.compile(r'^LETTER\b.*OPEN\s+ACCESS', re.IGNORECASE),      # LETTER • OPEN ACCESS
    re.compile(r'^Original\s+[Cc]ontent\s+from', re.IGNORECASE), # Original Content from...
    re.compile(r'^Content\s+from\s+this\s+work', re.IGNORECASE), # Content from this work...
    # Government / institutional disclaimers
    re.compile(r'^This document was prepared as an account', re.IGNORECASE),
    re.compile(r'^Neither the United States government', re.IGNORECASE),
    re.compile(r'^\(Manuscript received', re.IGNORECASE),            # (Manuscript received 9 July 2020...)
    re.compile(r'^Manuscript received', re.IGNORECASE),
    re.compile(r'^Supplementary material for this article', re.IGNORECASE),
]

# ── Sections to Strip Entirely ────────────────────────────────────────────────
# These sections (matched by header text) are removed with all their content.
STRIP_SECTIONS = {
    # Front matter
    "key points", "key points:",
    "supporting information", "supporting information:",
    "correspondence to", "correspondence to:",
    "correspondence", "correspondence:",
    "citation", "citation:",
    # Back matter
    "reference", "references", "bibliography",
    "references from the supporting information",
    "acknowledgements", "acknowledgments", "acknowledgement", "acknowledgment",
    "author contributions", "contributions",
    "author information", "authors and affiliations",
    "competing interests", "conflict of interest", "conflicts of interest",
    "declaration of competing interest", "declaration of interests",
    "declaration of competing interests",
    "disclosure statement", "disclosure",
    "additional information", "supplementary information",
    "supplementary material", "supplementary materials",
    "supplementary data", "supplement",
    "data availability", "data availability statement",
    "code availability", "code and data availability",
    "availability of data and materials",
    "ethics declarations", "declarations",
    "ethics approval", "ethics statement",
    "compliance with ethical standards",
    "human and animal rights",
    "rights and permissions",
    "about this article", "cite this article",
    "funding", "financial support",
    "review statement", "disclaimer", "change history",
    "peer review information", "peer review",
    "publisher's note", "publisher\u2019s note",
    "submission history",
    "notes",
    "open access",
    "orcid ids", "orcids", "orcidids",
    "keywords", "keywords:",
    "erratum", "corrigendum", "correction",
    "general rights",
    "affiliations", "affiliations:",
    "creditauthorship contribution statement",
    # Web noise
    "explore content", "about the journal", "publish with us",
    "similar content being viewed by others",
    "you may also like",
}

# ── Tail Noise Patterns ──────────────────────────────────────────────────────
# Lines at the bottom of the file (after last real section) that are noise
TAIL_NOISE_RE = [
    re.compile(r'^Open\s+Access\s+This\s+article', re.IGNORECASE),
    re.compile(r'^This\s+article\s+is\s+licensed\s+under', re.IGNORECASE),
    re.compile(r'^Creative\s+Commons', re.IGNORECASE),
    re.compile(r'^Attribution\s+[34]\.0', re.IGNORECASE),
    re.compile(r'^Reprints\s+and\s+permission', re.IGNORECASE),
    re.compile(r'^Springer\s+Nature', re.IGNORECASE),
    re.compile(r'^©\s*The\s+Author', re.IGNORECASE),
    re.compile(r'^©\s*\d{4}', re.IGNORECASE),
    re.compile(r'^Copyright\s+[©:]', re.IGNORECASE),
    re.compile(r'^Copyright:\s+©', re.IGNORECASE),
    re.compile(r'^Peer\s+review\s+information', re.IGNORECASE),
    re.compile(r'^Publisher.s\s+note', re.IGNORECASE),
    re.compile(r'^Correspondence\s+and\s+requests', re.IGNORECASE),
    re.compile(r'^The\s+Editor\s+thanks', re.IGNORECASE),
    re.compile(r'^Original\s+[Cc]ontent\s+from', re.IGNORECASE),
    re.compile(r'^Content\s+from\s+this\s+work', re.IGNORECASE),
    re.compile(r'^This\s+is\s+an\s+open\s+access', re.IGNORECASE),
    re.compile(r'^Full\s+Terms\s+\\&\s+Conditions', re.IGNORECASE),
    re.compile(r'^Submit\s+your\s+article', re.IGNORECASE),
    re.compile(r'^Article\s+views:', re.IGNORECASE),
    re.compile(r'^View\s+related\s+articles', re.IGNORECASE),
    re.compile(r'^View\s+Crossmark\s+data', re.IGNORECASE),
    re.compile(r'^Citing\s+articles:', re.IGNORECASE),
    # CC license text blocks (Nature/Springer/IOP/PLOS)
    re.compile(r'^Licensed\s+under\s+a\s+Creative', re.IGNORECASE),
    re.compile(r'^which\s+permits\s+unrestricted\s+use', re.IGNORECASE),
    re.compile(r'^distribution,?\s+and\s+reproduction\s+in\s+any\s+medium', re.IGNORECASE),
    re.compile(r'^provided\s+the\s+original\s+(?:work|author)', re.IGNORECASE),
    re.compile(r'^The\s+images\s+or\s+other\s+third\s+party', re.IGNORECASE),
    re.compile(r'^material\s+in\s+this\s+article', re.IGNORECASE),
    re.compile(r'^unless\s+indicated\s+otherwise', re.IGNORECASE),
    re.compile(r"^article's\s+Creative\s+Commons", re.IGNORECASE),
    re.compile(r'^you\s+will\s+need\s+to\s+obtain\s+permission', re.IGNORECASE),
    re.compile(r'^the\s+copyright\s+holder', re.IGNORECASE),
    re.compile(r'^To\s+view\s+a\s+copy\s+of\s+this\s+licen[sc]e', re.IGNORECASE),
    re.compile(r'^http://creativecommons\.org/', re.IGNORECASE),
    re.compile(r'^https?://creativecommons\.org/', re.IGNORECASE),
    # npj / Nature journal citation line
    re.compile(r'^npj\s+\w', re.IGNORECASE),
    re.compile(r'^Published\s+in\s+partnership\s+with', re.IGNORECASE),
]

# ── LaTeX Patterns ────────────────────────────────────────────────────────────
# Decorative LaTeX to strip (affiliations, superscripts)
DECORATIVE_LATEX_RE = re.compile(
    r'\$\^\{?\d+\}?\$'                # $^{1}$ or $^1$ — affiliation superscripts
)

# Common unit normalisations
LATEX_UNIT_SUBS = [
    (re.compile(r'\$\\mathrm\{Gt\}\\mathrm\{yr\}\^\{-1\}\$'), 'Gt/yr'),
    (re.compile(r'\$\\mathrm\{Gt\}\s*\\mathrm\{yr\}\^\{-1\}\$'), 'Gt/yr'),
    (re.compile(r'\$\\mathrm\{W\}\s*\\mathrm\{m\}\^\{-2\}\$'), 'W/m²'),
    (re.compile(r'\$\\mathrm\{Wm\}\^\{-2\}\$'), 'W/m²'),
    (re.compile(r'\\mathrm\{CO\}_\{?2\}?'), 'CO₂'),
    (re.compile(r'\\mathrm\{CO\}_2'), 'CO₂'),
    (re.compile(r'CO\s*\$_\{?2\}?\$'), 'CO₂'),
    (re.compile(r'\$\\mathrm\{CH\}_\{?4\}?\$'), 'CH₄'),
    (re.compile(r'\$\\mathrm\{N\}_\{?2\}?\\mathrm\{O\}\$'), 'N₂O'),
    (re.compile(r'\$\s*°\s*\$'), '°'),
    (re.compile(r'\$\^\{?\\circ\}?\$'), '°'),
    (re.compile(r'\^\{?\\circ\}?'), '°'),
]


def _fix_sim_match(m: re.Match) -> str:
    """Fix $\sim X$ -> ~X or ~$X$ depending on whether X has LaTeX."""
    content = m.group(1).strip()
    if not content:
        return '~'
    if '\\' in content:
        # Content has LaTeX commands — keep $ delimiters
        return '~$' + content + '$'
    return '~' + content


# $\sim X$ — must NOT match $\simeq$, $\similar$, etc.
_SIM_RE = re.compile(r'\$\\sim(?![a-zA-Z])\s*([^$]*)\$')

# ── Image-Caption Linking ─────────────────────────────────────────────────────
IMAGE_RE = re.compile(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$')
FIGURE_CAPTION_RE = re.compile(
    r'^(?:Figure|Fig\.|Plate|Scheme)\s+(\S+?)[\s.:]\s*(.+)',
    re.IGNORECASE
)
# Table captions should NOT be linked to images — tables are text, not images.
# If an image sits next to a Table caption, the image becomes orphaned [IMAGE:].
TABLE_CAPTION_RE = re.compile(
    r'^Table\s+(\S+?)[\s.:]\s*(.+)',
    re.IGNORECASE
)

# ── HTML Table Detection ──────────────────────────────────────────────────────
HTML_TABLE_START_RE = re.compile(r'<table\b', re.IGNORECASE)
HTML_TABLE_END_RE = re.compile(r'</table>', re.IGNORECASE)

# ── Author/Affiliation Block Detection ────────────────────────────────────────
AFFILIATION_RE = re.compile(
    r'(?:Department|School|University|Institute|Laboratory|Center|Centre|'
    r'Faculty|College|Division|Program|NASA|NOAA|CSIRO|CNRS|Max Planck|'
    r'Helmholtz|Leibniz|Alfred Wegener|Met Office|ECMWF)',
    re.IGNORECASE
)
EMAIL_RE = re.compile(r'[\w.+-]+@[\w.-]+\.\w{2,}')

# ── OCR 'ris' Fix (imported from chunk_papers.py, expanded) ──────────────────
# We import the corrections from the chunker to avoid duplication.
# If chunk_papers is not importable, inline a subset.
try:
    sys.path.insert(0, str(RAG_DIR))
    from chunk_papers import fix_ocr_ris_stripping
except ImportError:
    def fix_ocr_ris_stripping(text: str) -> str:
        """Fallback: basic OCR ris fix."""
        subs = [
            ("compaon", "comparison"),
            ("charactetic", "characteristic"),
            ("parameteization", "parameterization"),
            ("parameteize", "parameterize"),
        ]
        for old, new in subs:
            text = text.replace(old, new)
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def is_preamble_noise(line: str) -> bool:
    """Check if a line is preamble noise (DOI, dates, copyright, etc.)."""
    s = line.strip()
    if not s:
        return True
    for pat in PREAMBLE_NOISE_RE:
        if pat.match(s):
            return True
    return False


def is_journal_header(text: str) -> bool:
    """Check if text is a known journal name."""
    return text.strip().lower() in JOURNAL_HEADERS


def is_article_type(text: str) -> bool:
    """Check if text is an article type label."""
    return text.strip().lower() in ARTICLE_TYPE_LABELS


def is_strip_section(header: str) -> bool:
    """Check if a section header should be stripped entirely."""
    return header.strip().lower().rstrip(':') in STRIP_SECTIONS


def is_tail_noise(line: str) -> bool:
    """Check if a line is tail noise (CC license, publisher note, etc.)."""
    s = line.strip()
    for pat in TAIL_NOISE_RE:
        if pat.match(s):
            return True
    return False


def is_affiliation_line(line: str) -> bool:
    """Check if a line looks like an author affiliation block."""
    s = line.strip()
    # Lines with $^{N}$ patterns + institution names
    if DECORATIVE_LATEX_RE.search(s) and AFFILIATION_RE.search(s):
        return True
    # Lines that are mostly institution names
    if AFFILIATION_RE.search(s) and len(s) > 30:
        words = s.split()
        # If >40% of the line is affiliation-like, strip it
        affil_words = sum(1 for w in words if AFFILIATION_RE.match(w))
        if affil_words >= 2:
            return True
    return False


def normalise_latex(text: str) -> str:
    """Normalise common LaTeX patterns to readable text."""
    # Remove decorative superscripts
    text = DECORATIVE_LATEX_RE.sub('', text)
    # Normalise units
    for pat, repl in LATEX_UNIT_SUBS:
        text = pat.sub(repl, text)
    # Fix $\sim X$ → ~X or ~$X$ (safe: won't match $\simeq$ etc.)
    text = _SIM_RE.sub(_fix_sim_match, text)
    return text


def convert_html_table_to_markdown(html: str) -> str:
    """Convert a simple HTML table to markdown table format."""
    try:
        # Strip tags, extract rows
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
        if not rows:
            return html
        
        md_rows = []
        for row in rows:
            cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row, re.DOTALL | re.IGNORECASE)
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if cells:
                md_rows.append('| ' + ' | '.join(cells) + ' |')
        
        if len(md_rows) >= 1:
            # Add separator after first row (header)
            header = md_rows[0]
            n_cols = header.count('|') - 1
            separator = '|' + '|'.join([' --- '] * n_cols) + '|'
            md_rows.insert(1, separator)
        
        return '\n'.join(md_rows)
    except Exception:
        # If conversion fails, return original stripped of tags
        return re.sub(r'<[^>]+>', ' ', html).strip()


def _is_short_interleaved_text(line: str) -> bool:
    """Check if a line is a short label/text that sits between sub-images.
    
    e.g. '(a)', 'ERA5', 'MIROC6', panel labels, axis labels.
    These should NOT break the image grouping.
    """
    s = line.strip()
    if not s:
        return True
    # Short enough to be a label (< 10 words, < 100 chars)
    if len(s) < 100 and len(s.split()) <= 10:
        # Not a header or caption
        if not s.startswith('#') and not FIGURE_CAPTION_RE.match(s):
            return True
    return False


def link_images_to_captions(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Link ![](image) references to adjacent Figure/Table captions.
    
    Tolerates short text lines (axis labels, panel letters) between
    consecutive images in multi-panel figures.
    
    Returns:
        - Modified lines with [FIGURE N: caption | image: path] placeholders
        - List of figure metadata dicts
    """
    result = []
    figures = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        img_match = IMAGE_RE.match(stripped)
        if img_match:
            img_path = img_match.group(2)
            
            # Collect consecutive images, tolerating short interleaved text
            image_paths = [img_path]
            interleaved_texts = []  # short labels between images
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                if not next_stripped:
                    j += 1
                    continue
                next_img = IMAGE_RE.match(next_stripped)
                if next_img:
                    image_paths.append(next_img.group(2))
                    interleaved_texts = []  # reset — text was between images
                    j += 1
                elif _is_short_interleaved_text(lines[j]):
                    # Short text between images — tolerate but track
                    interleaved_texts.append(next_stripped)
                    j += 1
                    # But only tolerate up to 5 consecutive short lines
                    if len(interleaved_texts) > 5:
                        # Too much text — this isn't interleaved labels
                        j -= len(interleaved_texts)
                        interleaved_texts = []
                        break
                else:
                    break
            
            # If we collected interleaved text at the end (not followed by
            # another image), put j back before those lines
            if interleaved_texts:
                j -= len(interleaved_texts)
                interleaved_texts = []
            
            # Check if next non-empty line (or after short text) is a caption
            # Skip over sub-labels like (a), (b), ERA5 etc. to find the caption
            caption = ""
            caption_lines = []
            k = j
            short_skip_count = 0
            while k < len(lines):
                next_stripped = lines[k].strip()
                if not next_stripped:
                    k += 1
                    continue
                cap_match = FIGURE_CAPTION_RE.match(next_stripped)
                if cap_match:
                    # Collect full caption (may span multiple lines)
                    caption_lines.append(next_stripped)
                    k += 1
                    while k < len(lines):
                        cl = lines[k].strip()
                        if not cl:
                            break
                        # If next line is a new header or image, stop
                        if cl.startswith('#') or IMAGE_RE.match(cl):
                            break
                        caption_lines.append(cl)
                        k += 1
                    caption = ' '.join(caption_lines)
                    break
                # Skip short text lines (sub-labels) between images and caption
                if _is_short_interleaved_text(lines[k]):
                    short_skip_count += 1
                    k += 1
                    if short_skip_count > 5:
                        break  # too much text, give up
                    continue
                break
            
            # Build the linked placeholder
            fig_num = ""
            fig_match = FIGURE_CAPTION_RE.match(caption) if caption else None
            if fig_match:
                fig_num = fig_match.group(1).rstrip('.,:')
            
            if caption:
                images_str = ', '.join(image_paths)
                placeholder = f"[FIGURE {fig_num}: {caption} | images: {images_str}]"
                result.append(placeholder)
                figures.append({
                    "figure_id": fig_num,
                    "caption": caption,
                    "images": image_paths,
                })
                i = k  # skip past caption
            else:
                # Orphaned image(s) — no caption found; keep a minimal reference
                images_str = ', '.join(image_paths)
                result.append(f"[IMAGE: {images_str}]")
                figures.append({
                    "figure_id": None,
                    "caption": None,
                    "images": image_paths,
                })
                i = j
        else:
            result.append(line)
            i += 1
    
    return result, figures


def detect_preamble_end(lines: list[str]) -> int:
    """Find where the preamble ends and real content (title/abstract) begins.
    
    Strategy: scan for the first # header that looks like a real paper title
    (not a journal name, article type, or metadata header).
    The abstract or first real header marks the end of the preamble.
    """
    header_re = re.compile(r'^(#{1,6})\s+(.+)$')
    
    # First pass: find the real title
    # The title is usually the first # header that:
    # 1. Is not a journal name
    # 2. Is not an article type
    # 3. Is not a metadata field (Key Points, Correspondence, etc.)
    # 4. Is reasonably long (>3 words for titles)
    
    metadata_headers = {
        "key points", "key points:", "supporting information",
        "supporting information:", "correspondence to", "correspondence to:",
        "correspondence", "correspondence:", "citation", "citation:",
    }
    
    for i, line in enumerate(lines):
        s = line.strip()
        m = header_re.match(s)
        if not m:
            continue
        header_text = m.group(2).strip()
        ht_lower = header_text.lower().rstrip(':')
        
        # Skip journal names
        if is_journal_header(header_text):
            continue
        # Skip article types
        if is_article_type(header_text):
            continue
        # Skip metadata headers
        if ht_lower in metadata_headers:
            continue
        # Skip very short headers that look like DOI or labels
        if len(header_text.split()) <= 2 and not ht_lower.startswith('abstract'):
            continue
        
        # This looks like the real title — return its line number
        return i
    
    # If no clear title found, keep everything from line 0
    return 0


# Extra patterns for author block detection
_ORCID_RE = re.compile(r'orcid\.org/|\$\\mathbb\{O\}|ORCID', re.IGNORECASE)
_AUTHOR_DATE_RE = re.compile(
    r'^(?:Received|Accepted|Published|Revised|Submitted|Available online)[:\s]',
    re.IGNORECASE
)
_CORRESPONDENCE_RE = re.compile(
    r'^(?:Correspondence|Corresponding author|\*Correspondence)[:\s]',
    re.IGNORECASE
)
_AUTHOR_NAME_LINE_RE = re.compile(
    r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*\s*$'
)

def detect_author_block_end(lines: list[str], title_line: int) -> int:
    """Find where the author/affiliation block ends after the title.
    
    After the title, there's usually:
    - Author names (with $^{1}$ superscripts)
    - Affiliation block
    - Correspondence / dates
    - Abstract
    
    Returns the line index where real content (Abstract) begins.
    """
    header_re = re.compile(r'^(#{1,6})\s+(.+)$')
    i = title_line + 1
    
    while i < len(lines):
        s = lines[i].strip()
        
        # Skip empty lines
        if not s:
            i += 1
            continue
        
        # If we hit "Abstract" header, we're done
        m = header_re.match(s)
        if m:
            ht = m.group(2).strip().lower()
            if ht == 'abstract':
                return i
            # If we hit any section header, stop looking for author block
            if not is_strip_section(m.group(2).strip()):
                return i
        
        # Check if this line starts with "Abstract" (without header marker)
        if s.lower().startswith('abstract ') or s.lower() == 'abstract':
            return i
        
        # Check if this line is author names or affiliations
        has_superscript = DECORATIVE_LATEX_RE.search(s)
        has_affiliation = AFFILIATION_RE.search(s)
        has_email = EMAIL_RE.search(s)
        has_orcid = _ORCID_RE.search(s)
        has_date = _AUTHOR_DATE_RE.match(s)
        has_corresp = _CORRESPONDENCE_RE.match(s)
        is_preamble = is_preamble_noise(lines[i])
        
        if has_superscript or has_affiliation or has_email or is_preamble:
            i += 1
            continue
        if has_orcid or has_date or has_corresp:
            i += 1
            continue
        
        # Skip HTML sup tags for affiliations: <sup>1</sup>, <sup>2</sup>
        if re.search(r'<sup>\d+</sup>', s, re.IGNORECASE):
            i += 1
            continue
        
        # Skip short lines that look like author names or metadata
        word_count = len(s.split())
        if word_count <= 15:
            # Skip copyright/license lines
            if s.lower().startswith('copyright') or s.startswith('©'):
                i += 1
                continue
            # Skip "now at:" or "a now at:" lines
            if 'now at:' in s.lower():
                i += 1
                continue
            # Skip lines that are entirely commas/names (author lists)
            # but not scientific text (no periods ending sentences)
            if word_count <= 5 and not s.endswith('.'):
                i += 1
                continue
        
        # If the line is long text without affiliation markers, it might be
        # the start of content (e.g., an abstract without a header)
        if word_count > 15 and not has_affiliation:
            return i
        
        i += 1
    
    return i


def strip_tail_noise(lines: list[str]) -> list[str]:
    """Remove CC license, publisher notes, and other tail noise."""
    # Work backwards from the end
    end = len(lines)
    while end > 0:
        s = lines[end - 1].strip()
        if not s:
            end -= 1
            continue
        if is_tail_noise(s):
            end -= 1
            continue
        # Check for orphaned images at the very end (CC license logo)
        if IMAGE_RE.match(s):
            end -= 1
            continue
        break
    return lines[:end]


# ── Headerless Boilerplate Patterns ───────────────────────────────────────────
# MDPI, IOP, Springer etc. use bold text instead of # headers for back-matter.
# These patterns match lines that START a boilerplate block without a header.
HEADERLESS_BOILERPLATE_RE = [
    # Bold-text section headers (MDPI style)
    re.compile(r'^\*\*Author Contributions?\*\*', re.IGNORECASE),
    re.compile(r'^\*\*Funding\*\*', re.IGNORECASE),
    re.compile(r'^\*\*Data Availability\b', re.IGNORECASE),
    re.compile(r'^\*\*Code Availability\b', re.IGNORECASE),
    re.compile(r'^\*\*Conflicts? of Interest\*\*', re.IGNORECASE),
    re.compile(r'^\*\*Competing Interests\*\*', re.IGNORECASE),
    re.compile(r'^\*\*Institutional Review Board\b', re.IGNORECASE),
    re.compile(r'^\*\*Informed Consent\b', re.IGNORECASE),
    re.compile(r'^\*\*Ethics\b', re.IGNORECASE),
    re.compile(r'^\*\*Acknowledgment', re.IGNORECASE),
    re.compile(r'^\*\*Acknowledgement', re.IGNORECASE),
    re.compile(r'^\*\*Declaration of', re.IGNORECASE),
    re.compile(r'^\*\*Disclosure\b', re.IGNORECASE),
    re.compile(r'^\*\*CRediT\b', re.IGNORECASE),
    re.compile(r'^\*\*Availability of data', re.IGNORECASE),
    # Plain text variants (no bold)
    re.compile(r'^Author Contributions?:', re.IGNORECASE),
    re.compile(r'^Funding:', re.IGNORECASE),
    re.compile(r'^Data [Aa]vailability\b', re.IGNORECASE),
    re.compile(r'^Code [Aa]vailability\b', re.IGNORECASE),
    re.compile(r'^Conflicts? of [Ii]nterest:', re.IGNORECASE),
    re.compile(r'^Institutional Review Board Statement:', re.IGNORECASE),
    re.compile(r'^Informed Consent Statement:', re.IGNORECASE),
    re.compile(r'^Declaration of (?:competing |)interest', re.IGNORECASE),
    re.compile(r'^Disclosure [Ss]tatement', re.IGNORECASE),
    re.compile(r'^CRediT authorship', re.IGNORECASE),
    re.compile(r'^Availability of data', re.IGNORECASE),
    re.compile(r'^Acknowledgments?\.', re.IGNORECASE),  # Acknowledgments. This work...
    re.compile(r'^Acknowledgements?\.', re.IGNORECASE),
    # Publisher disclaimers
    re.compile(r'^Publisher.s Note:', re.IGNORECASE),
    re.compile(r'^Academic Editor', re.IGNORECASE),
    re.compile(r'^Cite this article', re.IGNORECASE),
    re.compile(r'^Citation:', re.IGNORECASE),
    re.compile(r'^We are providing an unedited version', re.IGNORECASE),
    re.compile(r'^If this paper is publishing under', re.IGNORECASE),
    re.compile(r'^To cite this article', re.IGNORECASE),
    re.compile(r'^ORCID iDs?', re.IGNORECASE),
    # IOP / web noise (no header)
    re.compile(r'^You may also like', re.IGNORECASE),
    re.compile(r'^AFFILIATIONS:', re.IGNORECASE),
    re.compile(r'^Writing - (?:original draft|review)', re.IGNORECASE),
    re.compile(r'^Resources?:\s+', re.IGNORECASE),
    re.compile(r'^Conceptualization:', re.IGNORECASE),
    re.compile(r'^Methodology:', re.IGNORECASE),
    re.compile(r'^Visualization:', re.IGNORECASE),
    re.compile(r'^Supervision:', re.IGNORECASE),
    re.compile(r'^Investigation:', re.IGNORECASE),
    re.compile(r'^Project administration:', re.IGNORECASE),
    # Copernicus plain-text sections (period after name, no # or **)
    re.compile(r'^Author contributions\.', re.IGNORECASE),
    re.compile(r'^Competing interests\.', re.IGNORECASE),
    re.compile(r'^Financial support\.', re.IGNORECASE),
    re.compile(r'^Review statement\.', re.IGNORECASE),
    re.compile(r'^Disclaimer\.', re.IGNORECASE),
    re.compile(r'^Code and data availability\.', re.IGNORECASE),
    re.compile(r'^Code availability\.', re.IGNORECASE),
    re.compile(r'^Data availability\.', re.IGNORECASE),
    re.compile(r'^Supplement\.', re.IGNORECASE),
    re.compile(r'^Acknowledgements?\.', re.IGNORECASE),
    re.compile(r'^Acknowledgments?\.', re.IGNORECASE),
    re.compile(r'^The supplement related to this article', re.IGNORECASE),
    # Copernicus editorial notes (no # header)
    re.compile(r'^Edited by:', re.IGNORECASE),
    re.compile(r'^Reviewed by:', re.IGNORECASE),
    re.compile(r'^This paper was edited by', re.IGNORECASE),
    re.compile(r'^Special issue statement\.', re.IGNORECASE),
    # Elsevier front-matter
    re.compile(r'^ARTICLEINFO', re.IGNORECASE),
    re.compile(r'^Handling Editor', re.IGNORECASE),
    re.compile(r'^Supplemental information related to', re.IGNORECASE),
    re.compile(r'^Corresponding author', re.IGNORECASE),
]


def is_headerless_boilerplate(line: str) -> bool:
    """Check if a line starts a headerless boilerplate block."""
    s = line.strip()
    for pat in HEADERLESS_BOILERPLATE_RE:
        if pat.match(s):
            return True
    return False


def strip_sections(lines: list[str]) -> list[str]:
    """Remove entire sections that match STRIP_SECTIONS by header,
    AND headerless boilerplate blocks (MDPI/IOP/Springer bold-text sections).
    
    IMPORTANT: stops skipping if we encounter real scientific content
    (images, HTML tables, Figure/Table captions) even within a stripped section.
    This prevents killing figures/tables that sit after References.
    """
    header_re = re.compile(r'^(#{1,6})\s+(.+)$')
    result = []
    skipping = False
    skip_level = 0
    skip_headerless = False
    skip_headerless_blanks = 0  # count consecutive blanks during headerless skip
    
    for line in lines:
        s = line.strip()
        m = header_re.match(s)
        
        if m:
            level = len(m.group(1))
            header_text = m.group(2).strip()
            
            # Any header stops headerless skipping
            skip_headerless = False
            skip_headerless_blanks = 0
            
            if is_strip_section(header_text):
                skipping = True
                skip_level = level
                continue
            
            if skipping and level <= skip_level:
                skipping = False
            
            if not skipping:
                result.append(line)
        elif skipping:
            # Check if we hit real content (images, tables, figures)
            # that should NOT be skipped even inside a stripped section
            if IMAGE_RE.match(s):
                skipping = False
                result.append(line)
            elif s.startswith('<table'):
                skipping = False
                result.append(line)
            elif FIGURE_CAPTION_RE.match(s):
                skipping = False
                result.append(line)
            elif s.startswith('Table ') and ('|' in s or '<' in s or len(s.split()) > 3):
                skipping = False
                result.append(line)
            else:
                continue
        elif skip_headerless:
            # Skip until next empty line or header (single-paragraph block)
            if not s:
                skip_headerless = False
            continue
        else:
            # Check for headerless boilerplate
            if s and is_headerless_boilerplate(line):
                skip_headerless = True
                continue
            result.append(line)
    
    return result


def clean_html_tables(lines: list[str]) -> list[str]:
    """Convert inline HTML tables to markdown format."""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if HTML_TABLE_START_RE.search(line):
            # Collect the full HTML table
            table_html = [line]
            depth = len(HTML_TABLE_START_RE.findall(line)) - len(HTML_TABLE_END_RE.findall(line))
            i += 1
            while i < len(lines) and depth > 0:
                table_html.append(lines[i])
                depth += len(HTML_TABLE_START_RE.findall(lines[i]))
                depth -= len(HTML_TABLE_END_RE.findall(lines[i]))
                i += 1
            
            # Convert to markdown
            html = '\n'.join(table_html)
            md_table = convert_html_table_to_markdown(html)
            result.append(md_table)
        else:
            result.append(line)
            i += 1
    
    return result


def extract_key_points(lines: list[str]) -> list[str]:
    """Extract Key Points from the preamble for metadata."""
    header_re = re.compile(r'^(#{1,6})\s+(.+)$')
    key_points = []
    in_key_points = False
    
    for line in lines:
        s = line.strip()
        m = header_re.match(s)
        
        if m:
            ht = m.group(2).strip().lower().rstrip(':')
            if ht in ('key points', 'highlights'):
                in_key_points = True
                continue
            elif in_key_points:
                break
        
        if in_key_points and s:
            # Remove bullet markers
            point = re.sub(r'^[-•*]\s*', '', s).strip()
            if point:
                key_points.append(point)
    
    return key_points


def extract_title_and_authors(lines: list[str], title_line: int) -> tuple[str, list[str]]:
    """Extract the paper title and author names."""
    header_re = re.compile(r'^(#{1,6})\s+(.+)$')
    
    # Title
    title = ""
    m = header_re.match(lines[title_line].strip())
    if m:
        title = m.group(2).strip()
    
    # Authors: lines between title and abstract/affiliations
    authors = []
    i = title_line + 1
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        # Stop at headers
        if header_re.match(s):
            break
        # Stop at long paragraphs (abstract without header)
        if len(s.split()) > 20 and not DECORATIVE_LATEX_RE.search(s):
            break
        # Stop at affiliations
        if AFFILIATION_RE.search(s) and not any(c.isupper() and len(c) > 1 for c in s.split(',')):
            break
        # If the line has names (commas, "and") and isn't too long
        if len(s) < 500 and (',' in s or ' and ' in s):
            # Strip superscripts and clean
            clean = DECORATIVE_LATEX_RE.sub('', s).strip()
            clean = re.sub(r'\s+', ' ', clean)
            if clean:
                authors.append(clean)
        i += 1
    
    return title, authors


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_paper(md_text: str, doi: str = "") -> tuple[str, dict]:
    """Full preprocessing pipeline for a single MinerU VLM markdown file.
    
    Returns:
        - Cleaned markdown text
        - Metadata dict {title, authors, key_points, figures}
    """
    lines = md_text.split('\n')
    
    # Step 0: Extract metadata before stripping
    key_points = extract_key_points(lines)
    
    # Step 1: Find where preamble ends (locate real title)
    title_line = detect_preamble_end(lines)
    
    # Step 2: Extract title and authors
    title, authors = extract_title_and_authors(lines, title_line)
    
    # Step 3: Find where author block ends
    content_start = detect_author_block_end(lines, title_line)
    
    # Step 4: Keep only content from title onward
    # (keep title as it becomes the first header in output)
    content_lines = [lines[title_line]] + lines[content_start:]
    
    # Step 5: Strip sections to remove (references, acknowledgments, etc.)
    content_lines = strip_sections(content_lines)
    
    # Step 6: Strip tail noise (CC license, publisher notes)
    content_lines = strip_tail_noise(content_lines)
    
    # Step 7: Link images to captions
    content_lines, figures = link_images_to_captions(content_lines)
    
    # Step 8: Clean HTML tables
    content_lines = clean_html_tables(content_lines)
    
    # Step 9: Normalise LaTeX
    content_lines = [normalise_latex(line) for line in content_lines]
    
    # Step 10: Fix OCR 'ris' stripping
    content_lines = [fix_ocr_ris_stripping(line) for line in content_lines]
    
    # Step 11: Clean up double/triple spaces left by stripped citations
    content_lines = [re.sub(r'  +', ' ', line) for line in content_lines]
    
    # Step 11.5: Fix hanging punctuation from citation stripping
    # "measured ." → "measured.", "ENSO) , a" → "ENSO), a"
    for idx, line in enumerate(content_lines):
        content_lines[idx] = re.sub(r'\s+([.,;:!?)])', r'\1', line)
    
    # Step 12: Strip orphaned [IMAGE: ...] tags for publisher logos at top
    # (first 5 lines only — don't touch genuine figure images)
    for idx in range(min(5, len(content_lines))):
        s = content_lines[idx].strip()
        if s.startswith('[IMAGE:'):
            content_lines[idx] = ''
    
    # Step 12.5: Strip orphaned [IMAGE: ...] at the very end (CC logos)
    for idx in range(len(content_lines) - 1, max(len(content_lines) - 5, -1), -1):
        s = content_lines[idx].strip()
        if s.startswith('[IMAGE:'):
            content_lines[idx] = ''
        elif s:
            break
    
    # Step 12.7: Deduplicate consecutive identical titles
    # MinerU sometimes outputs the same title twice
    deduped = []
    for line in content_lines:
        if deduped and line.strip() and line.strip().startswith('#'):
            if line.strip() == deduped[-1].strip():
                continue  # skip duplicate
        deduped.append(line)
    content_lines = deduped
    
    # Step 13: Clean up excessive blank lines
    cleaned = []
    blank_count = 0
    for line in content_lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                cleaned.append('')
        else:
            blank_count = 0
            cleaned.append(line)
    
    # Build metadata
    metadata = {
        "doi": doi,
        "title": title,
        "authors": authors,
        "key_points": key_points,
        "figures": figures,
        "lines_original": len(lines),
        "lines_cleaned": len(cleaned),
        "reduction_pct": round((1 - len(cleaned) / max(len(lines), 1)) * 100, 1),
    }
    
    return '\n'.join(cleaned), metadata


def find_markdown_files(input_dir: Path) -> list[tuple[str, Path]]:
    """Find all MinerU VLM markdown files in the input directory."""
    results = []
    for doi_dir in sorted(input_dir.iterdir()):
        if not doi_dir.is_dir():
            continue
        doi = doi_dir.name
        vlm_dir = doi_dir / "vlm"
        md_file = vlm_dir / f"{doi}.md"
        if md_file.exists():
            results.append((doi, md_file))
    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess MinerU VLM markdown for RAG")
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help="Input directory with parsed_levante structure")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Output directory for cleaned markdown")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only N papers (0 = all)")
    parser.add_argument("--paper", type=str, default="",
                        help="Process a single paper by DOI folder name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()
    
    # Find all markdown files
    papers = find_markdown_files(args.input)
    print(f"Found {len(papers)} markdown files in {args.input}")
    
    if args.paper:
        papers = [(doi, p) for doi, p in papers if doi == args.paper]
        if not papers:
            print(f"ERROR: Paper '{args.paper}' not found")
            sys.exit(1)
    
    if args.limit > 0:
        papers = papers[:args.limit]
    
    print(f"Processing {len(papers)} papers...")
    
    # Create output directory
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
    
    # Process
    total_orig = 0
    total_clean = 0
    errors = 0
    
    for i, (doi, md_path) in enumerate(papers, 1):
        try:
            md_text = md_path.read_text(encoding='utf-8')
            cleaned, metadata = preprocess_paper(md_text, doi)
            
            total_orig += metadata['lines_original']
            total_clean += metadata['lines_cleaned']
            
            if not args.dry_run:
                # Write cleaned markdown
                out_md = args.output / f"{doi}.md"
                out_md.write_text(cleaned, encoding='utf-8')
                
                # Write metadata JSON
                out_meta = args.output / f"{doi}.meta.json"
                out_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                    encoding='utf-8')
            
            if i % 500 == 0 or i == len(papers):
                pct = metadata['reduction_pct']
                print(f"  [{i}/{len(papers)}] {doi}: "
                      f"{metadata['lines_original']} → {metadata['lines_cleaned']} lines "
                      f"(-{pct}%), {len(metadata['figures'])} figures")
        
        except Exception as e:
            errors += 1
            print(f"  ERROR [{doi}]: {e}")
    
    # Summary
    reduction = round((1 - total_clean / max(total_orig, 1)) * 100, 1)
    print(f"\n{'='*60}")
    print(f"DONE: {len(papers)} papers processed, {errors} errors")
    print(f"Lines: {total_orig:,} → {total_clean:,} (-{reduction}%)")
    if not args.dry_run:
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
