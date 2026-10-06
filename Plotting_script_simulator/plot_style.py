"""Shared matplotlib style for the simulator figures: Helvetica Neue Medium everywhere,
plus the print-size rules and palette shared with the evaluation figures (see
../plot_fonts.py).

macOS ships Helvetica Neue as a TrueType *collection* (HelveticaNeue.ttc). matplotlib
only ever reads the first face of a .ttc (Regular), so the Medium face is extracted once
into fonts/HelveticaNeue-Medium.ttf and registered explicitly.
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager

# The print-size style, palette and marker roles live one directory up so that the
# simulator figures and the evaluation figures share a single definition.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_fonts import (  # noqa: E402,F401
    paper_style, PaperStyle, PRINT, PALETTE, CATEGORY_PALETTE, COMPONENT_PALETTE,
    C_OURS, C_BASE, C_THIRD,
    M_OURS, M_BASE, M_THIRD,
)

FONT_FAMILY = 'Helvetica Neue'
FONT_WEIGHT = 'medium'

_SYSTEM_TTC = '/System/Library/Fonts/HelveticaNeue.ttc'
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
_FONT_FILE = os.path.join(_FONT_DIR, 'HelveticaNeue-Medium.ttf')


def _extract_medium_face():
    from fontTools.ttLib import TTCollection

    if not os.path.exists(_SYSTEM_TTC):
        raise RuntimeError(f'{_SYSTEM_TTC} not found; copy HelveticaNeue-Medium.ttf into {_FONT_DIR} manually.')
    for face in TTCollection(_SYSTEM_TTC).fonts:
        names = face['name']
        if names.getDebugName(1) == FONT_FAMILY and names.getDebugName(2) == 'Medium':
            os.makedirs(_FONT_DIR, exist_ok=True)
            face.save(_FONT_FILE)
            return
    raise RuntimeError(f'Helvetica Neue Medium face not found in {_SYSTEM_TTC}')


def apply():
    """Register Helvetica Neue Medium and make it the default for every text element."""
    if not os.path.exists(_FONT_FILE):
        _extract_medium_face()
    font_manager.fontManager.addfont(_FONT_FILE)

    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.weight': FONT_WEIGHT,
        'axes.labelweight': FONT_WEIGHT,
        'axes.titleweight': FONT_WEIGHT,
        # Scientific-notation offset text (e.g. "x10^5") is rendered through mathtext;
        # route it to the same font instead of DejaVu.
        'mathtext.fontset': 'custom',
        'mathtext.rm': f'{FONT_FAMILY}:{FONT_WEIGHT}',
        'mathtext.it': f'{FONT_FAMILY}:{FONT_WEIGHT}',
        'mathtext.bf': f'{FONT_FAMILY}:{FONT_WEIGHT}',
        'mathtext.default': 'regular',
        # Embed the TrueType font (Type 42) rather than converting glyphs to Type 3 outlines.
        'pdf.fonttype': 42,
    })
