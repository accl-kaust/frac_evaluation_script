"""Shared figure font setup.

If fonts/ holds the Helvetica Neue Regular / Medium / Bold .ttf files (generated
on a Mac by extract_fonts.py), importing this module registers them with
matplotlib and makes Helvetica Neue at medium weight the default. Without them
it does nothing and matplotlib's default font is used.

Why the fonts/ directory: matplotlib (3.5) reads only the first face of a
TrueType collection, and macOS ships Helvetica Neue as one .ttc, so the Medium
face is invisible to it. extract_fonts.py pulls the three faces out of the
system .ttc into standalone .ttf files that matplotlib can see. The fonts are
Apple-licensed and therefore not tracked in git.
"""
import glob
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

_font_files = sorted(glob.glob(os.path.join(FONT_DIR, "HelveticaNeue-*.ttf")))
for _path in _font_files:
    fm.fontManager.addfont(_path)

if _font_files:
    plt.rcParams.update({
        'font.family': 'Helvetica Neue',
        'font.weight': 'medium',
        'axes.labelweight': 'medium',
        'axes.titleweight': 'medium',
    })
