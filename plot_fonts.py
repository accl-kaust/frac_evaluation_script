"""Shared figure style: Helvetica Neue Medium everywhere, one palette, and sizes that
come out identical *on the printed page* for every figure.

Fonts
-----
If fonts/ holds the Helvetica Neue Regular / Medium / Bold .ttf files (generated
on a Mac by extract_fonts.py), importing this module registers them with
matplotlib and makes Helvetica Neue at medium weight the default. Without them
it does nothing and matplotlib's default font is used.

Why the fonts/ directory: matplotlib (3.5) reads only the first face of a
TrueType collection, and macOS ships Helvetica Neue as one .ttc, so the Medium
face is invisible to it. extract_fonts.py pulls the three faces out of the
system .ttc into standalone .ttf files that matplotlib can see. The fonts are
Apple-licensed and therefore not tracked in git.

Print-size style
----------------
Each figure is scaled by LaTeX (\\includegraphics[width=...]) by a different factor,
so "fontsize=16" prints at 5 pt in one figure and 7 pt in another. PRINT below lists
the sizes we want *on the page* (in PostScript points, as measured in the ATC paper
PDF), and ``paper_style()`` converts them into the matplotlib sizes a given figure
needs from (a) the width it is printed at and (b) the width of its tight-cropped PDF:

    style = plot_fonts.paper_style(printed_width_pt=161.3, cropped_width_pt=568.5)
    plt.plot(x, y, linewidth=style.line, markersize=style.marker)
    plt.plot(x, cdf, linewidth=style.curve)        # marker-less curve (CDF)
    ax.bar(..., linewidth=style.edge, error_kw=style.error_kw)

``paper_style()`` also loads the sizes into rcParams, so text, ticks, spines, grids
and legends pick them up without explicit arguments. After saving, call
``style.report(pdf_path)`` to print the cropped size and the scale it implies; if the
cropped width drifted from ``cropped_width_pt`` by more than a few percent, update the
constant in the script.

LaTeX widths in the paper: \\columnwidth = 240 pt, \\textwidth = 504 pt.
"""
import glob
import logging
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

_font_files = sorted(glob.glob(os.path.join(FONT_DIR, "HelveticaNeue-*.ttf")))
for _path in _font_files:
    fm.fontManager.addfont(_path)

FONT_FAMILY = 'Helvetica Neue'
FONT_WEIGHT = 'medium'

if _font_files:
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.weight': FONT_WEIGHT,
        'axes.labelweight': FONT_WEIGHT,
        'axes.titleweight': FONT_WEIGHT,
        # Mathtext (e.g. the "x10^4" axis offset) in the same face instead of DejaVu.
        'mathtext.fontset': 'custom',
        'mathtext.rm': f'{FONT_FAMILY}:{FONT_WEIGHT}',
        'mathtext.it': f'{FONT_FAMILY}:{FONT_WEIGHT}',
        'mathtext.bf': f'{FONT_FAMILY}:{FONT_WEIGHT}',
        'mathtext.default': 'regular',
    })

# Embed the TrueType font (Type 42) instead of Type 3 glyph outlines in every PDF.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# When subsetting the embedded font, fontTools prints "<table> NOT subset; don't know
# how to subset; dropped" for Apple-specific tables in Helvetica Neue. Harmless; hide it.
logging.getLogger('fontTools').setLevel(logging.ERROR)

# --------------------------------------------------------------------------- #
# Palette (colour-blind safe, Okabe-Ito based). Every colour has ONE meaning across
# all figures.
#
# System colours - which platform / design produced the result:
#   C_OURS  - fRAC: direct offload, reassembly buffer, "with" (ours)
#   C_BASE  - baseline: CPU, traditional offload, per-accelerator buffer, "without"
#   C_THIRD - DPU
#
# Category colours - accelerated functions / request types (figs 9, 14, 18).
# Component colours - the latency components of one invocation (fig 1).
# All ten colours (Tableau 10, Okabe-Ito, Paul Tol, ColorBrewer Dark2 picks) are at
# least 35 dE apart in CIELAB, so dark blue never means anything but fRAC, vermilion
# never anything but the baseline, and so on.
# --------------------------------------------------------------------------- #
C_OURS = '#004D7A'    # dark blue
C_BASE = '#D55E00'    # vermilion
C_THIRD = '#D98CBF'   # pink
PALETTE = [C_OURS, C_BASE, C_THIRD]                        # systems: fRAC, baseline, DPU
CATEGORY_PALETTE = ['#44AA99', '#7B3294', '#E6AB02', '#8C564B']  # teal, purple, gold, brown
COMPONENT_PALETTE = ['#EFD999', '#FB9676', '#64903F']           # sand, salmon, green
# Fig 1 stacks the three components, so they also step in lightness (L* 87 / 72 / 55:
# network transfer, host-FPGA DMA, accelerator control) and stay apart in greyscale; all
# three are light enough for black value labels (contrast >= 5.6:1) and >= 36 dE from the
# seven colours above. Checked for protan/deutan/tritan separation (worst pair dE 10 in OKLab).

# Marker shapes by role (figs 3, 13, 16): ours = square, baseline = triangle, third = circle.
M_OURS, M_BASE, M_THIRD = 's', '^', 'o'

# Target sizes on the printed page, in points.
PRINT = {
    'font': 6.0,     # every piece of text: ticks, labels, titles, legends, annotations
    'stroke': 0.6,   # every line: data lines, bar/box edges, whiskers, error bars, medians,
                     # dashed separators, spines, tick marks, grid lines, legend frames
    'curve': 1.0,    # data curves drawn without markers (the CDFs in figs 17 and 18). At
                     # 0.6 pt a bare curve reads lighter than a 0.6 pt line carrying 3 pt
                     # markers (figs 13, 16) or a bar with a 0.6 pt edge, so these get a
                     # heavier stroke; axes, ticks and legend frames stay at 'stroke'.
    'marker': 3.0,   # marker size
    'cap': 1.4,      # error-bar cap size
    'tick': 1.2,     # major tick length and tick-label padding
    'titlepad': 2.0,
    'labelpad': 1.5,
}

# Figs 13, 14 and 15 sit side by side at 0.32\textwidth, so their axes boxes print at
# one height (page points); PaperStyle.fix_axes_box enforces it. The box width follows
# from cropped_width_pt minus the tick and axis labels, so equal labels give equal widths.
AXES_HEIGHT_13_15 = 70.0


class PaperStyle:
    """Sizes (in matplotlib points) that print at the PRINT sizes for one figure."""

    def __init__(self, printed_width_pt, cropped_width_pt):
        self.printed_width_pt = float(printed_width_pt)
        self.cropped_width_pt = float(cropped_width_pt)
        self.scale = self.printed_width_pt / self.cropped_width_pt   # page pt per figure pt
        k = 1.0 / self.scale
        self.font = PRINT['font'] * k
        # One stroke width for everything; the role names are kept so scripts read clearly.
        self.line = self.median = self.edge = self.axes = PRINT['stroke'] * k
        self.curve = PRINT['curve'] * k
        self.marker = PRINT['marker'] * k
        self.cap = PRINT['cap'] * k
        self.tick = PRINT['tick'] * k
        self.titlepad = PRINT['titlepad'] * k
        self.labelpad = PRINT['labelpad'] * k
        # kwargs for error bars drawn by ax.bar(..., yerr=...)
        self.error_kw = dict(elinewidth=self.edge, capsize=self.cap, capthick=self.edge, ecolor='black')

    def fix_axes_box(self, fig, axes, height_pt, pad_in=0.1):
        """Resize ``fig`` and reposition ``axes`` (one axes or a vertical stack of them)
        so the axes box prints exactly ``height_pt`` page points high, and so the tight
        crop is exactly ``cropped_width_pt`` wide (the box takes the width left over by
        the tick and axis labels, so the declared scale, and every font and stroke size,
        holds without re-tuning the constant).

        Everything outside the axes keeps its size; the figure grows or shrinks around
        the box, and the crop margins are preserved. Call it after the layout
        (tight_layout / subplots_adjust) and before anything placed in figure
        coordinates, then save with bbox_inches='tight' (``pad_in`` is its pad).
        Returns the printed axes-box width in page points."""
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        target_h = height_pt / self.scale / 72.0          # inches on the figure page
        crop_w = self.cropped_width_pt / 72.0
        for _ in range(2):   # second pass picks up decorations placed in axes fractions
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            inv = fig.dpi_scale_trans.inverted()
            tight = fig.get_tightbbox(r)
            boxes = [ax.get_window_extent(r).transformed(inv) for ax in axes]
            sx0, sx1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
            sy0, sy1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
            left, right = sx0 - tight.x0, tight.x1 - sx1
            bottom, top = sy0 - tight.y0, tight.y1 - sy1
            target_w = crop_w - 2 * pad_in - left - right
            new_w = left + target_w + right + 2 * pad_in
            new_h = bottom + target_h + top + 2 * pad_in
            fig.set_size_inches(new_w, new_h, forward=False)
            for ax, b in zip(axes, boxes):
                y0 = bottom + pad_in + (b.y0 - sy0) / (sy1 - sy0) * target_h
                h = b.height / (sy1 - sy0) * target_h
                ax.set_position([(left + pad_in) / new_w, y0 / new_h, target_w / new_w, h / new_h])
        return target_w * 72.0 * self.scale

    def pt(self, printed_pt):
        """Convert any size given in printed points to this figure's points."""
        return printed_pt / self.scale

    def apply(self):
        f = self.font
        plt.rcParams.update({
            'font.size': f, 'axes.titlesize': f, 'axes.labelsize': f,
            'xtick.labelsize': f, 'ytick.labelsize': f,
            'legend.fontsize': f, 'legend.title_fontsize': f, 'figure.titlesize': f,
            'lines.linewidth': self.line, 'lines.markersize': self.marker,
            'lines.markeredgewidth': self.axes,
            'patch.linewidth': self.axes,   # legend frames; bars pass style.edge explicitly
            'axes.linewidth': self.axes,
            'xtick.major.width': self.axes, 'ytick.major.width': self.axes,
            'xtick.minor.width': self.axes, 'ytick.minor.width': self.axes,
            'xtick.major.size': self.tick, 'ytick.major.size': self.tick,
            'xtick.minor.size': 0.6 * self.tick, 'ytick.minor.size': 0.6 * self.tick,
            'xtick.major.pad': self.tick, 'ytick.major.pad': self.tick,
            'grid.linewidth': self.axes, 'grid.linestyle': '--', 'grid.alpha': 0.7,
            # Dash patterns in units of the line width: denser than matplotlib's defaults
            # (3.7/1.6 and 1/1.65) so dashed and dotted lines carry the same visual weight
            # as solid ones.
            'lines.scale_dashes': True,
            'lines.dashed_pattern': [3.0, 1.2], 'lines.dotted_pattern': [1.0, 1.0],
            'lines.dashdot_pattern': [4.0, 1.2, 1.0, 1.2],
            'hatch.linewidth': self.axes,
            'errorbar.capsize': self.cap,
            'axes.titlepad': self.titlepad, 'axes.labelpad': self.labelpad,
            'boxplot.boxprops.linewidth': self.edge, 'boxplot.whiskerprops.linewidth': self.edge,
            'boxplot.capprops.linewidth': self.edge, 'boxplot.medianprops.linewidth': self.median,
        })
        return self

    def report(self, pdf_path):
        """Print the saved PDF's cropped size and what it implies on the page."""
        try:
            with open(pdf_path, 'rb') as fh:
                head = fh.read()
            m = re.search(rb'/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]', head)
            if not m:
                return
            w = float(m.group(3)) - float(m.group(1))
            h = float(m.group(4)) - float(m.group(2))
        except OSError:
            return
        scale = self.printed_width_pt / w
        drift = 100.0 * (w / self.cropped_width_pt - 1.0)
        print(f"{os.path.basename(pdf_path)}: cropped {w:.1f} x {h:.1f} pt -> printed "
              f"{self.printed_width_pt:.1f} x {h * scale:.1f} pt, text {PRINT['font'] * self.scale / scale:.2f} pt"
              + (f"  [cropped width drifted {drift:+.1f}% from cropped_width_pt; update the constant]"
                 if abs(drift) > 2.0 else ""))


def paper_style(printed_width_pt, cropped_width_pt):
    """Load the print-size style for a figure printed at ``printed_width_pt`` whose
    tight-cropped PDF is ``cropped_width_pt`` wide, and return the size table."""
    return PaperStyle(printed_width_pt, cropped_width_pt).apply()
