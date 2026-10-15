"""Seaborn palette for the paper figures.

Figures 1, 3, 5, 7, 8, 9 and 13 to 16 share three hues from seaborn's circular HUSL
palette, sns.husl_palette(3, h=HUSL_H, s=HUSL_S): 0 tomato, 1 green, 2 blue. Each
color has its own lightness so the three stay distinct in greyscale; lines use the
darker tones, bars and boxes lighter tints of the same hues.
"""
import seaborn as sns


HUSL_H = 0.05
HUSL_S = 0.7


def _husl(hue, lightness):
    return sns.husl_palette(3, h=HUSL_H, s=HUSL_S, l=lightness)[hue]


# Lines and markers (Figures 3, 13 and 16); each is at least 3:1 against white.
# Ours: fRAC / Direct / Transport+fRAC. Baseline: CPU / Traditional / Transport.
# Third: DPU.
LINE_OURS = _husl(2, 0.45)      # blue       #4566b6
LINE_BASELINE = _husl(0, 0.60)  # tomato     #e66853
LINE_THIRD = _husl(1, 0.30)     # dark green #234f31

# Bars and boxes (Figures 1, 5, 7, 8, 9, 14 and 15).
BAR_OURS = _husl(2, 0.65)       # #829ae1
BAR_THIRD = _husl(1, 0.48)      # #3d7e52
# A light tint of a red-orange reads pink, so this one is nudged in hue (18 -> 22.5
# degrees) and saturation to keep the line tomato's perceived hue.
BAR_BASELINE = sns.husl_palette(1, h=22.5 / 359, s=1.0, l=0.80)[0]  # #ffb1a1


def bar_palette():
    """Return the baseline (tomato) and improved-design (blue) bar fills."""
    return [BAR_BASELINE, BAR_OURS]
