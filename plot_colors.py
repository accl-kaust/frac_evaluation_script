"""Seaborn palette for the paper figures.

Figures 1, 3, 5, 7, 8, 9 and 13 to 16 share seaborn's "colorblind" green (ours) and
purple (baselines). Lines use strong tones that are at least 3:1 against white; bars
and boxes use lighter tints, with sand as the third bar color. Each figure's colors
also step apart in greyscale.
"""
import seaborn as sns


_CB = sns.color_palette("colorblind")  # 1 orange, 2 green, 4 purple


def _tone(color, *, white=0.0, black=0.0, saturation=1.0):
    """Desaturate, blend toward white, then toward black (all controls: 0 to 1)."""
    color = sns.desaturate(color, saturation)
    return tuple((channel * (1 - white) + white) * (1 - black) for channel in color)


# Lines and markers (Figures 3, 13 and 16).
# Ours: fRAC / Direct / Transport+fRAC. Baseline: CPU / Traditional / Transport.
# Third: DPU.
LINE_OURS = _CB[2]                          # green     #029e73
LINE_BASELINE = _tone(_CB[4], black=0.58)   # deep plum #56324f
# DPU is a second baseline: the lighter mauve keeps it purple but clearly apart from
# the deep plum CPU line, in color and in greyscale.
LINE_THIRD = _tone(_CB[4], black=0.15, saturation=0.8)  # mauve #a66d9b

# Bars and boxes (Figures 1, 5, 7, 8, 9, 14 and 15).
BAR_OURS = _tone(_CB[2], white=0.5, saturation=0.8)       # mint  #88c7b6
BAR_BASELINE = _tone(_CB[4], black=0.15, saturation=0.8)  # mauve #a66d9b
BAR_THIRD = _tone(_CB[1], white=0.75, saturation=0.8)     # sand  #f1e2c6


def bar_palette():
    """Return the baseline (mauve) and improved-design (mint) bar fills."""
    return [BAR_BASELINE, BAR_OURS]
