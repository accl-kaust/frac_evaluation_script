"""Seaborn palettes and tone controls for the paper figures.

Palette indices: 0 blue, 1 orange, 2 green, 3 red, 4 purple, 5 brown,
6 pink, 7 gray, 8 yellow, 9 cyan. Hex colors are also accepted.
"""
import seaborn as sns


# Approved Figure 5 fills; Figures 1, 7, 8, 9, 14 and 15 reuse them.
# Order: baseline purple, improved-design green. Each has its own brightness.
BAR_PALETTE = "colorblind"
BAR_INDICES = [4, 2]
BAR_SATURATION = 0.8
BAR_WHITE = [0.0, 0.5]
BAR_BLACK = [0.15, 0.0]

# Approved Figure 13 line palette, shared by Figures 3, 13 and 16.
# Order: fRAC / Direct, CPU / Traditional / Transport, DPU.
# Seaborn colorblind blue and vermilion; darkened palette gray for DPU.
LINE_PALETTE = "colorblind"
LINE_INDICES = [0, 3, 7]
LINE_SATURATION = [1.0, 1.0, 1.0]
LINE_WHITE = [0.0, 0.0, 0.0]
LINE_BLACK = [0.0, 0.0, 0.6]


def toned_palette(colors, *, palette="colorblind", saturation=1.0, white=0.0, black=0.0):
    """Desaturate, blend toward white, then toward black (all controls: 0 to 1)."""
    if not all(0 <= value <= 1 for value in (saturation, white, black)):
        raise ValueError("saturation, white and black must be between 0 and 1")
    base = sns.color_palette(palette, n_colors=10)
    selected = []
    for color in colors:
        if isinstance(color, int):
            if color not in range(10):
                raise ValueError("Palette indices must be between 0 and 9")
            color = base[color]
        selected.append(color)
    toned = sns.color_palette(selected, desat=saturation)
    return sns.color_palette([
        tuple((channel * (1 - white) + white) * (1 - black) for channel in color)
        for color in toned
    ])


def bar_palette():
    """Return Figure 5's darker purple and lighter green, with solid fills."""
    return [
        toned_palette([index], palette=BAR_PALETTE, saturation=BAR_SATURATION,
                      white=BAR_WHITE[n], black=BAR_BLACK[n])[0]
        for n, index in enumerate(BAR_INDICES)
    ]
