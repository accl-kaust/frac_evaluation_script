"""Extract Helvetica Neue Regular/Medium/Bold from the macOS system .ttc into fonts/.

Run once on a Mac: python3 extract_fonts.py
"""
import os

from fontTools.ttLib import TTCollection

SOURCE = "/System/Library/Fonts/HelveticaNeue.ttc"
WANT = {"Helvetica Neue": "Regular", "Helvetica Neue Medium": "Medium", "Helvetica Neue Bold": "Bold"}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

os.makedirs(OUT_DIR, exist_ok=True)
for font in TTCollection(SOURCE).fonts:
    full_name = font["name"].getDebugName(4)
    if full_name in WANT:
        out = os.path.join(OUT_DIR, f"HelveticaNeue-{WANT[full_name]}.ttf")
        font.save(out)
        print("wrote", out)
