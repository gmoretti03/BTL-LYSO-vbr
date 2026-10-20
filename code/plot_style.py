"""Shared CMS plot style (mplhep), used by every analysis script in this repo."""

import mplhep as hep

CMS_LABEL_TEXT = "Private Work"

# Same 6-color, colorblind-safe palette hep.style.use("CMS") sets as the axes
# color cycle (mplhep.styles.cms.cmap_petroff) - named here so every script
# can assign the same semantic color to the same quantity (e.g. array A vs B)
# instead of picking arbitrary matplotlib color names.
PETROFF = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
COLOR_A = PETROFF[2]     # red    - array A / channel 1
COLOR_B = PETROFF[0]     # blue   - array B / channel 2
COLOR_DATA = PETROFF[1]  # orange - generic data histograms
COLOR_DIFF = "#029e73"   # teal (Okabe-Ito) - residual / difference histograms
COLOR_BAND = PETROFF[4]  # gray   - +-1 std shading

# Perceptually-uniform colormap for ordinal/categorical indices (e.g. per-pair
# coloring in a scatter plot) - avoids "rainbow", which is not colorblind-safe
# and not perceptually uniform.
CMAP_ORDINAL = "viridis"


def use_cms_style():
    """Apply the official CMS matplotlib style. Call once, before creating any figure."""
    hep.style.use("CMS")


def cms_label(ax, **kwargs):
    """Add the standard 'CMS Private Work' label above the axes.

    This is lab/test-beam data, not CMS collision data, so no lumi/com-energy
    right-hand label is shown.
    """
    return hep.cms.label(ax=ax, text=CMS_LABEL_TEXT, data=True, rlabel="", **kwargs)
