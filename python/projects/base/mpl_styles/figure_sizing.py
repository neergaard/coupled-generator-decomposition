# Figure width in number of columns
mm_per_inch = 25.4
inch_per_mm = 1 / mm_per_inch


# Sizes taken from
# https://www.elsevier.com/authors/policies-and-guidelines/artwork-and-media-instructions/artwork-sizing
SINGLE_COLUMN = 90
ONEHALF_COLUMN = 140
DOUBLE_COLUMN = 190
fig_width = dict(
    mm=dict(single=SINGLE_COLUMN, onehalf=ONEHALF_COLUMN, double=DOUBLE_COLUMN)
)
fig_width["inch"] = {k: v * inch_per_mm for k, v in fig_width["mm"].items()}

# fontsize = dict(small=7, normal=8, large=9)


def get_pixel(size, dpi=300):
    return size * inch_per_mm * dpi


def get_figsize(width, height_fraction=None, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    w = fig_width["mm"][width] if isinstance(width, str) else width

    # # Width of figure (in pts)
    # fig_width_pt = width_pt * fraction
    # # Convert from pt to inches
    # inches_per_pt = 1 / 72.27
    # # inches_per_mm = 1 / 2.5

    if height_fraction:
        h = w * height_fraction
    else:
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**.5 - 1) / 2
        h = w * golden_ratio
    h *= subplots[0] / subplots[1]

    # To inches
    w *= inch_per_mm
    h *= inch_per_mm

    return w, h