import colorsys
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np

tissue_to_simnibs_index = dict(
    wm=0,
    gm=1,
    csf=2,
    bone=3,
    skin=4,
    eyes=5,
    bone_compact=6,
    bone_spongy=7,
    blood=8,
    muscle=9,
)

# SimNIBS colormap
def get_simnibs_cmap(mask=None):
    simnibs_tissue_colors = np.array(
        [
            [230, 230, 230],  # White matter
            [129, 129, 129],  # Gray matter
            [104, 163, 255],  # CSF
            [255, 239, 179],  # Bone
            [255, 166, 133],  # Skin
            [255, 240, 0],  # Eyes
            [255, 239, 179],  # Bone (compact)
            [255, 138, 57],  # Bone (spongy)
            [0, 65, 142],  # Blood
            [0, 118, 14],  # Muscle
        ],
        dtype=float,
    )
    simnibs_tissue_colors /= 255
    if mask is not None:
        simnibs_tissue_colors = simnibs_tissue_colors[mask]
    return ListedColormap(simnibs_tissue_colors)


def get_random_cmap(
    nlabels,
    color_type="bright",
    first_color_black=False,
    last_color_black=False,
    seed=None,
):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :return: colormap for matplotlib
    """

    rng = np.random.default_rng(seed=seed)

    if color_type == "bright":
        randHSVcolors = [
            (
                rng.uniform(low=0.0, high=1),
                rng.uniform(low=0.2, high=1),
                rng.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]
    elif color_type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                rng.uniform(low=low, high=high),
                rng.uniform(low=low, high=high),
                rng.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]
    else:
        raise ValueError

    # Convert HSV list to RGB
    randRGBcolors = [
        colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
        for HSVcolor in randHSVcolors
    ]

    if first_color_black:
        randRGBcolors[0] = [0, 0, 0]
    if last_color_black:
        randRGBcolors[-1] = [0, 0, 0]

    return LinearSegmentedColormap.from_list(
        f"random_{color_type}", randRGBcolors, N=nlabels
    )
