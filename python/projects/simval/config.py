from pathlib import Path


class PathConfig:
    PROJECT = Path("/mrhome/jesperdn/INN_JESPER/projects/simval")

    SPHERE = PROJECT / "sphere"
    HEAD = PROJECT / "head_3l"
    HEAD = PROJECT / "head_5l"

    DATA = PROJECT / "data"
    # ANALYSIS = PROJECT / "analysis"
    RESULTS = PROJECT / "results"
    SIMNIBS = PROJECT / "simnibs"
    SIMNIBS_TEMPLATE = PROJECT / "simnibs_template"
    RESOURCES = PROJECT / "resources"


class SphereConfig:
    names = ["brain", "skull", "scalp"]
    radii = [80, 86, 92]
    node_densities = [0.5] # [0.065, 0.125, 0.25, 0.5, 1]

    n_sensors = 100

    # Sources
    n_rays = 1000
    ray_start = 2
    ray_stop = 79 # 77

    # radii = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, 90, 92.5, 95]
    # density = 0.05

    conductivity = dict(brain=0.3, skull=0.006, scalp=0.3, brain_scalp=0.3)


class Config:
    path = PathConfig
    sphere = SphereConfig
