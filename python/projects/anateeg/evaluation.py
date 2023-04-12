from projects.anateeg import utils
from projects.anateeg.config import Config

from projects.mnieeg import evaluation

if __name__ == "__main__":
    evaluation.forward_collect(Config, utils.GroupIO())
    evaluation.forward_collect_distance_matrix(Config, utils.GroupIO())
    evaluation.forward_plot(Config)
