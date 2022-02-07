# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
import sys
sys.path.append('../..')

from src.rrt.informed_rrt_star import InformedRRTStar  # noqa: E402
from src.search_space.search_space import SearchSpace  # noqa: E402
from src.utilities.plotting import Plot  # noqa: E402
from src.utilities.obstacle_generation \
    import generate_random_obstacles  # noqa: E402

# Problem 1 - Randomly generated N obstacles

L = 100
N = 50

X_dimensions = np.array([(0, L), (0, L)])  # dimensions of Search Space

x_init = (0, 0)  # starting location
x_goal = (100, 100)  # goal location

Q = np.array([(4, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024*1024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.0  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions)

Obstacles = generate_random_obstacles(X, x_init, x_goal, N)
# create rrt_search
rrt = InformedRRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()

# plot
plot = Plot("Informed_rrt_star_2d", dimension_sizes=(0, L, 0, L))
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
