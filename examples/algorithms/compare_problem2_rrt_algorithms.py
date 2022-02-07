# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
import sys
sys.path.append('../..')

from src.rrt.informed_rrt_star import InformedRRTStar  # noqa: E402
from src.search_space.search_space import SearchSpace  # noqa: E402
from src.utilities.plotting import Plot  # noqa: E402

# Problem 2

L = 100
T = 10
dgoal = 80
h = 50
yg = 30
hg_h = 0.05
hg = h*hg_h

X_dimensions = np.array([(0, L), (0, L)])  # dimensions of Search Space

Obstacles = np.array([
    (L/2 - T/2, (L-h)/2,
     L/2 + T/2, (L-h)/2 + yg),
    (L/2 - T/2, (L-h)/2 + yg + hg,
     L/2 + T/2, (L-h)/2 + h)])
x_init = ((L-dgoal)/2, L/2)  # starting location
x_goal = ((L+dgoal)/2, L/2)  # goal location

Q = np.array([(8, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024*256  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

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
