# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
import sys
sys.path.append('../..')

from src.rrt.informed_rrt_star import InformedRRTStar  # noqa: E402
from src.search_space.search_space import SearchSpace  # noqa: E402
from src.utilities.plotting import Plot  # noqa: E402

# Problem 3 - Passing through narrow gap.

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

Q = np.array([(4, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.0  # probability of checking for a connection to goal

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

x_init_arr = np.array(x_init)
down_arr = np.array([L/2 - T/2, (L-h)/2])
up_arr = np.array([L/2 - T/2, (L-h)/2 + h])

not_optimal_path_cost = T + 2*np.min([
    np.linalg.norm(down_arr-x_init_arr),
    np.linalg.norm(up_arr-x_init_arr)])

optimal_path_cost = T + 2*np.sqrt(
    ((dgoal-T)/2)**2 +
    (np.max([yg, h-yg-hg, h/2]) - h/2)**2
    )

path_cost = 0.0
for i in range(len(path)-1):
    node1 = np.array(path[i])
    node2 = np.array(path[i+1])
    path_cost += np.linalg.norm(node2-node1)

print("Path: ", path)
print("Not optimal path cost: ", not_optimal_path_cost)
print("Optimal path cost: ", optimal_path_cost)
print("Calculated path cost: ", path_cost)
