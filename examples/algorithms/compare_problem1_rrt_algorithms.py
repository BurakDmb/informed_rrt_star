import numpy as np
import sys
sys.path.append('../..')

from src.rrt.informed_rrt_star import InformedRRTStar  # noqa: E402
from src.rrt.rrt_star import RRTStar  # noqa: E402
from src.search_space.search_space import SearchSpace  # noqa: E402
from src.utilities.plotting import Plot  # noqa: E402
from src.utilities.obstacle_generation \
    import generate_random_obstacles  # noqa: E402

# Problem 1 - Randomly generated N obstacles

L = 100
N = 30

X_dimensions = np.array([(0, L), (0, L)])  # dimensions of Search Space

x_init = (0, L/2)  # starting location
x_goal = (L, L/2)  # goal location

Q = np.array([4, 4])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024*4  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.0  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, seed=0)

Obstacles = generate_random_obstacles(X, x_init, x_goal, N)

# ----- Informed RRT Star Search -----
print("Starting Informed RRT Star Search")
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


informed_rrt_star_path_cost = 0.0
for i in range(len(path)-1):
    node1 = np.array(path[i])
    node2 = np.array(path[i+1])
    informed_rrt_star_path_cost += np.linalg.norm(node2-node1)
# print("Path: ", path)


# ----- RRT Star Search -----
# create rrt_search
print("Starting RRT Star Search")
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()

# plot
plot = Plot("rrt_star_2d", dimension_sizes=(0, L, 0, L))
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)


rrt_star_path_cost = 0.0
for i in range(len(path)-1):
    node1 = np.array(path[i])
    node2 = np.array(path[i+1])
    rrt_star_path_cost += np.linalg.norm(node2-node1)

print("Informed RRT Star Path Cost: ", informed_rrt_star_path_cost)
print("RRT Star Path Cost: ", rrt_star_path_cost)
