import numpy as np
import sys
sys.path.append('../..')


from src.rrt.rrt_star import RRTStar  # noqa: E402
from src.search_space.search_space import SearchSpace  # noqa: E402
from src.utilities.obstacle_generation \
    import generate_random_obstacles  # noqa: E402
from src.utilities.plotting import Plot  # noqa: E402

X_dimensions = np.array([
    (0, 100), (0, 100), (0, 100)])  # dimensions of Search Space
x_init = (0, 0, 0)  # starting location
x_goal = (100, 100, 100)  # goal location

Q = np.array([2, 2])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions)
n = 50
Obstacles = generate_random_obstacles(X, x_init, x_goal, n)
# create rrt_search
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()

# plot
plot = Plot("rrt_star_3d_with_random_obstacles")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)

plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
