import numpy as np
import time

from src.rrt.informed_rrt_star import InformedRRTStar  # noqa: E402
from src.rrt.rrt_star import RRTStar  # noqa: E402
from src.search_space.search_space import SearchSpace  # noqa: E402
from src.utilities.plotting import Plot  # noqa: E402

# Problem 2 - Single Obstacle
# ----- Start - Environment Creation -----
L = 100
w = 10
dgoal = 80

X_dimensions = np.array([(0, L), (0, L)])  # dimensions of Search Space

Obstacles = np.array([
    (L//2 - w//2, L//2 - w//2, L//2 + w//2, L//2 + w//2)])
x_init = ((L-dgoal)//2, L//2)  # starting location
x_goal = ((L+dgoal)//2, L//2)  # goal location

Q = np.array([2, 2])  # length of tree edges

r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024*2  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.0  # probability of checking for a connection to goal
ptc_opt_cost = 0.03
# optimal_path_cost = 2 * np.sqrt(((dgoal-w)/2)**2 + (w/2)**2) + w
optimal_path_cost = None

# create Search Space
X = SearchSpace(X_dimensions, Obstacles, seed=0)


# ----- End - Environment Creation -----

# ----- Informed RRT Star Search -----
start_time = time.time()
rrt = InformedRRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
print("Starting Informed RRT Star Search")
path = rrt.rrt_star(
    optimal_cost=optimal_path_cost, percentage_of_optimal_cost=ptc_opt_cost)
print("--- %s seconds ---" % (time.time() - start_time))
print("Informed RRT*, timestep: ", rrt.samples_taken)
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
start_time = time.time()
path = rrt.rrt_star(
    optimal_cost=optimal_path_cost, percentage_of_optimal_cost=ptc_opt_cost)
print("--- %s seconds ---" % (time.time() - start_time))
print("RRT*, timestep: ", rrt.samples_taken)
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


print("Optimum path cost: ", optimal_path_cost)
print("Informed RRT Star Path Cost: ", informed_rrt_star_path_cost)
print("RRT Star Path Cost: ", rrt_star_path_cost)
