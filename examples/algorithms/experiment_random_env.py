import numpy as np
import time
import pickle
import sys
sys.path.append('../..')

from src.rrt.informed_rrt_star import InformedRRTStar  # noqa: E402
from src.rrt.rrt_star import RRTStar  # noqa: E402
from src.search_space.search_space import SearchSpace  # noqa: E402
# from src.utilities.plotting import Plot  # noqa: E402
from src.utilities.obstacle_generation \
    import generate_random_obstacles  # noqa: E402


print("Experiment: Random Environment Performance")

# Problem 1 - Randomly generated N obstacles

L = 100
N = 30

X_dimensions = np.array([(0, L), (0, L)])  # dimensions of Search Space

x_init = (0, L/2)  # starting location
x_goal = (L, L/2)  # goal location

Q = np.array([4, 4])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024*2  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.0  # probability of checking for a connection to goal
ptc_opt_cost = 0.1
optimal_path_cost = None

# create Search Space
X = SearchSpace(X_dimensions, seed=0)

Obstacles = generate_random_obstacles(X, x_init, x_goal, N)

# Parallel executing
parallel_count = 1
informed_rrt_star_results = []
rrt_star_results = []

for run_id in range(parallel_count):
    informed_rrt = InformedRRTStar(
        X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    print("Starting Informed RRT Star Search, id:", run_id, "")
    informed_start_time = time.time()
    informed_path = informed_rrt.rrt_star(
        optimal_cost=optimal_path_cost,
        percentage_of_optimal_cost=ptc_opt_cost)
    informed_run_time = time.time() - informed_start_time
    print("Informed RRT*, time: %s seconds" % informed_run_time)
    print("Informed RRT*, timestep: ", informed_rrt.samples_taken)
    informed_rrt_star_path_cost = 0.0
    for i in range(len(informed_path)-1):
        node1 = np.array(informed_path[i])
        node2 = np.array(informed_path[i+1])
        informed_rrt_star_path_cost += np.linalg.norm(node2-node1)
    print("Informed RRT* path cost: ", informed_rrt_star_path_cost)
    informed_rrt_star_results.append(informed_rrt.cost_versus_time_list)
    #

    print("Starting RRT Star Search")
    rrt = RRTStar(
        X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
    rrt_start_time = time.time()
    rrt_path = rrt.rrt_star(
        optimal_cost=optimal_path_cost,
        percentage_of_optimal_cost=ptc_opt_cost)
    rrt_run_time = time.time() - rrt_start_time
    print("RRT*, time: %s seconds" % rrt_run_time)
    print("RRT*, timestep: ", rrt.samples_taken)
    rrt_star_path_cost = 0.0
    for i in range(len(rrt_path)-1):
        node1 = np.array(rrt_path[i])
        node2 = np.array(rrt_path[i+1])
        rrt_star_path_cost += np.linalg.norm(node2-node1)
    print("RRT* path cost: ", rrt_star_path_cost)

    rrt_star_results.append(rrt.cost_versus_time_list)

print("Experiments have been completed. Results: ")
print("informed_rrt_star_results: ", informed_rrt_star_results)
print("rrt_star_results:", rrt_star_results)

with open('results/experiment4_random_cost_informed_results.pkl', 'wb') as f:
    pickle.dump(informed_rrt_star_results, f)
with open('results/experiment4_random_cost_rrt_results.pkl', 'wb') as f:
    pickle.dump(rrt_star_results, f)

# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
