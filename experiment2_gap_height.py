import numpy as np
import time
import pickle

from src.rrt.informed_rrt_star import InformedRRTStar  # noqa: E402
from src.rrt.rrt_star import RRTStar  # noqa: E402
from src.search_space.search_space import SearchSpace  # noqa: E402
# from src.utilities.plotting import Plot  # noqa: E402


gap_height_rates = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]
parallel_count = 16

informed_rrt_star_results = dict((el, []) for el in gap_height_rates)
rrt_star_results = dict((el, []) for el in gap_height_rates)

print("Experiment: Comparing performance with gap height rate:")
for gap_height_rate in gap_height_rates:
    # ----- Start - Environment Creation -----
    L = 100
    T = 10
    dgoal = 80
    h = 80
    yg = 30
    hg_h = gap_height_rate*0.01
    hg = h*hg_h

    X_dimensions = np.array([(0, L), (0, L)])  # dimensions of Search Space

    Obstacles = np.array([
        (L/2 - T/2, (L-h)/2,
         L/2 + T/2, (L-h)/2 + yg),
        (L/2 - T/2, (L-h)/2 + yg + hg,
         L/2 + T/2, (L-h)/2 + h)])
    x_init = ((L-dgoal)/2, L/2)  # starting location
    x_goal = ((L+dgoal)/2, L/2)  # goal location

    # Calculation of optimal cost
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

    Q = np.array([2, 2])  # length of tree edges
    r = 1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1024*2  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.0  # probability of checking for a connection to goal
    ptc_opt_cost = 0.1
    optimal_path_cost = optimal_path_cost
    # create Search Space
    X = SearchSpace(X_dimensions, Obstacles, seed=0)

    print("Gap Height Rate: ", gap_height_rate)
    # create Search Space
    X = SearchSpace(X_dimensions, Obstacles, seed=None)

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

        informed_rrt_star_results[gap_height_rate].append(informed_run_time)
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

        rrt_star_results[gap_height_rate].append(rrt_run_time)

print("Experiments have been completed. Results: ")
print("informed_rrt_star_results: ", informed_rrt_star_results)
print("rrt_star_results:", rrt_star_results)

with open('results/experiment2_gap_height_informed_results.pkl', 'wb') as f:
    pickle.dump(informed_rrt_star_results, f)
with open('results/experiment2_gap_height_rrt_results.pkl', 'wb') as f:
    pickle.dump(rrt_star_results, f)

# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
