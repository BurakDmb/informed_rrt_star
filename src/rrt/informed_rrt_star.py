# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
from operator import itemgetter

from src.rrt.heuristics import cost_to_go
from src.rrt.heuristics import segment_cost, path_cost
from src.rrt.rrt import RRT
import numpy as np


class InformedRRTStar(RRT):
    def __init__(self, X, Q, x_init, x_goal, max_samples,
                 r, prc=0.01, rewire_count=None):
        """
        Informed RRT* Search
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge
        when checking for collisions
        :param prc: probability of checking whether there is a solution
        :param rewire_count: number of nearby vertices to rewire
        """
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc)
        self.rewire_count = rewire_count if rewire_count is not None else 0
        self.c_best = float('inf')  # length of best solution thus far

        self.RotationMatrix = self.RotationToWorldFrame(
            self.x_start, self.x_goal,
            np.linalg.norm(np.array(x_goal) - np.array(x_init))
            )

    def get_nearby_vertices(self, tree, x_init, x_new):
        """
        Get nearby vertices to new vertex and their associated path costs
        from the root of tree
        as if new vertex is connected to each one separately.

        :param tree: tree in which to search
        :param x_init: starting vertex used to calculate path cost
        :param x_new: vertex around which to find nearby vertices
        :return: list of nearby vertices and their costs, sorted
        in ascending order by cost
        """
        X_near = self.nearby(tree, x_new, self.current_rewire_count(tree))
        L_near = [
            (path_cost(self.trees[tree].E, x_init, x_near) +
             segment_cost(x_near, x_new), x_near) for x_near in X_near]
        # noinspection PyTypeChecker
        L_near.sort(key=itemgetter(0))

        return L_near

    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array([[(x_goal[0] - x_start[0]) / L],
                       [(x_goal[1] - x_start[1]) / L], [0.0]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag(
            [1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C

    def rewire(self, tree, x_new, L_near):
        """
        Rewire tree to shorten edges if possible
        Only rewires vertices according to rewire count
        :param tree: int, tree to rewire
        :param x_new: tuple, newly added vertex
        :param L_near: list of nearby vertices used to rewire
        :return:
        """
        for c_near, x_near in L_near:
            curr_cost = path_cost(self.trees[tree].E, self.x_init, x_near)
            tent_cost = path_cost(
                self.trees[tree].E, self.x_init, x_new
                ) + segment_cost(x_new, x_near)
            if tent_cost < curr_cost and self.X.collision_free(
                    x_near, x_new, self.r):
                self.trees[tree].E[x_near] = x_new

    def connect_shortest_valid(self, tree, x_new, L_near):
        """
        Connect to nearest vertex that has an unobstructed path
        :param tree: int, tree being added to
        :param x_new: tuple, vertex being added
        :param L_near: list of nearby vertices
        """
        # check nearby vertices for total cost and connect shortest valid edge
        for c_near, x_near in L_near:
            if c_near + cost_to_go(x_near, self.x_goal) < self.c_best\
                    and self.connect_to_point(tree, x_near, x_new):
                break

    def current_rewire_count(self, tree):
        """
        Return rewire count
        :param tree: tree being rewired
        :return: rewire count
        """
        # if no rewire count specified, set rewire count to be all vertices
        if self.rewire_count is None:
            return self.trees[tree].V_count

        # max valid rewire count
        return min(self.trees[tree].V_count, self.rewire_count)

    def InGoalRegion(self, x_new, q):
        dist = np.linalg.norm(np.array(self.x_goal) - np.array(x_new))
        if dist < q:
            return True
        return False

    def findCost(self, path):
        if len(path) == 0:
            return float("inf")

        path_cost = 0.0
        for i in range(len(path)-1):
            node1 = np.array(path[i])
            node2 = np.array(path[i+1])
            path_cost += np.linalg.norm(node2-node1)
        return path_cost

    def rrt_star(self):
        """
        Based on algorithm found in: Incremental Sampling-based Algorithms for
        Optimal Motion Planning
        http://roboticsproceedings.org/rss06/p34.pdf
        :return: set of Vertices; Edges in form: vertex:
        [neighbor_1, neighbor_2, ...]
        """
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        self.X_soln = []
        c_best = float("inf")

        while True:
            # TODO: Find the purpose of these loops.
            # Not exists in the original implementation

            # iterate over different edge lengths
            for q in self.Q:
                # iterate over number of edges of given length to add
                for i in range(q[1]):

                    # TODO: implement finding c_best, needs testing
                    if self.X_soln:
                        cost = {
                            path: self.findCost(path) for path in self.X_soln}
                        x_best = min(cost, key=cost.get)
                        c_best = cost[x_best]

                    x_new, x_nearest = self.informed_new_and_near(
                        0, q, c_best, self.RotationMatrix)

                    # If x_new is none, then it is not collision free,
                    # therefore continue'ing the execution.
                    if x_new is None:
                        continue

                    # get nearby vertices and cost-to-come
                    L_near = self.get_nearby_vertices(0, self.x_init, x_new)

                    # check nearby vertices for total cost and
                    # connect shortest valid edge
                    self.connect_shortest_valid(0, x_new, L_near)

                    if x_new in self.trees[0].E:
                        # rewire tree
                        self.rewire(0, x_new, L_near)

                    if self.InGoalRegion(x_new, q):
                        self.X_soln.append(self.get_path)

                    solution = self.check_solution()
                    if solution[0]:
                        return solution[1]
