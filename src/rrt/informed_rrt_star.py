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
            self.x_init, self.x_goal,
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

    def RotationToWorldFrame(self, x_start, x_goal, L):
        dim = 2
        # Transverse axis of the ellipsoid in the world frame
        E1 = (np.array(x_goal) - np.array(x_start)) / L
        # first basis vector of the world frame [1,0,0,...]
        W1 = [1]+[0]*(dim - 1)
        # outer product of E1 and W1
        M = np.outer(E1, W1)
        # SVD decomposition od outer product
        U, S, V = np.linalg.svd(M)
        # Calculate the middle diagonal matrix
        middleM = np.eye(dim)
        middleM[-1, -1] = np.linalg.det(U)*np.linalg.det(V)
        # calculate the rotation matrix
        C = U@middleM@V.T
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
        if (dist < q).all():
            return True
        return False

    def findCost(self, X_soln):
        if len(X_soln) == 0:
            return float("inf")

        minimum_cost = float("inf")
        minimum_path = []
        for solution in X_soln:

            path_cost = 0.0
            for i in range(len(solution)-1):
                node1 = np.array(solution[i])
                node2 = np.array(solution[i+1])
                path_cost += np.linalg.norm(node2-node1)

            if path_cost < minimum_cost:
                minimum_cost = path_cost
                minimum_path = solution
        return minimum_path, minimum_cost

    def rrt_star(
            self, optimal_cost=None, percentage_of_optimal_cost=0.01):
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

        # TODO: Remove
        self.new_Q = self.Q
        while True:
            # TODO: Find the purpose of these loops.
            # Not exists in the original implementation

            if self.X_soln:
                _, c_best = self.findCost(self.X_soln)

            x_new, x_nearest = self.informed_new_and_near(
                0, self.new_Q, c_best, self.RotationMatrix)

            # If x_new is none, then it is not collision free,
            # therefore continue'ing the execution.
            if x_new is None:
                continue

            elif not self.X.collision_free(
                    x_nearest, x_new, self.r):
                continue

            # get nearby vertices and cost-to-come
            L_near = self.get_nearby_vertices(0, self.x_init, x_new)

            # check nearby vertices for total cost and
            # connect shortest valid edge
            self.connect_shortest_valid(0, x_new, L_near)

            if x_new in self.trees[0].E:
                # rewire tree
                self.rewire(0, x_new, L_near)

            if self.InGoalRegion(x_new, self.new_Q):
                self.X_soln.append(self.get_path())

            solution = self.check_solution(
                optimal_cost=optimal_cost,
                percentage_of_optimal_cost=percentage_of_optimal_cost)
            if solution[0]:
                return solution[1]
