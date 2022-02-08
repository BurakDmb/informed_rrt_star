import random

import numpy as np

from src.rrt.tree import Tree
from src.utilities.geometry import steer


class RRTBase(object):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when
        checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.Q = Q
        self.r = r
        self.prc = prc
        self.x_init = x_init
        self.x_goal = x_goal
        self.trees = []  # list of all trees
        self.add_tree()  # add initial tree

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(Tree(self.X))

    def add_vertex(self, tree, v):
        """
        Add vertex to corresponding tree
        :param tree: int, tree to which to add vertex
        :param v: tuple, vertex to add
        """
        self.trees[tree].V.insert(0, v + v, v)
        self.trees[tree].V_count += 1  # increment number of vertices in tree
        self.samples_taken += 1  # increment number of samples taken

    def add_edge(self, tree, child, parent):
        """
        Add edge to corresponding tree
        :param tree: int, tree to which to add vertex
        :param child: tuple, child vertex
        :param parent: tuple, parent vertex
        """
        self.trees[tree].E[child] = parent

    def nearby(self, tree, x, n):
        """
        Return nearby vertices
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :param n: int, max number of neighbors to return
        :return: list of nearby vertices
        """
        return self.trees[tree].V.nearest(x, num_results=n, objects="raw")

    def get_nearest(self, tree, x):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """
        return next(self.nearby(tree, x, 1))

    def informed_new_and_near(self, tree, q, c_best, rotationMatrix):
        """
        Return a new steered vertex and the vertex in tree that is nearest
        :param tree: int, tree being searched
        :param q: length of edge when steering
        :return: vertex, new steered vertex, vertex,
        nearest vertex in tree to new vertex
        """
        x_rand = self.X.informed_rrt_sample(
            self.x_init, self.x_goal, c_best, rotationMatrix)
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))
        # check if new point is in X_free and not already in V
        if not self.trees[0].V.count(x_new) == 0 \
                or not self.X.obstacle_free(x_new):
            return None, None
        self.samples_taken += 1
        return x_new, x_nearest

    def new_and_near(self, tree, q):
        """
        Return a new steered vertex and the vertex in tree that is nearest
        :param tree: int, tree being searched
        :param q: length of edge when steering
        :return: vertex, new steered vertex, vertex,
        nearest vertex in tree to new vertex
        """
        x_rand = self.X.sample_free()
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))
        # check if new point is in X_free and not already in V
        if not self.trees[0].V.count(x_new) == 0 \
                or not self.X.obstacle_free(x_new):
            return None, None
        self.samples_taken += 1
        return x_new, x_nearest

    def connect_to_point(self, tree, x_a, x_b):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex
        :param x_b: tuple, vertex
        :return: bool, True if able to add edge,
        False if prohibited by an obstacle
        """
        if self.trees[tree].V.count(x_b) == 0 \
                and self.X.collision_free(x_a, x_b, self.r):
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False

    def can_connect_to_goal(self, tree):
        """
        Check if the goal can be connected to the graph
        :param tree: rtree of all Vertices
        :return: True if can be added, False otherwise
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        if self.x_goal in self.trees[tree].E \
                and x_nearest == self.trees[tree].E[self.x_goal]:
            # tree is already connected to goal using nearest vertex
            return True

        if (np.linalg.norm(
                np.array(self.x_goal) - np.array(x_nearest)) < self.Q).all():

            # check if obstacle-free
            if self.X.collision_free(x_nearest, self.x_goal, self.r):
                return True

        return False

    def get_path(self):
        """
        Return path through tree from start to goal
        :return: path if possible, None otherwise
        """
        if self.can_connect_to_goal(0):
            # print("Can connect to goal")
            self.connect_to_goal(0)
            return self.reconstruct_path(0, self.x_init, self.x_goal)
        # print("Could not connect to goal")
        return None

    def connect_to_goal(self, tree):
        """
        Connect x_goal to graph
        (does not check if this should be possible,
        for that use: can_connect_to_goal)
        :param tree: rtree of all Vertices
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        self.trees[tree].E[self.x_goal] = x_nearest

    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :param tree: int, tree in which to find path
        :param x_init: tuple, starting vertex
        :param x_goal: tuple, ending vertex
        :return: sequence of vertices from start to goal
        """
        path = [x_goal]
        current = x_goal
        if x_init == x_goal:
            return path
        while not self.trees[tree].E[current] == x_init:
            path.append(self.trees[tree].E[current])
            current = self.trees[tree].E[current]
        path.append(x_init)
        path.reverse()
        return path

    # This function calculates the cost of the given path array.
    # Path array is simply list of states connecting from start to goal
    def calculate_cost_of_path(self, path):
        if path is None:
            return np.nan
        else:
            path_cost = 0.0
            for i in range(len(path)-1):
                node1 = np.array(path[i])
                node2 = np.array(path[i+1])
                path_cost += np.linalg.norm(node2-node1)
            return path_cost

    def check_solution(
            self, optimal_cost=None, percentage_of_optimal_cost=0.01,
            record_cost_per_timestep=False):
        if optimal_cost is not None:
            path = self.get_path()
            if path is not None:

                path_cost = self.calculate_cost_of_path(path)
                cost_percent = np.abs((path_cost / optimal_cost) - 1)
                if cost_percent < percentage_of_optimal_cost:
                    resultSolutionExists, resultSolution,\
                        resultPathCost = True, path, path_cost
                else:
                    resultSolutionExists, resultSolution,\
                        resultPathCost = False, path, path_cost
            else:
                resultSolutionExists, resultSolution,\
                    resultPathCost = False, None, np.nan
        else:
            # probabilistically check if solution found
            if self.prc and random.random() < self.prc:
                path = self.get_path()
                if path is not None:
                    resultSolutionExists, resultSolution,\
                        resultPathCost = True, path, np.nan
                else:
                    resultSolutionExists, resultSolution, \
                        resultPathCost = False, None, np.nan
            # check if can connect to goal after generating max_samples
            elif self.samples_taken >= self.max_samples:
                path = self.get_path()
                resultSolutionExists, resultSolution,\
                    resultPathCost = True, \
                    path, self.calculate_cost_of_path(path)
            else:
                path = self.get_path()
                resultSolutionExists, resultSolution, \
                    resultPathCost = False, None,\
                    self.calculate_cost_of_path(path)

        return resultSolutionExists, resultSolution, resultPathCost

    def bound_point(self, point):
        # if point is out-of-bounds, set to bound
        point = np.maximum(point, self.X.dimension_lengths[:, 0])
        point = np.minimum(point, self.X.dimension_lengths[:, 1])
        return tuple(point)
