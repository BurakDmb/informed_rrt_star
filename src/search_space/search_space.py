# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
from rtree import index

from src.utilities.geometry import es_points_along_line
from src.utilities.obstacle_generation import obstacle_generator


class SearchSpace(object):
    def __init__(self, dimension_lengths, Obstacles=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p = index.Property()
        p.dimension = self.dimensions
        if Obstacles is None:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            # r-tree representation of obstacles
            # sanity check
            if any(len(o) / 2 != len(dimension_lengths) for o in Obstacles):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)]
                    for o in Obstacles for i in range(int(len(o) / 2))):
                raise Exception(
                    "Obstacle start must be less than obstacle end")
            self.obs = index.Index(
                obstacle_generator(Obstacles), interleaved=True, properties=p)

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        return self.obs.count(x) == 0

    # Informed RRT* Algorithm Sample Function
    def informed_rrt_sample(self, x_start, x_goal, c_max, rotationMatrix):
        """
        Informed RRT Sample Function
        :return: sample location within X_free
        """

        x_start_arr = np.array(x_start)
        x_goal_arr = np.array(x_goal)

        # TODO: Check for input and output data types.
        if c_max < float("inf"):
            c_min = np.linalg.norm(x_goal_arr - x_start_arr)
            x_center = (x_start_arr + x_goal_arr) / 2
            r = [c_max / 2.0,
                 np.sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                 np.sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            L = np.diag(r)

            while True:
                x_ball = self.SampleUnitBall()
                x_rand = np.dot(np.dot(rotationMatrix, L), x_ball) + x_center

                # If randomized state is in the environment
                if (self.dimension_lengths[0, 0] <= x_rand[0]
                        <= self.dimension_lengths[0, 1]) and \
                    (self.dimension_lengths[1, 0] <= x_rand[1]
                        <= self.dimension_lengths[1, 1]):
                    break
            return tuple(x_rand)
        else:
            while True:  # sample until not inside of an obstacle
                x = self.sample()
                if self.obstacle_free(x):
                    return x

    def SampleUnitBall(self):
        while True:
            x, y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.obstacle_free(x):
                return x

    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when
        checking for collisions
        :return: True if line segment does not intersect an obstacle,
        False otherwise
        """
        points = es_points_along_line(start, end, r)
        coll_free = all(map(self.obstacle_free, points))
        return coll_free

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(
            self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)
