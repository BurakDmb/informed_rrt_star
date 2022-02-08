import numpy as np
from rtree import index

from src.utilities.geometry import es_points_along_line
from src.utilities.obstacle_generation import obstacle_generator


class SearchSpace(object):
    def __init__(self, dimension_lengths, Obstacles=None, seed=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        np.random.seed(seed)
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

        if c_max < float("inf"):
            dim = 2
            c_min = np.linalg.norm(x_goal_arr - x_start_arr)
            x_center = (x_start_arr + x_goal_arr) / 2
            r1 = c_max/2
            ri = np.sqrt(c_max**2 - c_min**2)/2

            R = [r1] + [ri]*(dim - 1)
            L = np.diag(R)

            while True:
                x_ball = self.SampleUnitNBall()
                x_rand = tuple((rotationMatrix @ L @ x_ball.T).T + x_center)

                # If randomized state is in the environment
                if (self.dimension_lengths[0, 0] <= x_rand[0]
                        <= self.dimension_lengths[0, 1]) and \
                    (self.dimension_lengths[1, 0] <= x_rand[1]
                        <= self.dimension_lengths[1, 1]):
                    break
            return x_rand
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

    def SampleUnitNBall(self):
        '''
        uniformly sample a N-dimensional unit UnitBall
        Reference:
        https://github.com/Bharath2/Informed-RRT-star/blob/main/PathPlanning/sampleutils.py
        Efficiently sampling vectors and coordinates from
        the n-sphere and n-ball
        http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
        Input:
            num - no. of samples
            dim - dimensions
        Output:
            uniformly sampled points within N-dimensional unit ball
        '''
        dim = 2
        u = np.random.normal(0, 1, (1, dim + 2))
        norm = np.linalg.norm(u, axis=-1, keepdims=True)
        u = u/norm

        return u[0, :dim]

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
