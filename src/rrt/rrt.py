from src.rrt.rrt_base import RRTBase


class RRT(RRTBase):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge
        when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc)

    def rrt_search(self, optimal_cost=None, percentage_of_optimal_cost=0.01):
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding
        until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of
        tree in form E[child] = parent
        """
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        self.cost_versus_time_list = []

        while True:
            x_new, x_nearest = self.new_and_near(0, self.Q)

            if x_new is None:
                continue

            # connect shortest valid edge
            self.connect_to_point(0, x_nearest, x_new)

            solution = self.check_solution(
                optimal_cost=optimal_cost,
                percentage_of_optimal_cost=percentage_of_optimal_cost)

            # Cost versus time plot data gathering
            self.cost_versus_time_list.append(solution[2])

            if solution[0]:
                return solution[1]
