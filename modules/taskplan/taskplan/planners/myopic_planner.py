import taskplan
import pddlstream
import pddlstream.algorithms.meta
import pddlstream.language.constants
import os
import time
from pddlstream.algorithms.search import solve_from_pddl


class MyopicPlanner:
    def __init__(self, domain=taskplan.pddl.domain.get_domain()):
        self.domain = domain
        self.planner = 'ff-astar2'

    def get_cost_and_state_from_task(self, proc_data, task):
        pddl_problem = taskplan.pddl.problem.get_problem(proc_data, task)
        # plan, cost = solution
        plan, cost = solve_from_pddl(
            self.domain,
            pddl_problem,
            planner=self.planner,
            max_planner_time=600
        )
        if plan is None:
            return None, None
        return plan, cost

    def get_expected_cost(self, proc_data, task_distribution,
                          return_alt_goals=False):
        expected_costs = list()
        alternate_goal_states = list()
        for task in task_distribution:
            plan, cost = self.get_cost_and_state_from_task(proc_data, task)
            if cost is None:
                # print(task)
                # raise NotImplementedError
                return None, None
            expected_costs.append(cost)
            if return_alt_goals and cost != 0:
                goal_state = proc_data.get_final_state_from_plan(plan)
                alternate_goal_states.append(goal_state)
        expected_cost = sum(expected_costs)/len(expected_costs)
        if return_alt_goals:
            return alternate_goal_states, expected_cost
        return expected_cost
