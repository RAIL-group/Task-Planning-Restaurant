import taskplan_multi
import pddlstream
import pddlstream.algorithms.meta
import pddlstream.language.constants
from pddlstream.algorithms.search import solve_from_pddl


class MyopicPlanner:
    def __init__(self, domain=taskplan_multi.pddl.domain.get_domain(), args=None):
        self.domain = domain

    def get_cost_and_state_from_task(self, proc_data, task):
        pddl_problem = taskplan_multi.pddl.problem.get_problem(proc_data, task)
        planner = 'ff-astar'
        plan, cost = solve_from_pddl(
            self.domain,
            pddl_problem,
            planner=planner,
            max_planner_time=120
        )
        return plan, cost
    
    def get_expected_cost(self, proc_data, task_distribution):
        expected_costs = list()
        for task in task_distribution:
            plan, cost = self.get_cost_and_state_from_task(proc_data, task)
            # print(cost)
            if plan is None:
                expected_costs.append(10000)
            else:
                expected_costs.append(cost)
        # print(expected_costs)
        expected_cost = sum(expected_costs)/len(expected_costs)
        return expected_cost
