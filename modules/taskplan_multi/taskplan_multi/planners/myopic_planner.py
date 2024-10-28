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
            max_planner_time=max_time[count]
        )
        if plan is None:
            return None, None
        return plan, cost
