import os
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
                expected_costs.append(5000)
            else:
                expected_costs.append(cost)
        # print(expected_costs)
        expected_cost = sum(expected_costs)/len(expected_costs)
        return expected_cost
    
    def get_seq_cost(self, args, restaurant, task_seq, seq_num):
        file_name = 'myopic.txt'
        logfile = os.path.join(args.save_dir, file_name)
        task_file_name = 'tasks.txt'
        task_logfile = os.path.join(args.save_dir, task_file_name)
        costs = list()
        for idx, item in enumerate(task_seq):
            active_agent = item[0]
            task = item[1]
            with open(task_logfile, "a+") as f:
                f.write(
                    f" | active: {active_agent}"
                    f" | task: {task} \n"
                )
            restaurant.active_robot = active_agent
            plan, cost = (
                self.get_cost_and_state_from_task(
                    restaurant, task)
            )
            if plan is None:
                costs.append(10000)
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | cost: 10000 \n"
                    )
                continue
            with open(logfile, "a+") as f:
                f.write(
                    f" | seq: S{seq_num}"
                    f" | num: T{idx+1}"
                    f" | cost: {cost:0.4f} \n"
                )
            costs.append(cost)
            new_state = restaurant.get_final_state_from_plan(plan)
            restaurant.update_container_props(new_state)
