import os
import taskplan_multi
import pddlstream
import pddlstream.algorithms.meta
import pddlstream.language.constants
from pddlstream.algorithms.search import solve_from_pddl


class MyopicPlanner:
    def __init__(self, domain=taskplan_multi.pddl.workshop_domain.get_domain(), args=None):
        self.domain = domain

    def get_cost_and_state_from_task(self, proc_data, task):
        pddl_problem = taskplan_multi.pddl.workshop_problem.get_problem(proc_data, task)
        planner = 'max-astar'
        plan, cost = solve_from_pddl(
            self.domain,
            pddl_problem,
            planner=planner,
            max_planner_time=120
        )
        # if plan is None:
        #     proc_data.active_all = True
        #     pddl_problem = taskplan_multi.pddl.problem.get_problem(proc_data, task)
        #     mod_plan, cost = solve_from_pddl(
        #         self.domain,
        #         pddl_problem,
        #         planner=planner,
        #         max_planner_time=120
        #     )
        #     proc_data.active_all = False
        #     return mod_plan, cost + 2000
        # else:
        return plan, cost
    
    def get_expected_cost(self, proc_data, task_distribution):
        expected_costs = list()
        for task in task_distribution:
            plan, cost = self.get_cost_and_state_from_task(proc_data, task)
            if plan is None:
                expected_costs.append(5000)
            else:
                expected_costs.append(cost)
        # print(expected_costs)
        expected_cost = sum(expected_costs)/len(expected_costs)
        return expected_cost
    
    def get_oracle_failure_ratio(self, restaurant, task_seq):
        failed = 0
        for idx, item in enumerate(task_seq):
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                plan, cost = (
                    self.get_cost_and_state_from_task(
                        restaurant, task)
                )
                if plan is None:
                    failed+=1
        return failed
    
    def get_seq_cost(self, args, restaurant, task_seq, seq_num, no_prep_state=None, prep_state=None):
        if no_prep_state:
            restaurant.update_container_props(no_prep_state)
            file_name = 'np_myopic.txt'
            logfile = os.path.join(args.save_dir, file_name)
            for idx, item in enumerate(task_seq):
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                restaurant.asked_help = False
                plan, cost = (
                    self.get_cost_and_state_from_task(
                        restaurant, task)
                )
                if plan is None:
                    # costs.append(10000)
                    with open(logfile, "a+") as f:
                        f.write(
                            f" | seq: S{seq_num}"
                            f" | num: T{idx+1}"
                            f" | help: 0"
                            f" | cost: 10000 \n"
                        )
                    continue
                help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | help: {help_stat}"
                        f" | cost: {cost:0.4f} \n"
                    )
                # costs.append(cost)
                new_state = restaurant.get_final_state_from_plan(plan)
                restaurant.update_container_props(new_state)
        
        if prep_state:
            restaurant.update_container_props(prep_state)
            file_name = 'prep_myopic.txt'
            logfile = os.path.join(args.save_dir, file_name)
            for idx, item in enumerate(task_seq):
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                plan, cost = (
                    self.get_cost_and_state_from_task(
                        restaurant, task)
                )
                if plan is None:
                    # costs.append(10000)
                    with open(logfile, "a+") as f:
                        f.write(
                            f" | seq: S{seq_num}"
                            f" | num: T{idx+1}"
                            f" | help: 0"
                            f" | cost: 10000 \n"
                        )
                    continue
                help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | help: {help_stat}"
                        f" | cost: {cost:0.4f} \n"
                    )
                new_state = restaurant.get_final_state_from_plan(plan)
                restaurant.update_container_props(new_state)
