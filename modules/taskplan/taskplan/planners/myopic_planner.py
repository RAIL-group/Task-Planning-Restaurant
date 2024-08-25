import taskplan
import pddlstream
import pddlstream.algorithms.meta
import pddlstream.language.constants
import os
import time
from pddlstream.algorithms.search import solve_from_pddl
import torch


def get_tasks():
    return [
        # Water in several places
        taskplan.pddl.task.serve_water('servingtable1', 'cup1'),
        taskplan.pddl.task.serve_water('servingtable2', 'cup1'),
        taskplan.pddl.task.serve_water('servingtable3', 'cup1'),
        taskplan.pddl.task.serve_water('servingtable1', 'cup2'),
        taskplan.pddl.task.serve_water('servingtable2', 'cup2'),
        taskplan.pddl.task.serve_water('servingtable3', 'cup2'),
        taskplan.pddl.task.serve_water('servingtable1', 'mug1'),
        taskplan.pddl.task.serve_water('servingtable2', 'mug1'),
        taskplan.pddl.task.serve_water('servingtable3', 'mug1'),
        taskplan.pddl.task.serve_water('servingtable1', 'mug2'),
        taskplan.pddl.task.serve_water('servingtable2', 'mug2'),
        taskplan.pddl.task.serve_water('servingtable3', 'mug2'),
        taskplan.pddl.task.serve_water('servingtable1'),
        taskplan.pddl.task.serve_water('servingtable2'),
        taskplan.pddl.task.serve_water('servingtable3'),
        # Coffe Serving
        taskplan.pddl.task.fill_coffeemachine_with_water(),
        taskplan.pddl.task.serve_coffee('servingtable1', 'cup1'),
        taskplan.pddl.task.serve_coffee('servingtable2', 'cup1'),
        taskplan.pddl.task.serve_coffee('servingtable3', 'cup1'),
        taskplan.pddl.task.serve_coffee('servingtable1', 'cup2'),
        taskplan.pddl.task.serve_coffee('servingtable2', 'cup2'),
        taskplan.pddl.task.serve_coffee('servingtable3', 'cup2'),
        taskplan.pddl.task.serve_coffee('servingtable1', 'mug1'),
        taskplan.pddl.task.serve_coffee('servingtable2', 'mug1'),
        taskplan.pddl.task.serve_coffee('servingtable3', 'mug1'),
        taskplan.pddl.task.serve_coffee('servingtable1', 'mug2'),
        taskplan.pddl.task.serve_coffee('servingtable2', 'mug2'),
        taskplan.pddl.task.serve_coffee('servingtable3', 'mug2'),
        taskplan.pddl.task.serve_coffee('servingtable1'),
        taskplan.pddl.task.serve_coffee('servingtable2'),
        taskplan.pddl.task.serve_coffee('servingtable3'),
        # Serving Sandwich
        taskplan.pddl.task.serve_sandwich('servingtable1', 'orangespread'),
        taskplan.pddl.task.serve_sandwich('servingtable2', 'orangespread'),
        taskplan.pddl.task.serve_sandwich('servingtable3', 'orangespread'),
        taskplan.pddl.task.serve_sandwich('servingtable1', 'peanutbutterspread'),
        taskplan.pddl.task.serve_sandwich('servingtable2', 'peanutbutterspread'),
        taskplan.pddl.task.serve_sandwich('servingtable3', 'peanutbutterspread'),
        taskplan.pddl.task.serve_sandwich('servingtable1'),
        taskplan.pddl.task.serve_sandwich('servingtable2'),
        taskplan.pddl.task.serve_sandwich('servingtable3'),

        # Clearing the surface
        taskplan.pddl.task.clear_surface('servingtable1'),
        taskplan.pddl.task.clear_surface('servingtable2'),
        taskplan.pddl.task.clear_surface('servingtable3'),
        taskplan.pddl.task.clear_surface('countertop'),

        # Place Spreads
        taskplan.pddl.task.place_something('orangespread', 'spreadshelf'),
        taskplan.pddl.task.place_something('peanutbutterspread',
                                           'spreadshelf'),
        # Clean
        taskplan.pddl.task.clean_something('knife1'),
        taskplan.pddl.task.clean_something('knife2'),
        taskplan.pddl.task.clean_something('cup1'),
        taskplan.pddl.task.clean_something('cup2'),
        taskplan.pddl.task.clean_something('mug1'),
        taskplan.pddl.task.clean_something('mug2'),

        # Keep items at their location
        taskplan.pddl.task.clean_and_place('mug1', 'cupshelf'),
        taskplan.pddl.task.clean_and_place('mug2', 'cupshelf'),
        taskplan.pddl.task.clean_and_place('cup1', 'cupshelf'),
        taskplan.pddl.task.clean_and_place('cup1', 'cupshelf'),
        taskplan.pddl.task.clean_and_place('knife1', 'cutleryshelf'),
        taskplan.pddl.task.clean_and_place('knife2', 'cutleryshelf'),
    ]


class MyopicPlanner:
    def __init__(self, domain=taskplan.pddl.domain.get_domain(), args=None):
        self.domain = domain
        self.planner = 'ff-astar2'
        if args is not None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            self.eval_net = taskplan.models.gcn.AnticipateGCN.get_net_eval_fn(
                network_file=args.network_file, device=device)

    def get_anticipated_cost(self, proc_data):
        whole_graph = taskplan.utils.get_graph(proc_data)
        anticipated_cost = self.eval_net(whole_graph)
        return anticipated_cost

    def get_cost_and_state_from_task(self, proc_data, task):
        pddl_problem = taskplan.pddl.problem.get_problem(proc_data, task)
        count = 0
        planners = ['ff-astar', 'ff-astar2',
                    'ff-wastar2', 'ff-eager', 'ff-lazy']
        max_time = [60, 60,
                    60, 60, 60]
        while count < len(planners):
            planner = planners[count]
            # plan, cost = solution
            plan, cost = solve_from_pddl(
                self.domain,
                pddl_problem,
                planner=planner,
                max_planner_time=max_time[count]
            )
            if plan is None:
                count += 1
            else:
                return plan, cost
        return None, None

    def get_expected_cost(self, proc_data, task_distribution,
                          return_alt_goals=False):
        expected_costs = list()
        alternate_goal_states = list()
        for task in task_distribution:
            plan, cost = self.get_cost_and_state_from_task(proc_data, task)
            if cost is None:
                print(task)
                if return_alt_goals:
                    # raise NotImplementedError
                    return None, None
                else:
                    return None
            expected_costs.append(cost)
            if return_alt_goals and cost != 0:
                final_state = proc_data.get_final_state_from_plan(
                    plan)
                alternate_goal_states.append(final_state)
        expected_cost = sum(expected_costs)/len(expected_costs)
        if return_alt_goals:
            return alternate_goal_states, expected_cost
        return expected_cost

    def get_seq_cost(self, args, proc_data, task_seq,
                     seq_num,
                     prep=False, all_task=False):
        file_name = 'no_prep_myopic/beta_' + args.logfile_name
        if prep:
            file_name = 'prep_myopic/beta_' + args.logfile_name
        logfile = os.path.join(args.save_dir, file_name)
        total_cost = 0
        start_time = time.time()
        # print("Myopic:")
        for idx, task in enumerate(task_seq):
            # print(proc_data.rob_at)
            exp_bef = self.get_anticipated_cost(proc_data)
            # if all_task:
            #     e_tasks = get_tasks()
            #     exp_aft = self.get_expected_cost(proc_data, e_tasks)
            key = list(task.keys())[0]
            val = task[key]
            plan, cost = (
                self.get_cost_and_state_from_task(
                    proc_data, val)
            )
            if plan is None or cost is None:
                print(val)
                print(key)
                return None, None
            new_state = proc_data.get_final_state_from_plan(plan)
            total_cost += cost
            proc_data.update_container_props(new_state[0],
                                             new_state[1],
                                             new_state[2])
            with open(logfile, "a+") as f:
                f.write(
                    f" | seq: S{seq_num}"
                    f" | desc: {key}"
                    f" | num: T{idx+1}"
                    f" | cost: {cost:0.4f}"
                    f" | ec_est: {exp_bef:0.4f}\n"
                    # f" | ec_tr: {exp_aft:0.4f}\n"
                )
        return total_cost, time.time() - start_time
