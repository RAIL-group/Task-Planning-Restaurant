import taskplan
import torch
import copy
import random
import math
from itertools import combinations_with_replacement, combinations
from itertools import permutations, product
import numpy as np
import time
import os


class AntcipatoryPlanner:
    def __init__(self, args, domain=taskplan.pddl.domain.get_domain()):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.domain = domain
        self.eval_net = taskplan.models.gcn.AnticipateGCN.get_net_eval_fn(
            network_file=args.network_file, device=device)
        self.myopic_planner = taskplan.planners.myopic_planner.MyopicPlanner(
            self.domain)

    def get_anticipated_cost(self, proc_data):
        whole_graph = taskplan.utils.get_graph(proc_data)
        anticipated_cost = self.eval_net(whole_graph)
        return anticipated_cost

    def get_prepared_state(self, proc_data,
                           task_list, n_iterations=1000):
        def safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                # If the argument is too large in magnitude, return an approximation
                return 0.0
        save_file = '/data/figs/learned_prep' + str(
            proc_data.seed) + '.txt'

        prepared_state = None
        int_cost = self.get_anticipated_cost(proc_data)
        with open(save_file, "a+") as f:
            f.write(
                f"| Initial State"
                f"| Expected Cost: {int_cost}\n")
        i = 0
        temp = 1000
        cooling_rate = 0.99
        while (i < n_iterations):
            i += 1
            task_to_solve = random.choice(task_list)
            plan, _ = (
                    self.myopic_planner.get_cost_and_state_from_task(
                        proc_data, task_to_solve))
            # candidate_state = proc_data.randomize_objects()
            if plan is None:
                continue
            candidate_state = proc_data.get_final_state_from_plan(plan)
            proc_data.update_container_props(candidate_state)
            can_exp_cost = self.get_anticipated_cost(proc_data)
            delta = round(can_exp_cost-int_cost, 2)
            exp = safe_exp(-delta/temp)
            if delta < 0:
                prepared_state = copy.deepcopy(candidate_state)
                proc_data.update_container_props(prepared_state)
                int_cost = can_exp_cost
                with open(save_file, "a+") as f:
                    f.write(
                        f"| idx: {i}"
                        # f"| task: {task_to_solve}"
                        f"| exp: {exp}"
                        f"| delta: {delta}"
                        f"| cost: {int_cost}\n")
            temp = max(temp * cooling_rate, 1)
        with open(save_file, "a+") as f:
            f.write(
                f"| found after: {i}"
                f"| state: {prepared_state}"
                f"| prepared: {int_cost}\n")
        return prepared_state
