import taskplan_multi
import torch
import copy
import random
import math
from itertools import combinations_with_replacement, combinations
from itertools import permutations, product
import numpy as np
import time
import os
import itertools
import matplotlib.pyplot as plt

COOK_BOT_REACHABLES = ['stove', 'fridge', 'countertop']
SERVER_BOT_REACHABLES = ['servingtable1', 'servingtable2', 'cabinet', 'countertop', 'bussingcart']
CLEANER_BOT_REACHABLES = ['dishwasher', 'bussingcart', 'countertop']
MAX_AUG = 2

class AntcipatoryPlanner:
    def __init__(self, args, domain=taskplan_multi.pddl.domain.get_domain()):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.domain = domain
        self.eval_nets = {
            'cook_bot': taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(network_file=args.cook_network, device=device),
            'server_bot': taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(network_file=args.server_network, device=device),
            'cleaner_bot': taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(network_file=args.cleaner_network, device=device),
        }
        self.myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner(
            self.domain)
        self.concern = 'joint'

    def get_anticipated_cost(self, restaurant):
        current_active = restaurant.active_robot
        # image_graph = taskplan_multi.utils.get_image_for_data(whole_graph)
        # plt.clf()
        # plt.imshow(image_graph)
        # plt.savefig('/data/figs/graph.png', dpi=600)
        # restaurant.active_robot = key
        # whole_graph = taskplan_multi.utils.get_graph(restaurant)
        other_ac = list()
        for key in self.eval_nets:
            restaurant.active_robot = key
            whole_graph = taskplan_multi.utils.get_graph(restaurant)
            c = self.eval_nets[key](whole_graph)
            if key == current_active:
                self_ac = c
            other_ac.append(c)
        restaurant.active_robot = current_active
        if self.concern == 'joint':
            return sum(other_ac)
        if self.concern == 'self':
            return self_ac
        if self.concern == 'max':
            return max(other_ac)
        if self.concern == 'min':
            return min(other_ac)

    
    def get_seq_cost(self, args, restaurant, task_seq, seq_num, ap_concern='joint'):
        costs = list()
        file_name = 'ap_' + ap_concern + '.txt'
        task_file_name = ap_concern + '_tasks.txt'
        task_logfile = os.path.join(args.save_dir, task_file_name)
        self.concern = ap_concern
        logfile = os.path.join(args.save_dir, file_name)
        for idx, item in enumerate(task_seq):
            active_agent = item[0]
            task = item[1]
            restaurant.active_robot = active_agent
            new_state, cost, new_task = self.get_anticipatory_plan(restaurant, task)
            costs.append(cost)
            with open(task_logfile, "a+") as f:
                f.write(
                    f" | active: {active_agent}"
                    f" | task: {new_task} \n"
                )
            with open(logfile, "a+") as f:
                f.write(
                    f" | seq: S{seq_num}"
                    f" | num: T{idx+1}"
                    f" | cost: {cost:0.4f} \n"
                )
            restaurant.update_container_props(new_state)

    def get_anticipatory_plan(self, restaurant, task):
        init_state = copy.deepcopy(restaurant.get_current_object_state())
        plan, cost = (
            self.myopic_planner.get_cost_and_state_from_task(restaurant, task)
        )
        if plan is None:
            return init_state, 10000, task
        # if cost == 0:
        #     return init_state, cost, task
        used_items = set() 
        if restaurant.active_robot == 'server_bot':
            for p in plan:
                if "place" in p.name:
                    used_items.add(p.args[1])
                if "wash" in p.name:
                    used_items.add(p.args[1])
                if "cook" in p.name:
                    used_items.add(p.args[1])
                    used_items.add(p.args[2])
                if "serve" in p.name:
                    used_items.add(p.args[1])
        term_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(term_state)
        myopic_ex_cost = self.get_anticipated_cost(restaurant)
        myopic_ant_cost = myopic_ex_cost + cost
        ant_cost = cost
        ant_state = copy.deepcopy(term_state)
        conts = list()
        aug_predicates = list()
        if restaurant.active_robot == 'cook_bot':
            conts = COOK_BOT_REACHABLES
            for cnt in conts:
                ant_objects = restaurant.get_objects_by_container_name(cnt)
                for tmp in ant_objects:
                    if tmp['assetId'] in used_items:
                        continue
                    if 'cookable' in tmp and ('cooked' not in tmp or tmp['cooked'] == 0):
                        aug_predicates.append(taskplan_multi.pddl.task.cook_something(tmp['assetId']))
                    for other_cnt in conts:
                        if other_cnt != cnt:  # Exclude the current container
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(tmp['assetId'], other_cnt))
                    
        elif restaurant.active_robot == 'server_bot':
            conts = SERVER_BOT_REACHABLES
            for cnt in conts:
                ant_objects = restaurant.get_objects_by_container_name(cnt)
                for tmp in ant_objects:
                    if tmp['assetId'] in used_items:
                        continue
                    for other_cnt in conts:
                        if other_cnt != cnt:  # Exclude the current container
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(tmp['assetId'], other_cnt))
        else:
            conts = CLEANER_BOT_REACHABLES
            for cnt in conts:
                ant_objects = restaurant.get_objects_by_container_name(cnt)
                for tmp in ant_objects:
                    if tmp['assetId'] in used_items:
                        continue
                    if 'washable' in tmp and 'dirty' in tmp and tmp['dirty'] == 1:
                        aug_predicates.append(taskplan_multi.pddl.task.clean_something(tmp['assetId']))
                    for other_cnt in conts:
                        if other_cnt != cnt:  # Exclude the current container
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(tmp['assetId'], other_cnt))
        # ant_objects = list()
        # for obj in init_state:
        #     if obj['assetId'] not in task:
        #         ant_objects.append(obj['assetId'])
        # ant_task = list(product(ant_objects, conts))
        ant_task = task
        
        for aug_pred in aug_predicates:
            restaurant.update_container_props(init_state)
            # aug_pred = taskplan_multi.pddl.task.place_something(td[0], td[1])
            ant_task_pred = f'(and {aug_pred} {task})'
            plan, c_cost = (
                self.myopic_planner.get_cost_and_state_from_task(
                    restaurant, ant_task_pred))
            if plan is None:
                continue
            can_state = restaurant.get_final_state_from_plan(plan)
            restaurant.update_container_props(can_state)
            can_ex_cost = self.get_anticipated_cost(restaurant)
            can_ant_cost = can_ex_cost + c_cost
            if can_ex_cost < myopic_ex_cost and can_ant_cost < myopic_ant_cost:
                ant_state = copy.deepcopy(can_state)
                ant_cost = c_cost
                myopic_ant_cost = can_ant_cost
                ant_task = ant_task_pred
        # if len(aug_predicates) > 5: 
        #     for i in range(20):
        #         restaurant.update_container_props(init_state)
        #         aug_pred = random.sample(aug_predicates, MAX_AUG)
        #         ant_task_pred = f'(and {aug_pred[0]} {aug_pred[1]} {task})'
        #         plan, c_cost = (
        #             self.myopic_planner.get_cost_and_state_from_task(
        #                 restaurant, ant_task_pred))
        #         if plan is None:
        #             continue
        #         can_state = restaurant.get_final_state_from_plan(plan)
        #         restaurant.update_container_props(can_state)
        #         can_ex_cost = self.get_anticipated_cost(restaurant)
        #         can_ant_cost = can_ex_cost + c_cost
        #         if can_ex_cost < myopic_ex_cost and can_ant_cost < myopic_ant_cost:
        #             ant_state = copy.deepcopy(can_state)
        #             ant_cost = c_cost
        #             myopic_ant_cost = can_ant_cost
        #             ant_task = ant_task_pred
        return ant_state, ant_cost, ant_task

    def get_ap_plan_only(self, restaurant, task):
        init_state = copy.deepcopy(restaurant.get_current_object_state())
        plan, cost = (
            self.myopic_planner.get_cost_and_state_from_task(restaurant, task)
        )
        if plan is None:
            return None, 'inf'
        if cost == 0:
            return plan, cost
        term_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(term_state)
        myopic_ex_cost = self.get_anticipated_cost(restaurant)
        myopic_ant_cost = myopic_ex_cost + cost
        ant_cost = cost
        ant_state = copy.deepcopy(term_state)
        conts = [c for (c, v) in restaurant.get_container_pos_list()]
        ant_objects = list()
        for obj in init_state:
            if obj['assetId'] not in task:
                ant_objects.append(obj['assetId'])
        ant_task = list(product(ant_objects, conts))
        ant_plan = copy.deepcopy(plan)
        for td in ant_task:
            restaurant.update_container_props(init_state)
            aug_pred = taskplan_multi.pddl.task.place_something(td[0], td[1])
            ant_task_pred = f'(and {aug_pred} {task})'
            plan, c_cost = (
                self.myopic_planner.get_cost_and_state_from_task(
                    restaurant, ant_task_pred))
            if plan is None:
                continue
            can_state = restaurant.get_final_state_from_plan(plan)
            restaurant.update_container_props(can_state)
            can_ex_cost = self.get_anticipated_cost(restaurant)
            can_ant_cost = can_ex_cost + c_cost
            if can_ex_cost < myopic_ex_cost and can_ant_cost < myopic_ant_cost:
                ant_state = copy.deepcopy(can_state)
                ant_cost = c_cost
                myopic_ant_cost = can_ant_cost
                ant_plan = copy.deepcopy(plan)
        return ant_plan, ant_cost
    
    def get_prepared_state(self, restaurant, task_sequence, n_iterations=1000):
        def safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                # If the argument is too large in magnitude, return an approximation
                return 0.0
        save_file = '/data/figs/learned_prep' + str(
            restaurant.seed) + '.txt'

        prepared_state = copy.deepcopy(restaurant.get_current_object_state())
        int_cost = self.get_anticipated_cost(restaurant)
        with open(save_file, "a+") as f:
            f.write(
                f"| Initial State"
                f"| Expected Cost: {int_cost}\n")
        i = 0
        temp = 1000
        cooling_rate = 0.99
        for idx, item in enumerate(task_seq):
            active_agent = item[0]
            task = item[1]
            restaurant.active_robot = active_agent
            new_state, cost, new_task = self.get_anticipatory_plan(restaurant, task)
            costs.append(cost)
        while (i < n_iterations):
            proc_data.update_container_props(prepared_state)
            i += 1
            item = random.choice(task_sequence)
            active_agent = item[0]
            task_to_solve = item[1]
            restaurant.active_robot = active_agent
            plan, cost = (
                self.myopic_planner.get_cost_and_state_from_task(
                    restaurant, task)
            )
            # candidate_state = proc_data.randomize_objects()
            if plan is None:
                continue

            candidate_state = change_state(proc_data)
            proc_data.update_container_props(candidate_state[0],
                                             candidate_state[1],
                                             candidate_state[2])
            can_exp_cost = self.get_anticipated_cost(proc_data)
            delta = can_exp_cost-int_cost
            exp = safe_exp(-delta/temp)
            if delta < 0 or random.uniform(0, 1) < exp:
                prepared_state = copy.deepcopy(candidate_state)
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