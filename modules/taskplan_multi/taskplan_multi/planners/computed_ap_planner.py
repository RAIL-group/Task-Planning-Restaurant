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
import gc

COOK_BOT_REACHABLES = ['stove', 'fridge', 'countertop']
SERVER_BOT_REACHABLES = ['servingtable1', 'servingtable2', 'cabinet', 'countertop', 'bussingcart']
CLEANER_BOT_REACHABLES = ['dishwasher', 'bussingcart', 'countertop']
MAX_AUG = 2
SCALE = {
    'cook_bot': 0,
    'server_bot': 0,
    'cleaner_bot': 0

}

TIRE_FIXER_REACHABLES = ['tirerack', 'tirefixingstation', 'toolrack']
BATTERY_FIXER_REACHABLES = ['tirerack', 'tirefixingstation', 'toolrack', 'mirrorrack', 'mirrorfixingstation',]
MIRROR_FIXER_REACHABLES = ['mirrorrack', 'mirrorfixingstation', 'toolrack']

def change_state(restaurant):
    candidate_state = None
    while (candidate_state is None):
        objs = restaurant.get_current_object_state()
        obj = random.choice(objs)
        cont_list = [v for (c, v) in restaurant.get_container_pos_list()]
        container = random.choice(cont_list)
        if random.random() > 0.5:
            candidate_state = restaurant.place_object(obj, container)
        else:
            if 'washable' in obj:
                if 'dirty' in obj and obj['dirty'] == 1:
                    candidate_state = restaurant.place_washables(obj, container, dirty=0)
                else:
                    candidate_state = restaurant.place_washables(obj, container, dirty=1)
            elif 'cookable' in obj:
                if 'cooked' in obj and obj['cooked'] == 1:
                    candidate_state = restaurant.place_food_items(obj, container, cooked=0)
                else:
                    candidate_state = restaurant.place_food_items(obj, container, cooked=1)
            else:
                candidate_state = restaurant.place_object(obj, container)
    return candidate_state

def get_representative_value(values, weights=None):
    """
    Calculate a representative value for a list of numbers.

    Args:
        values (list): List of numbers to process.
        weights (list): Optional weights for the values (default: equal weights).

    Returns:
        float: Representative value for the list.
    """
    if not values or len(values) == 0:
        return 0  # Handle empty list

    if weights is None:
        # Default to equal weights
        weights = [1] * len(values)

    # Normalize weights
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]

    # Normalize values relative to their sum
    value_sum = sum(values)
    if value_sum == 0:
        return 0  # Avoid division by zero
    normalized_values = [v / value_sum for v in values]

    # Compute representative value
    representative_value = sum(w * v for w, v in zip(normalized_weights, values))
    return representative_value

class AntcipatoryPlanner:
    def __init__(self, args, domain=taskplan_multi.pddl.workshop_domain.get_domain()):
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
        tasks = list()
        probs = [1,1,1,1,1,1]
        if self.concern == 'self':
            if current_active == 'tire_bot':
                tasks.append(
                    ('tire_bot', '(and (not (is-bad tire1)) (is-at tire1 tirerack))')
                )
                tasks.append(
                    ('tire_bot', '(and (not (is-bad tire2)) (is-at tire2 tirerack))')
                )
                tasks.append(
                    ('tire_bot', '(and (not (is-bad tire3)) (is-at tire3 tirerack))')
                )
            else:
                tasks.append(
                    ('mirror_bot', '(and (not (is-bad mirror1)) (is-at mirror1 mirrorrack))')
                )
                tasks.append(
                    ('mirror_bot', '(and (not (is-bad mirror2)) (is-at mirror2 mirrorrack))')
                )
                tasks.append(
                    ('mirror_bot', '(and (not (is-bad mirror2)) (is-at mirror2 mirrorrack))')
                )
        else:
            tasks.append(
                ('tire_bot', '(and (not (is-bad tire1)) (is-at tire1 tirerack))')
            )
            tasks.append(
                ('tire_bot', '(and (not (is-bad tire2)) (is-at tire2 tirerack))')
            )
            tasks.append(
                ('tire_bot', '(and (not (is-bad tire3)) (is-at tire3 tirerack))')
            )
            tasks.append(
                ('mirror_bot', '(and (not (is-bad mirror1)) (is-at mirror1 mirrorrack))')
            )
            tasks.append(
                ('mirror_bot', '(and (not (is-bad mirror2)) (is-at mirror2 mirrorrack))')
            )
            tasks.append(
                ('mirror_bot', '(and (not (is-bad mirror3)) (is-at mirror3 mirrorrack))')
            )
        exp_cost = 0
        for idx, item in enumerate(tasks):
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                plan, cost = (
                    self.myopic_planner.get_cost_and_state_from_task(
                        restaurant, task)
                )
                prob_cost = probs[idx] * cost
                exp_cost += prob_cost
        restaurant.active_robot = current_active
        return exp_cost




    
    def get_seq_cost(self, args, restaurant, task_seq, seq_num, no_prep_state=None, prep_state=None, ap_concern='joint'):
        if prep_state:
            restaurant.update_container_props(prep_state)
            file_name = 'prep_ap_' + ap_concern + '.txt'
            self.concern = ap_concern
            logfile = os.path.join(args.save_dir, file_name)
            for idx, item in enumerate(task_seq):
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                new_state, cost, new_task, help_stat = self.get_anticipatory_plan(restaurant, task)
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | help: {help_stat}"
                        f" | cost: {cost:0.4f} \n"
                    )
                restaurant.update_container_props(new_state)
        
        if no_prep_state:
            restaurant.update_container_props(no_prep_state)
            file_name = 'np_ap_' + ap_concern + '.txt'
            self.concern = ap_concern
            logfile = os.path.join(args.save_dir, file_name)
            for idx, item in enumerate(task_seq):
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                new_state, cost, new_task, help_stat = self.get_anticipatory_plan(restaurant, task)
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | help: {help_stat}"
                        f" | cost: {cost:0.4f} \n"
                    )
                restaurant.update_container_props(new_state)

    def get_anticipatory_plan(self, restaurant, task):
        save_file = '/data/figs/new_demo' + str(
            restaurant.seed) + '.txt'
        init_state = copy.deepcopy(restaurant.get_current_object_state())
        plan, myopic_cost = (
            self.myopic_planner.get_cost_and_state_from_task(restaurant, task)
        )
        if plan is None:
            return init_state, 10000, task, 0
        help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
        # for p in plan:
        #     if "place" in p.name:
        #         used_items.add(p.args[1])
        #     if "wash" in p.name:
        #         used_items.add(p.args[1])
        #     if "cook" in p.name:
        #         used_items.add(p.args[1])
        #         used_items.add(p.args[2])
        #     if "serve" in p.name:
        #         used_items.add(p.args[1])
        # with open(save_file, "a+") as f:
        #     f.write(f"| Items in Plan: {used_items}\n")
        term_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(term_state)
        myopic_ex_cost = self.get_anticipated_cost(restaurant)
        myopic_ant_cost = myopic_ex_cost + myopic_cost
        with open(save_file, "a+") as f:
            f.write(f"| Initiating...... \n")
            f.write(f"| Myopic Cost: {myopic_cost}")
            f.write(f"| Myopic Exp Cost: {myopic_ex_cost}\n")
        ant_cost = myopic_cost
        ant_state = copy.deepcopy(term_state)
        conts = list()
        aug_predicates = list()
        # if restaurant.active_robot == 'tire_bot':
        #     conts = TIRE_FIXER_REACHABLES
        # else:
        #     conts = MIRROR_FIXER_REACHABLES
        conts = BATTERY_FIXER_REACHABLES
        
        for cnt in conts:
            ant_objects = restaurant.get_objects_by_container_name(cnt)
            for obj in ant_objects:
                # if obj['assetId'] in used_items:
                #     continue
                for (oth_cnt, val) in restaurant.get_container_pos_list():
                    if oth_cnt not in conts:
                        continue
                    # with open(save_file, "a+") as f:
                    #     f.write(f"| Object: {obj['assetId']}: Container: {oth_cnt}\n")
                    restaurant.update_container_props(term_state)
                    candidate_state = restaurant.place_object(obj, val)
                    restaurant.update_container_props(candidate_state)
                    can_exp_cost = self.get_anticipated_cost(restaurant)
                    if can_exp_cost < myopic_ex_cost:
                        if taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt) not in aug_predicates:
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
                    # if 'bad' in obj and obj['bad'] == 1:
                    #     candidate_state = restaurant.place_car_items(obj, val, bad=0)
                    #     restaurant.update_container_props(candidate_state)
                    #     can_exp_cost = self.get_anticipated_cost(restaurant)
                    #     if can_exp_cost < myopic_ex_cost:
                    #         if taskplan_multi.pddl.task.fix_and_place_something(obj['assetId'], oth_cnt) not in aug_predicates:
                    #             aug_predicates.append(taskplan_multi.pddl.task.fix_and_place_something(obj['assetId'], oth_cnt))
        
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
            with open(save_file, "a+") as f:
                f.write(f"Expected Cost for {ant_task_pred} \n")
            can_ex_cost = self.get_anticipated_cost(restaurant)
            can_ant_cost = can_ex_cost + c_cost
            with open(save_file, "a+") as f:
                f.write(f"| Cost: {c_cost}")
                f.write(f"| Exp Cost: {can_ex_cost}\n")
                f.write(f"--------------------------\n")
            if can_ant_cost < myopic_ant_cost:
                ant_state = copy.deepcopy(can_state)
                ant_cost = c_cost
                myopic_ant_cost = can_ant_cost
                ant_task = ant_task_pred
                help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
        return ant_state, ant_cost, ant_task, help_stat

