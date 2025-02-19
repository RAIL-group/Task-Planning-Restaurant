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
        expc_costs = list()
        # save_file = '/data/figs/ap_debug' + str(
        #     restaurant.seed) + '.txt'
        for key in self.eval_nets:
            restaurant.active_robot = key
            whole_graph = taskplan_multi.utils.get_graph(restaurant)
            c = self.eval_nets[key](whole_graph)
            # with open(save_file, "a+") as f:
            #     f.write(f"| {key}: {c}")
            if key == current_active:
                selfish_cost = c
            expc_costs.append(c)
            del whole_graph, c
            gc.collect()
        restaurant.active_robot = current_active
        avg = sum(expc_costs)/len(expc_costs)
        # with open(save_file, "a+") as f:
        #     f.write(f"| avg: {avg} \n")
        if self.concern == 'joint':
            return sum(expc_costs)
        if self.concern == 'self':
            return selfish_cost
        if self.concern == 'other':
            return (sum(expc_costs) - selfish_cost)



    
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
        init_state = copy.deepcopy(restaurant.get_current_object_state())
        plan, myopic_cost = (
            self.myopic_planner.get_cost_and_state_from_task(restaurant, task)
        )
        if plan is None:
            return init_state, 10000, task, 0
        # if cost == 0:
        #     return init_state, cost, task
        used_items = set()
        save_file = '/data/figs/ap_debug' + str(
            restaurant.seed) + '.txt'
        # if restaurant.active_robot == 'server_bot':
        help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
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
        # with open(save_file, "a+") as f:
        #     f.write(f"| Items in Plan: {used_items}\n")
        term_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(term_state)
        myopic_ex_cost = self.get_anticipated_cost(restaurant)
        myopic_ant_cost = myopic_ex_cost + myopic_cost
        # with open(save_file, "a+") as f:
        #     f.write(f"| Myopic Cost: {myopic_cost}")
        #     f.write(f"| Myopic Exp Cost: {myopic_ex_cost}\n")
        ant_cost = myopic_cost
        ant_state = copy.deepcopy(term_state)
        conts = list()
        aug_predicates = list()
        if restaurant.active_all:
            objs = restaurant.get_current_object_state()
            for obj in objs:
                if obj['assetId'] in used_items:
                        continue
                for (cnt, val) in restaurant.get_container_pos_list():
                    # with open(save_file, "a+") as f:
                    #     f.write(f"| Object: {obj['assetId']}: Container: {cnt}\n")
                    if 'washable' in obj:
                        if 'dirty' in obj and obj['dirty'] == 1:
                            candidate_state = restaurant.place_washables(obj, val, dirty=0)
                            # with open(save_file, "a+") as f:
                            #     f.write(f"Cleaned \n")
                            restaurant.update_container_props(candidate_state)
                            can_exp_cost = self.get_anticipated_cost(restaurant)
                            if can_exp_cost < myopic_ex_cost:
                                aug_predicates.append(taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], cnt))
                            candidate_state = restaurant.place_object(obj, val)
                            restaurant.update_container_props(candidate_state)
                            # with open(save_file, "a+") as f:
                            #     f.write(f"Without Cleaning, Just Moved \n")
                            can_exp_cost = self.get_anticipated_cost(restaurant)
                            if can_exp_cost < myopic_ex_cost:
                                aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], cnt))
                        else:
                            # with open(save_file, "a+") as f:
                            #     f.write(f"Cleaned Item, Just Moved \n")
                            candidate_state = restaurant.place_object(obj, val)
                            restaurant.update_container_props(candidate_state)
                            can_exp_cost = self.get_anticipated_cost(restaurant)
                            if can_exp_cost < myopic_ex_cost:
                                aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], cnt))
                    elif 'cookable' in obj:
                        if 'cooked' in obj and obj['cooked'] == 1:
                            # with open(save_file, "a+") as f:
                            #     f.write(f"Cooked Item Moved \n")
                            candidate_state = restaurant.place_object(obj, val)
                            restaurant.update_container_props(candidate_state)
                            can_exp_cost = self.get_anticipated_cost(restaurant)
                            if can_exp_cost < myopic_ex_cost:
                                aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], cnt))
                        else:
                            # with open(save_file, "a+") as f:
                            #     f.write(f"Uncooked Item, Cooked & Moved \n")
                            candidate_state = restaurant.place_food_items(obj, val, cooked=1)
                            restaurant.update_container_props(candidate_state)
                            can_exp_cost = self.get_anticipated_cost(restaurant)
                            if can_exp_cost < myopic_ex_cost:
                                aug_predicates.append(taskplan_multi.pddl.task.cook_and_place_something(obj['assetId'], cnt))
                            candidate_state = restaurant.place_object(obj, val)
                            restaurant.update_container_props(candidate_state)
                            # with open(save_file, "a+") as f:
                            #     f.write(f"Uncooked Item,  Just Moved \n")
                            can_exp_cost = self.get_anticipated_cost(restaurant)
                            if can_exp_cost < myopic_ex_cost:
                                aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], cnt))
                    else:
                        # with open(save_file, "a+") as f:
                        #     f.write(f"Regular Item,  Just Moved \n")
                        candidate_state = restaurant.place_object(obj, val)
                        restaurant.update_container_props(candidate_state)
                        can_exp_cost = self.get_anticipated_cost(restaurant)
                        if can_exp_cost < myopic_ex_cost:
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], cnt))
        else:
            if restaurant.active_robot == 'cook_bot':
                conts = COOK_BOT_REACHABLES
                for cnt in conts:
                    ant_objects = restaurant.get_objects_by_container_name(cnt)
                    for obj in ant_objects:
                        if obj['assetId'] in used_items:
                            continue
                        for (oth_cnt, val) in restaurant.get_container_pos_list():
                            if oth_cnt not in conts:
                                continue
                            # with open(save_file, "a+") as f:
                            #     f.write(f"| Object: {obj['assetId']}: Container: {oth_cnt}\n")
                            if 'cookable' in obj:
                                if 'cooked' in obj and obj['cooked'] == 1:
                                    # with open(save_file, "a+") as f:
                                    #     f.write(f"Cooked Item Moved \n")
                                    candidate_state = restaurant.place_object(obj, val)
                                    restaurant.update_container_props(candidate_state)
                                    can_exp_cost = self.get_anticipated_cost(restaurant)
                                    if can_exp_cost < myopic_ex_cost:
                                        aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
                            else:
                                # with open(save_file, "a+") as f:
                                #     f.write(f"Uncooked Item, Cooked & Moved \n")
                                candidate_state = restaurant.place_food_items(obj, val, cooked=1)
                                restaurant.update_container_props(candidate_state)
                                can_exp_cost = self.get_anticipated_cost(restaurant)
                                if can_exp_cost < myopic_ex_cost:
                                    aug_predicates.append(taskplan_multi.pddl.task.cook_and_place_something(obj['assetId'], oth_cnt))
                                candidate_state = restaurant.place_object(obj, val)
                                restaurant.update_container_props(candidate_state)
                                # with open(save_file, "a+") as f:
                                #     f.write(f"Uncooked Item,  Just Moved \n")
                                can_exp_cost = self.get_anticipated_cost(restaurant)
                                if can_exp_cost < myopic_ex_cost:
                                    aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
            elif restaurant.active_robot == 'server_bot':
                conts = SERVER_BOT_REACHABLES
                for cnt in conts:
                    ant_objects = restaurant.get_objects_by_container_name(cnt)
                    for obj in ant_objects:
                        if obj['assetId'] in used_items:
                            continue
                        for (oth_cnt, val) in restaurant.get_container_pos_list():
                            if oth_cnt not in conts:
                                continue
                            # with open(save_file, "a+") as f:
                            #     f.write(f"| Object: {obj['assetId']}: Container: {oth_cnt} : ")
                            candidate_state = restaurant.place_object(obj, val)
                            restaurant.update_container_props(candidate_state)
                            can_exp_cost = self.get_anticipated_cost(restaurant)
                            if can_exp_cost < myopic_ex_cost:
                                aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
            else:
                conts = CLEANER_BOT_REACHABLES
                for cnt in conts:
                    ant_objects = restaurant.get_objects_by_container_name(cnt)
                    for obj in ant_objects:
                        if obj['assetId'] in used_items:
                            continue
                        for (oth_cnt, val) in restaurant.get_container_pos_list():
                            if oth_cnt not in conts:
                                continue
                            # with open(save_file, "a+") as f:
                            #     f.write(f"| Object: {obj['assetId']}: Container: {oth_cnt}\n")
                            if 'washable' in obj:
                                if 'dirty' in obj and obj['dirty'] == 1:
                                    candidate_state = restaurant.place_washables(obj, val, dirty=0)
                                    # with open(save_file, "a+") as f:
                                    #     f.write(f"Cleaned \n")
                                    restaurant.update_container_props(candidate_state)
                                    can_exp_cost = self.get_anticipated_cost(restaurant)
                                    if can_exp_cost < myopic_ex_cost:
                                        aug_predicates.append(taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], oth_cnt))
                                    candidate_state = restaurant.place_object(obj, val)
                                    restaurant.update_container_props(candidate_state)
                                    # with open(save_file, "a+") as f:
                                    #     f.write(f"Without Cleaning, Just Moved \n")
                                    can_exp_cost = self.get_anticipated_cost(restaurant)
                                    if can_exp_cost < myopic_ex_cost:
                                        aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
                                else:
                                    # with open(save_file, "a+") as f:
                                    #     f.write(f"Cleaned Item, Just Moved \n")
                                    candidate_state = restaurant.place_object(obj, val)
                                    restaurant.update_container_props(candidate_state)
                                    can_exp_cost = self.get_anticipated_cost(restaurant)
                                    if can_exp_cost < myopic_ex_cost:
                                        aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
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
            # with open(save_file, "a+") as f:
            #     f.write(f"Expected Cost for {aug_pred} \n")
            can_ex_cost = self.get_anticipated_cost(restaurant)
            can_ant_cost = can_ex_cost + c_cost
            # with open(save_file, "a+") as f:
            #     f.write(f"| Cost: {c_cost}")
            #     f.write(f"| Exp Cost: {can_ex_cost}\n")
            if can_ant_cost < myopic_ant_cost:
                ant_state = copy.deepcopy(can_state)
                ant_cost = c_cost
                myopic_ant_cost = can_ant_cost
                ant_task = ant_task_pred
                help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
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
        return ant_state, ant_cost, ant_task, help_stat


    # def get_anticipatory_plan(self, restaurant, task):
        init_state = copy.deepcopy(restaurant.get_current_object_state())
        plan, myopic_cost = (
            self.myopic_planner.get_cost_and_state_from_task(restaurant, task)
        )
        if plan is None:
            return init_state, 10000, task, 0
        # if cost == 0:
        #     return init_state, cost, task
        used_items = set()
        save_file = '/data/figs/ap_debug' + str(
            restaurant.seed) + '.txt'
        # if restaurant.active_robot == 'server_bot':
        help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
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
        # with open(save_file, "a+") as f:
        #     f.write(f"| Items in Plan: {used_items}\n")
        term_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(term_state)
        myopic_ex_cost = self.get_anticipated_cost(restaurant)
        myopic_ant_cost = myopic_ex_cost + myopic_cost
        # with open(save_file, "a+") as f:
        #     f.write(f"| Myopic Cost: {myopic_cost}")
        #     f.write(f"| Myopic Exp Cost: {myopic_ex_cost}\n")
        ant_cost = myopic_cost
        ant_state = copy.deepcopy(term_state)
        conts = list()
        aug_predicates = list()
        ant_objects = restaurant.get_current_object_state()
        # conts = [c for (c, v) in restaurant.get_container_pos_list()]
        # for cnt in conts:
        #     ant_objects = restaurant.get_objects_by_container_name(cnt)
        for obj in ant_objects:
            if obj['assetId'] in used_items:
                continue
            for (oth_cnt, val) in restaurant.get_container_pos_list():
                if 'washable' in obj:
                    if 'dirty' in obj and obj['dirty'] == 1:
                        candidate_state = restaurant.place_washables(obj, val, dirty=0)
                        restaurant.update_container_props(candidate_state)
                        can_exp_cost = self.get_anticipated_cost(restaurant)
                        if can_exp_cost < myopic_ex_cost:
                            aug_predicates.append(taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], oth_cnt))
                        candidate_state = restaurant.place_object(obj, val)
                        restaurant.update_container_props(candidate_state)
                        can_exp_cost = self.get_anticipated_cost(restaurant)
                        if can_exp_cost < myopic_ex_cost:
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
                    else:
                        candidate_state = restaurant.place_object(obj, val)
                        restaurant.update_container_props(candidate_state)
                        can_exp_cost = self.get_anticipated_cost(restaurant)
                        if can_exp_cost < myopic_ex_cost:
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
                elif 'cookable' in obj:
                    if 'cooked' in obj and obj['cooked'] == 1:
                        candidate_state = restaurant.place_object(obj, val)
                        restaurant.update_container_props(candidate_state)
                        can_exp_cost = self.get_anticipated_cost(restaurant)
                        if can_exp_cost < myopic_ex_cost:
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
                    else:
                        candidate_state = restaurant.place_food_items(obj, val, cooked=1)
                        restaurant.update_container_props(candidate_state)
                        can_exp_cost = self.get_anticipated_cost(restaurant)
                        if can_exp_cost < myopic_ex_cost:
                            aug_predicates.append(taskplan_multi.pddl.task.cook_and_place_something(obj['assetId'], oth_cnt))
                        candidate_state = restaurant.place_object(obj, val)
                        restaurant.update_container_props(candidate_state)
                        can_exp_cost = self.get_anticipated_cost(restaurant)
                        if can_exp_cost < myopic_ex_cost:
                            aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
                else:
                    candidate_state = restaurant.place_object(obj, val)
                    restaurant.update_container_props(candidate_state)
                    can_exp_cost = self.get_anticipated_cost(restaurant)
                    if can_exp_cost < myopic_ex_cost:
                        aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
                
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
            # with open(save_file, "a+") as f:
            #     f.write(f"Expected Cost for {aug_pred} \n")
            can_ex_cost = self.get_anticipated_cost(restaurant)
            can_ant_cost = can_ex_cost + c_cost
            # with open(save_file, "a+") as f:
            #     f.write(f"| Cost: {c_cost}")
            #     f.write(f"| Exp Cost: {can_ex_cost}\n")
            if can_ant_cost < myopic_ant_cost:
                ant_state = copy.deepcopy(can_state)
                ant_cost = c_cost
                myopic_ant_cost = can_ant_cost
                ant_task = ant_task_pred
                help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
        return ant_state, ant_cost, ant_task, help_stat
 
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
        temp = 100
        cooling_rate = 0.95
        # for idx, item in enumerate(task_seq):
        #     active_agent = item[0]
        #     task = item[1]
        #     restaurant.active_robot = active_agent
        #     new_state, cost, new_task = self.get_anticipatory_plan(restaurant, task)
        #     costs.append(cost)
        restaurant.active_all = True
        while (i < n_iterations):
            restaurant.update_container_props(prepared_state)
            i += 1
            # item = random.choice(task_sequence)
            # active_agent = item[0]
            # task_to_solve = item[1]
            # restaurant.active_robot = active_agent
            # plan, cost = (
            #     self.myopic_planner.get_cost_and_state_from_task(
            #         restaurant, task_to_solve)
            # )
            # candidate_state = proc_data.randomize_objects()
            # if plan is None:
            #     continue

            # candidate_state = restaurant.get_final_state_from_plan(plan)
            candidate_state = change_state(restaurant)
            restaurant.update_container_props(candidate_state)
            can_exp_cost = self.get_anticipated_cost(restaurant)
            delta = (can_exp_cost-int_cost)
            exp = safe_exp(-delta/temp)
            if delta < 0 or random.uniform(0, 1) < exp:
                prepared_state = copy.deepcopy(candidate_state)
                int_cost = can_exp_cost
                with open(save_file, "a+") as f:
                    f.write(
                        f"| idx: {i}"
                        # f"| task: {task_to_solve}"
                        # f"| exp: {exp}"
                        # f"| delta: {delta}"
                        f"| exp cost: {int_cost}\n")
            temp = max(temp * cooling_rate, 1)
        with open(save_file, "a+") as f:
            f.write(
                f"| found after: {i}"
                f"| prepared e_cost: {int_cost}\n"
                f"| state: {prepared_state}\n")
        return prepared_state

# def change_state(proc_data):
#     state = None
#     while (state is None):
#         obj = random.choice(taskplan.environments.sampling.load_movables())
#         cont_list = [v for (c, v) in proc_data.get_container_pos_list()]
#         container = random.choice(cont_list)
#         if 'jar' in obj:
#             state = proc_data.fill_up_jar_n_place(obj, container)
#         elif 'washable' in obj and 'dirty' in obj:
#             state = proc_data.place_n__clean_object(obj, container)
#         else:
#             state = proc_data.place_object(obj, container)
#     return state