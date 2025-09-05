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

from taskplan_multi.utilities.restaurant_primitives import (
    KITCHEN_CONTAINERS, SERVING_ROOM_CONTAINERS,
    COOK_BOT_RESTRICT, SERVER_BOT_RESTRICT, CLEANER_BOT_RESTRICT, ASSETS)

MAX_AUG = 2
SCALE = {
    'cook_bot': 0,
    'server_bot': 0,
    'cleaner_bot': 0

}

SCL = 1

def get_combo(item1, item2):
    combinations = [list(zip(item1, p)) for p in product(item2, repeat=len(item1))]
    return combinations

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
                # if 'dirty' in obj and obj['dirty'] == 1:
                if random.random() > 0.5:
                    candidate_state = restaurant.place_washables(obj, container, dirty=0)
                else:
                    candidate_state = restaurant.place_washables(obj, container, dirty=1)
            elif 'food' in obj:
                # if 'empty' in obj and obj['empty'] == 1:
                if random.random() > 0.5:
                    candidate_state = restaurant.place_food_items(obj, container, empty=0)
                else:
                    candidate_state = restaurant.place_food_items(obj, container, empty=1)
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
        for key in self.eval_nets:
            if key not in restaurant.agent_list:
                continue
            restaurant.active_robot = key
            whole_graph = taskplan_multi.utils.get_graph(restaurant)
            c = self.eval_nets[key](whole_graph)
            if key == current_active:
                selfish_cost = c
            expc_costs.append(c)
            del whole_graph, c
            gc.collect()
        restaurant.active_robot = current_active
        avg = sum(expc_costs)/len(expc_costs)
        if self.concern == 'joint':
            return sum(expc_costs)
        if self.concern == 'self':
            return selfish_cost*SCL
        if self.concern == 'other':
            return (sum(expc_costs) - selfish_cost)
    
    def get_seq_cost(self, args, restaurant, task_seq, seq_num, no_prep_state=None, prep_state=None, ap_concern='joint'):
        if prep_state:
            restaurant.update_container_props(prep_state)
            file_name = 'prep_ap_' + ap_concern + '.txt'
            self.concern = ap_concern
            logfile = os.path.join(args.save_dir, file_name)
            for idx, item in enumerate(task_seq):
                start = time.time()
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                last_task = False
                if idx+1 == len(task_seq):
                    last_task = True
                new_state, cost, new_task, help_stat = self.get_anticipatory_plan(restaurant, task, last_task=last_task)
                end = time.time()
                elapsed = end - start
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | active: {active_agent}"
                        f" | time: {elapsed:0.2f}"
                        f" | help: {help_stat}"
                        f" | cost: {cost:0.2f} \n"
                    )
                restaurant.update_container_props(new_state)
                # exp_c = self.get_anticipated_cost(restaurant)
                # taskplan_multi.utils.plot_state(restaurant, args, image_name=f'prep-ap-{seq_num}-T{idx+1}', title=f'State after Task {idx+1} \n Exp: {exp_c}: {self.concern}')
        
        if no_prep_state:
            restaurant.update_container_props(no_prep_state)
            file_name = 'np_ap_' + ap_concern + '.txt'
            self.concern = ap_concern
            logfile = os.path.join(args.save_dir, file_name)
            # task_file_name = 'ap_myopic_tasks.txt'
            # logfile_task = os.path.join(args.save_dir, task_file_name)
            for idx, item in enumerate(task_seq):
                start = time.time()
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                new_state, cost, new_task, help_stat = self.get_anticipatory_plan(restaurant, task)
                end = time.time()
                elapsed = end - start
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | active: {active_agent}"
                        f" | time: {elapsed:0.2f}"
                        f" | help: {help_stat}"
                        f" | cost: {cost:0.2f} \n"
                    )
                # with open(logfile_task, "a+") as f:
                #     f.write(
                #         f" | num: T{idx+1}"
                #         f" | task: {new_task}"
                #         f" | help_stat: {help_stat}\n"
                #     )
                restaurant.update_container_props(new_state)
                # taskplan_multi.utils.plot_state(restaurant, args, image_name=f'np-ap-{seq_num}-T{idx+1}', title=f'State after Task {idx+1} : {self.concern}')
    

    def aug_predicates_for_cleaner(self, restaurant, used_items, used_containers, myopic_ex_cost, other_bots=set()):
        aug_predicates = list()
        current_state = copy.deepcopy(restaurant.get_current_object_state())
        conts = [c for c in KITCHEN_CONTAINERS + SERVING_ROOM_CONTAINERS if c not in CLEANER_BOT_RESTRICT]

        if 'cook_bot' in other_bots:
            conts.append('stove')

        
        prior_combos = list()
        selected_conts = set()
        selected_assets = set()
        one_list_combo = list()

        for item1 in used_containers:
            for item2 in conts:
                if restaurant.known_cost[item1][item2] < 30:
                    selected_conts.add(item2)
                    ant_objects = restaurant.get_objects_by_container_name(item2)
                    for obj in ant_objects:
                        if obj['assetId'] not in used_items:
                            selected_assets.add(obj['assetId'])

        if len(used_items) >= 3:
            sampled_conts = random.sample(list(selected_conts), 4)
            prior_combos = get_combo(list(used_items), sampled_conts)
        else:
            for idx, take_one in enumerate(selected_assets):
                temp = [take_one]
                temp.extend(list(used_items))
                max_size = min(len(selected_conts), 4)
                sampled_conts = random.sample(list(selected_conts), max_size)
                temp_combos = get_combo(temp, sampled_conts)
                prior_combos.extend(temp_combos)

        if len(used_items) == 1:
            one_list_combo = get_combo(list(used_items), conts)
        
        max_size = min(len(prior_combos), 200)
        prior_combos = random.sample(prior_combos, max_size)
        one_list_combo.extend(prior_combos)

        count = 0
        
        for item_list in one_list_combo:
            if count >= 50:
                break
            restaurant.update_container_props(current_state)
            tt = '(and'
            for (obj_name, cnt_name) in item_list:
                obj = restaurant.get_object_props_by_name(obj_name)
                val = restaurant.get_container_pos(cnt_name)
                if 'washable' in obj and 'dirty' in obj and obj['dirty'] == 1:
                    candidate_state = restaurant.place_washables(obj, val, dirty=0)
                    t1 = taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], cnt_name)
                elif 'server_bot' in other_bots and 'food' in obj and 'empty' in obj and obj['empty'] == 1:
                    candidate_state = restaurant.place_food_items(obj, val, empty=0)
                    t1 = taskplan_multi.pddl.task.stock_and_place_something(obj['assetId'], cnt_name)
                else:
                    candidate_state = restaurant.place_object(obj, val)
                    t1 = taskplan_multi.pddl.task.place_something(obj['assetId'], cnt_name)
                restaurant.update_container_props(candidate_state)
                tt += t1
            tt += ')'
            can_exp_cost = self.get_anticipated_cost(restaurant)
            if can_exp_cost < myopic_ex_cost:
                count+=1
                aug_predicates.append(tt)
        return aug_predicates


    def aug_predicates_for_cook(self, restaurant, used_items, used_containers, myopic_ex_cost, other_bots=set()):
        aug_predicates = list()
        current_state = copy.deepcopy(restaurant.get_current_object_state())
        conts = [c for c in KITCHEN_CONTAINERS + SERVING_ROOM_CONTAINERS if c not in COOK_BOT_RESTRICT]
        item_to_use = set(['pasta', 'cereal', 'oats', 'milk'])

        if len(other_bots) > 0:
            conts.append('servingtable1')
            conts.append('servingtable2')

        
        # prior_combos = list()
        # selected_conts = set()
        # selected_assets = set()
        # item_to_use = set()
        # one_list_combo = list()

        # for itm in item_to_remove:
        #     if itm in used_items:
        #         used_items.remove(itm)
        #         item_to_use.add(itm)

        # for item1 in used_containers:
        #     for item2 in conts:
        #         if restaurant.known_cost[item1][item2] < 30:
        #             selected_conts.add(item2)
        #             ant_objects = restaurant.get_objects_by_container_name(item2)
        #             for obj in ant_objects:
        #                 if obj['assetId'] not in item_to_use:
        #                     selected_assets.add(obj['assetId'])

        # if len(item_to_use) >= 3:
        #     sampled_conts = random.sample(list(selected_conts), 4)
        #     prior_combos = get_combo(list(item_to_use), sampled_conts)
        # else:
        #     for idx, take_one in enumerate(selected_assets):
        #         temp = [take_one]
        #         temp.extend(list(item_to_use))
        #         max_size = min(len(selected_conts), 4)
        #         sampled_conts = random.sample(list(selected_conts), max_size)
        #         temp_combos = get_combo(temp, sampled_conts)
        #         prior_combos.extend(temp_combos)

        # if len(item_to_use) == 1:
        #     one_list_combo = get_combo(list(item_to_use), conts)
        
        # max_size = min(len(prior_combos), 200)
        # prior_combos = random.sample(prior_combos, max_size)
        # one_list_combo.extend(prior_combos)

        selected_conts = set()
        # selected_assets = set()
        # item_to_use = set()
        one_list_combo = list()

        # for itm in used_items:
        #     # if itm in used_items:
        #     #     used_items.remove(itm)
        #     item_to_use.add(itm)

        for item1 in used_containers:
            for item2 in conts:
                if restaurant.known_cost[item1][item2] < 30:
                    selected_conts.add(item2)
                    ant_objects = restaurant.get_objects_by_container_name(item2)
                    for obj in ant_objects:
                        if obj['assetId'] not in used_items:
                            item_to_use.add(obj['assetId'])

        # if len(item_to_use) >= 3:
        #     sampled_conts = random.sample(list(selected_conts), 4)
        #     prior_combos = get_combo(list(item_to_use), sampled_conts)
        # else:
        #     for idx, take_one in enumerate(selected_assets):
        #         temp = [take_one]
        #         temp.extend(list(item_to_use))
        #         max_size = min(len(selected_conts), 4)
        #         sampled_conts = random.sample(list(selected_conts), max_size)
        #         temp_combos = get_combo(temp, sampled_conts)
        #         prior_combos.extend(temp_combos)

        # if len(item_to_use) == 1:
        #     one_list_combo = get_combo(list(item_to_use), conts)
        # print(item_to_use)
        # print(conts)
        sampled_conts = random.sample(list(selected_conts), min(len(selected_conts), 4))
        for it in item_to_use:
            temp_combos = get_combo([it], sampled_conts)
            one_list_combo.extend(temp_combos)

        # one_list_combo = list(product(list(item_to_use), conts))
        
        max_size = min(len(one_list_combo), 200)
        # prior_combos = random.sample(prior_combos, max_size)
        # one_list_combo.extend(prior_combos)
        one_list_combo = random.sample(one_list_combo, max_size)

        count = 0
        
        for item_list in one_list_combo:
            if count >= 50:
                break
            restaurant.update_container_props(current_state)
            tt = '(and'
            for (obj_name, cnt_name) in item_list:
                obj = restaurant.get_object_props_by_name(obj_name)
                val = restaurant.get_container_pos(cnt_name)
                if 'cleaner_bot' in other_bots and 'washable' in obj and 'dirty' in obj and obj['dirty'] == 1:
                    candidate_state = restaurant.place_washables(obj, val, dirty=0)
                    t1 = taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], cnt_name)
                elif 'server_bot' in other_bots and 'food' in obj and 'empty' in obj and obj['empty'] == 1:
                    candidate_state = restaurant.place_food_items(obj, val, empty=0)
                    t1 = taskplan_multi.pddl.task.stock_and_place_something(obj['assetId'], cnt_name)
                else:
                    candidate_state = restaurant.place_object(obj, val)
                    t1 = taskplan_multi.pddl.task.place_something(obj['assetId'], cnt_name)
                restaurant.update_container_props(candidate_state)
                tt += t1
            tt += ')'
            can_exp_cost = self.get_anticipated_cost(restaurant)
            if can_exp_cost < myopic_ex_cost:
                count+=1
                aug_predicates.append(tt)
        return aug_predicates




    def aug_predicates_for_server(self, restaurant, used_items, used_containers, myopic_ex_cost, other_bots=set()):
        aug_predicates = list()
        current_state = copy.deepcopy(restaurant.get_current_object_state())
        conts = [c for c in KITCHEN_CONTAINERS + SERVING_ROOM_CONTAINERS if c not in SERVER_BOT_RESTRICT]
        item_to_use = set(['pasta', 'cereal', 'oats', 'milk'])

        if 'cook_bot' in other_bots:
            conts.append('stove')

        
        # prior_combos = list()
        selected_conts = set()
        # selected_assets = set()
        # item_to_use = set()
        one_list_combo = list()

        # for itm in used_items:
        #     # if itm in used_items:
        #     #     used_items.remove(itm)
        #     item_to_use.add(itm)

        for item1 in used_containers:
            for item2 in conts:
                if restaurant.known_cost[item1][item2] < 30:
                    selected_conts.add(item2)
                    ant_objects = restaurant.get_objects_by_container_name(item2)
                    for obj in ant_objects:
                        if obj['assetId'] not in used_items:
                            item_to_use.add(obj['assetId'])

        # if len(item_to_use) >= 3:
        #     sampled_conts = random.sample(list(selected_conts), 4)
        #     prior_combos = get_combo(list(item_to_use), sampled_conts)
        # else:
        #     for idx, take_one in enumerate(selected_assets):
        #         temp = [take_one]
        #         temp.extend(list(item_to_use))
        #         max_size = min(len(selected_conts), 4)
        #         sampled_conts = random.sample(list(selected_conts), max_size)
        #         temp_combos = get_combo(temp, sampled_conts)
        #         prior_combos.extend(temp_combos)

        # if len(item_to_use) == 1:
        #     one_list_combo = get_combo(list(item_to_use), conts)
        # print(item_to_use)
        # print(conts)
        sampled_conts = random.sample(list(selected_conts), min(len(selected_conts), 4))
        for it in item_to_use:
            temp_combos = get_combo([it], sampled_conts)
            one_list_combo.extend(temp_combos)

        # one_list_combo = list(product(list(item_to_use), conts))
        
        max_size = min(len(one_list_combo), 200)
        # prior_combos = random.sample(prior_combos, max_size)
        # one_list_combo.extend(prior_combos)
        one_list_combo = random.sample(one_list_combo, max_size)

        count = 0
        
        for item_list in one_list_combo:
            # print(f'{item_list} \n')
            if count >= 50:
                break
            restaurant.update_container_props(current_state)
            tt = '(and'
            for (obj_name, cnt_name) in item_list:
                obj = restaurant.get_object_props_by_name(obj_name)
                val = restaurant.get_container_pos(cnt_name)
                if 'food' in obj and 'empty' in obj and obj['empty'] == 1:
                    candidate_state = restaurant.place_food_items(obj, val, empty=0)
                    t1 = taskplan_multi.pddl.task.stock_and_place_something(obj['assetId'], cnt_name)
                elif 'cleaner_bot' in other_bots and 'washable' in obj and 'dirty' in obj and obj['dirty'] == 1:
                    candidate_state = restaurant.place_washables(obj, val, dirty=0)
                    t1 = taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], cnt_name)
                else:
                    candidate_state = restaurant.place_object(obj, val)
                    t1 = taskplan_multi.pddl.task.place_something(obj['assetId'], cnt_name)
                restaurant.update_container_props(candidate_state)
                tt += t1
            tt += ')'
            can_exp_cost = self.get_anticipated_cost(restaurant)
            if can_exp_cost < myopic_ex_cost:
                count+=1
                aug_predicates.append(tt)
        # raise NotImplementedError
        return aug_predicates

    def get_anticipatory_plan(self, restaurant, task, last_task=False, plan_only=False):
        # save_file = '/data/figs/ap_time_analysis' + str(restaurant.seed) + '.txt'
        init_state = copy.deepcopy(restaurant.get_current_object_state())
        plan, ant_cost = (
            self.myopic_planner.get_cost_and_state_from_task(restaurant, task)
        )
        if plan is None:
            if plan_only:
                return plan, ant_cost
            return init_state, 2000, task, 3
        
        myopic_help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
        ant_state = copy.deepcopy(restaurant.get_final_state_from_plan(plan))
        restaurant.update_container_props(ant_state)
        
        if last_task:
            if plan_only:
                return plan, ant_cost
            return ant_state, ant_cost, task, myopic_help_stat
        
        ant_task = task
        used_items = set()
        used_containers = set()
        other_bots = set()
        for p in plan:
            if "pick" in p.name:
                used_containers.add(p.args[2])
            if "place" in p.name:
                used_items.add(p.args[1])
                used_containers.add(p.args[2])
            if "wash" in p.name:
                used_items.add(p.args[1])
            if "mix" in p.name:
                used_items.add(p.args[1])
                used_items.add(p.args[2])
            if "serve" in p.name:
                used_items.add(p.args[1])
                used_items.add(p.args[2])
            if "restock" in p.name:
                used_items.add(p.args[1])
            if "ask-help" in p.name:
                other_bots.add(p.args[0])

        
        
        myopic_ex_cost = self.get_anticipated_cost(restaurant)
        myopic_ant_cost = myopic_ex_cost + ant_cost
        
        # Get the augmented predicates
        if restaurant.active_robot == 'server_bot':
            aug_predicates = self.aug_predicates_for_server(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)
        elif restaurant.active_robot == 'cook_bot':
            aug_predicates = self.aug_predicates_for_cook(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)
        else:
            aug_predicates = self.aug_predicates_for_cleaner(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)

        # with open(save_file, "a+") as f:
        #     f.write(f"---Myopic: {ant_task}: {ant_cost} {myopic_ex_cost}---\n")
        # with open(save_file, "a+") as f:
        #     f.write(f"--No of Predicates: {len(aug_predicates)}\n")
        exhausted = 0
        # random.shuffle(aug_predicates)
        for aug_pred in aug_predicates:
            if exhausted >= 2:
                break
            start_time = time.time()
            restaurant.update_container_props(init_state)
            ant_task_pred = f'(and {aug_pred} {task})'
            # with open(save_file, "a+") as f:
            #     f.write(f"---Task: {ant_task_pred}---\n")
            c_plan, c_cost = (
                self.myopic_planner.get_cost_and_state_from_task(
                    restaurant, ant_task_pred))
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time >= 100:
                exhausted+=1
            # with open(save_file, "a+") as f:
            #     f.write(f"Process took {elapsed_time:.2f} seconds to finish.\n")
            if c_plan is None:
                continue
            can_help_stat = taskplan_multi.utils.get_status_of_asking_help(c_plan)
            if can_help_stat > myopic_help_stat:
                continue
            can_state = restaurant.get_final_state_from_plan(c_plan)
            restaurant.update_container_props(can_state)
            can_ex_cost = self.get_anticipated_cost(restaurant)
            # with open(save_file, "a+") as f:
            #     f.write(f"---Task Cost: {can_ex_cost} + {c_cost} ---\n")
            can_ant_cost = can_ex_cost + c_cost
            if can_ant_cost < myopic_ant_cost:
                # found = 1
                # with open(save_file, "a+") as f:
                #     f.write(f"---Task: {aug_pred}: {c_cost} = {can_ex_cost}---\n")
                ant_state = copy.deepcopy(can_state)
                ant_cost = c_cost
                plan = c_plan
                myopic_ant_cost = can_ant_cost
                ant_task = ant_task_pred
                myopic_help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
        
        if plan_only:
            return plan, ant_cost

        # with open(save_file, "a+") as f:
        #     f.write(f"---Task: {can_ant_cost}: {c_cost} and {can_ex_cost}---\n")
        # raise NotImplementedError
        
        return ant_state, ant_cost, ant_task, myopic_help_stat

 
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
    
    def get_prepared_state(self, restaurant, n_iterations=1000):
        def safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                # If the argument is too large in magnitude, return an approximation
                return 0.0
        # save_file = '/data/figs/learned_prep' + str(
        #     restaurant.seed) + '.txt'

        prepared_state = copy.deepcopy(restaurant.get_current_object_state())
        int_cost = self.get_anticipated_cost(restaurant)
        # with open(save_file, "a+") as f:
        #     f.write(
        #         f"| Initial State"
        #         f"| Expected Cost: {int_cost}\n")
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
                # with open(save_file, "a+") as f:
                #     f.write(
                #         f"| idx: {i}"
                #         # f"| task: {task_to_solve}"
                #         # f"| exp: {exp}"
                #         # f"| delta: {delta}"
                #         f"| exp cost: {int_cost}\n")
            temp = max(temp * cooling_rate, 1)
        # with open(save_file, "a+") as f:
        #     f.write(
        #         f"| found after: {i}"
        #         f"| prepared e_cost: {int_cost}\n"
        #         f"| state: {prepared_state}\n")
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