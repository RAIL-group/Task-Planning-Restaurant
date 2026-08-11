import taskplan_multi
import torch
import copy
import random
import math
from itertools import product
import gc
import os
import time

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
    while candidate_state is None:
        objs = restaurant.get_current_object_state()
        obj = random.choice(objs)
        cont_list = [v for (c, v) in restaurant.get_container_pos_list()]
        container = random.choice(cont_list)
        if random.random() > 0.5:
            candidate_state = restaurant.place_object(obj, container)
        else:
            if 'washable' in obj:
                if random.random() > 0.5:
                    candidate_state = restaurant.place_washables(obj, container, dirty=0)
                else:
                    candidate_state = restaurant.place_washables(obj, container, dirty=1)
            elif 'food' in obj:
                if random.random() > 0.5:
                    candidate_state = restaurant.place_food_items(obj, container, empty=0)
                else:
                    candidate_state = restaurant.place_food_items(obj, container, empty=1)
            else:
                candidate_state = restaurant.place_object(obj, container)
    return candidate_state


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
        self.myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner(self.domain)
        self.concern = 'joint'

    def get_anticipated_cost(self, restaurant):
        current_active = restaurant.active_robot
        expc_costs = []
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
        if self.concern == 'joint':
            return sum(expc_costs)
        if self.concern == 'self':
            return selfish_cost * SCL
        if self.concern == 'other':
            return sum(expc_costs) - selfish_cost

    def _extract_plan_context(self, plan):
        used_items = set()
        used_containers = set()
        other_bots = set()
        for p in plan:
            if 'pick' in p.name:
                used_containers.add(p.args[2])
            if 'place' in p.name:
                used_items.add(p.args[1])
                used_containers.add(p.args[2])
            if 'wash' in p.name:
                used_items.add(p.args[1])
            if 'mix' in p.name:
                used_items.add(p.args[1])
                used_items.add(p.args[2])
            if 'serve' in p.name:
                used_items.add(p.args[1])
                used_items.add(p.args[2])
            if 'restock' in p.name:
                used_items.add(p.args[1])
            if 'ask-help' in p.name:
                other_bots.add(p.args[0])
        return used_items, used_containers, other_bots

    def aug_predicates_for_cleaner(self, restaurant, used_items, used_containers, myopic_ex_cost, other_bots=set()):
        aug_predicates = []
        current_state = copy.deepcopy(restaurant.get_current_object_state())
        conts = [c for c in KITCHEN_CONTAINERS + SERVING_ROOM_CONTAINERS if c not in CLEANER_BOT_RESTRICT]
        if 'cook_bot' in other_bots:
            conts.append('stove')

        selected_conts = set()
        selected_assets = set()
        one_list_combo = []

        for item1 in used_containers:
            for item2 in conts:
                if restaurant.known_cost[item1][item2] < 30:
                    selected_conts.add(item2)
                    for obj in restaurant.get_objects_by_container_name(item2):
                        if obj['assetId'] not in used_items:
                            selected_assets.add(obj['assetId'])

        prior_combos = []
        if len(used_items) >= 3:
            sampled_conts = random.sample(list(selected_conts), 4)
            prior_combos = get_combo(list(used_items), sampled_conts)
        else:
            for take_one in selected_assets:
                temp = [take_one] + list(used_items)
                sampled_conts = random.sample(list(selected_conts), min(len(selected_conts), 4))
                prior_combos.extend(get_combo(temp, sampled_conts))

        if len(used_items) == 1:
            one_list_combo = get_combo(list(used_items), conts)

        prior_combos = random.sample(prior_combos, min(len(prior_combos), 200))
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
                count += 1
                aug_predicates.append(tt)
        return aug_predicates

    def _aug_predicates_for_food_robot(self, robot, restaurant, used_items, used_containers, myopic_ex_cost, other_bots=set()):
        """Shared augmentation logic for cook_bot and server_bot."""
        aug_predicates = []
        current_state = copy.deepcopy(restaurant.get_current_object_state())

        restrict = COOK_BOT_RESTRICT if robot == 'cook_bot' else SERVER_BOT_RESTRICT
        conts = [c for c in KITCHEN_CONTAINERS + SERVING_ROOM_CONTAINERS if c not in restrict]

        if robot == 'cook_bot' and len(other_bots) > 0:
            conts += ['servingtable1', 'servingtable2']
        if robot == 'server_bot' and 'cook_bot' in other_bots:
            conts.append('stove')

        item_to_use = set(['pasta', 'cereal', 'oats', 'milk'])
        item_to_use.update(used_items)

        selected_conts = set()
        for item1 in used_containers:
            for item2 in conts:
                if restaurant.known_cost[item1][item2] < 30:
                    selected_conts.add(item2)
                    for obj in restaurant.get_objects_by_container_name(item2):
                        item_to_use.add(obj['assetId'])

        sampled_conts = random.sample(list(selected_conts), min(len(selected_conts), 4))
        one_list_combo = []
        for it in item_to_use:
            one_list_combo.extend(get_combo([it], sampled_conts))
        one_list_combo = random.sample(one_list_combo, min(len(one_list_combo), 200))

        count = 0
        for item_list in one_list_combo:
            if count >= 50:
                break
            restaurant.update_container_props(current_state)
            tt = '(and'
            for (obj_name, cnt_name) in item_list:
                obj = restaurant.get_object_props_by_name(obj_name)
                val = restaurant.get_container_pos(cnt_name)
                if robot == 'cook_bot':
                    if 'cleaner_bot' in other_bots and 'washable' in obj and obj.get('dirty') == 1:
                        candidate_state = restaurant.place_washables(obj, val, dirty=0)
                        t1 = taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], cnt_name)
                    elif 'server_bot' in other_bots and 'food' in obj and obj.get('empty') == 1:
                        candidate_state = restaurant.place_food_items(obj, val, empty=0)
                        t1 = taskplan_multi.pddl.task.stock_and_place_something(obj['assetId'], cnt_name)
                    else:
                        candidate_state = restaurant.place_object(obj, val)
                        t1 = taskplan_multi.pddl.task.place_something(obj['assetId'], cnt_name)
                else:  # server_bot
                    if 'food' in obj and obj.get('empty') == 1:
                        candidate_state = restaurant.place_food_items(obj, val, empty=0)
                        t1 = taskplan_multi.pddl.task.stock_and_place_something(obj['assetId'], cnt_name)
                    elif 'cleaner_bot' in other_bots and 'washable' in obj and obj.get('dirty') == 1:
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
                count += 1
                aug_predicates.append(tt)
        return aug_predicates

    def _aug_predicates_for_any_robot(self, robot, restaurant, myopic_ex_cost, other_bots=set()):
        """
        Generate augmentation predicates by enumerating all feasible single-fluent changes:
        every accessible item × every accessible location × every valid state variant
        (clean/dirty for washables, full/empty for food, plain place for others).
        Requires only the robot identity and current restaurant state — no myopic plan context.
        Robot location restrictions are respected. The anticipated cost oracle filters
        down to at most 50 candidates that actually reduce expected future cost.
        """
        current_state = copy.deepcopy(restaurant.get_current_object_state())

        if robot == 'cook_bot':
            restrict = COOK_BOT_RESTRICT
        elif robot == 'server_bot':
            restrict = SERVER_BOT_RESTRICT
        else:
            restrict = CLEANER_BOT_RESTRICT

        accessible_conts = [c for c in KITCHEN_CONTAINERS + SERVING_ROOM_CONTAINERS if c not in restrict]
        if robot == 'cook_bot' and len(other_bots) > 0:
            accessible_conts += ['servingtable1', 'servingtable2']
        if robot in ('server_bot', 'cleaner_bot') and 'cook_bot' in other_bots:
            accessible_conts.append('stove')

        # All items currently in any accessible container
        all_items = set()
        for cont in accessible_conts:
            for obj in restaurant.get_objects_by_container_name(cont):
                all_items.add(obj['assetId'])

        # Build every (pddl_predicate_str, candidate_state) pair up front
        candidates = []
        for item_name in all_items:
            restaurant.update_container_props(current_state)
            obj = restaurant.get_object_props_by_name(item_name)
            if obj is None:
                continue
            for cnt_name in accessible_conts:
                val = restaurant.get_container_pos(cnt_name)
                if 'washable' in obj:
                    # clean variant
                    s = restaurant.place_washables(obj, val, dirty=0)
                    p = taskplan_multi.pddl.task.clean_and_place_something(item_name, cnt_name)
                    candidates.append((f'(and {p})', s))
                    # dirty variant
                    s = restaurant.place_washables(obj, val, dirty=1)
                    p = taskplan_multi.pddl.task.place_something(item_name, cnt_name)
                    candidates.append((f'(and {p})', s))
                elif 'food' in obj:
                    # stocked variant
                    s = restaurant.place_food_items(obj, val, empty=0)
                    p = taskplan_multi.pddl.task.stock_and_place_something(item_name, cnt_name)
                    candidates.append((f'(and {p})', s))
                    # empty variant
                    s = restaurant.place_food_items(obj, val, empty=1)
                    p = taskplan_multi.pddl.task.place_something(item_name, cnt_name)
                    candidates.append((f'(and {p})', s))
                else:
                    s = restaurant.place_object(obj, val)
                    p = taskplan_multi.pddl.task.place_something(item_name, cnt_name)
                    candidates.append((f'(and {p})', s))

        restaurant.update_container_props(current_state)
        candidates = random.sample(candidates, min(len(candidates), 200))

        aug_predicates = []
        count = 0
        for pred_str, candidate_state in candidates:
            if count >= 50:
                break
            restaurant.update_container_props(candidate_state)
            can_exp_cost = self.get_anticipated_cost(restaurant)
            restaurant.update_container_props(current_state)
            if can_exp_cost < myopic_ex_cost:
                count += 1
                aug_predicates.append(pred_str)

        return aug_predicates

    def get_seq_cost(self, args, restaurant, task_seq, seq_num, no_prep_state=None, prep_state=None, ap_concern='joint'):
        init_state = prep_state if prep_state is not None else no_prep_state
        if init_state is None:
            return
        prefix = 'prep' if prep_state is not None else 'np'
        logfile = os.path.join(args.save_dir, f'{prefix}_ap_{ap_concern}.txt')
        restaurant.update_container_props(init_state)

        for idx, item in enumerate(task_seq):
            if prep_state is not None and (idx + 1) % 2 == 1:
                new_prep_state = self.get_prepared_state_by_cleaner(restaurant, n_iterations=50)
                if new_prep_state is not None:
                    restaurant.update_container_props(new_prep_state)
            self.concern = ap_concern
            start = time.time()
            active_agent, task = item[0], item[1]
            restaurant.active_robot = active_agent
            last_task = prep_state is not None and (idx + 1 == len(task_seq))
            new_state, cost, new_task, help_stat = self.get_anticipatory_plan(restaurant, task, last_task=last_task)
            elapsed = time.time() - start
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

    def get_anticipatory_plan(self, restaurant, task, last_task=False, plan_only=False):
        save_file = '/data/figs/ap_time_analysis' + str(restaurant.seed) + '.txt'
        init_state = copy.deepcopy(restaurant.get_current_object_state())
        plan, ant_cost = self.myopic_planner.get_cost_and_state_from_task(restaurant, task)

        if plan is None:
            if plan_only:
                return plan, ant_cost
            return init_state, 2000, task, 3

        myopic_help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
        ant_state = copy.deepcopy(restaurant.get_final_state_from_plan(plan))

        if last_task:
            if plan_only:
                return plan, ant_cost
            return ant_state, ant_cost, task, myopic_help_stat

        ant_task = task
        used_items, used_containers, other_bots = self._extract_plan_context(plan)

        restaurant.update_container_props(ant_state)
        myopic_ex_cost = self.get_anticipated_cost(restaurant)
        myopic_ant_cost = myopic_ex_cost + ant_cost

        if restaurant.active_robot == 'server_bot':
            aug_predicates = self._aug_predicates_for_food_robot('server_bot', restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)
        elif restaurant.active_robot == 'cook_bot':
            aug_predicates = self._aug_predicates_for_food_robot('cook_bot', restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)
        else:
            aug_predicates = self.aug_predicates_for_cleaner(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)

        with open(save_file, "a+") as f:
            f.write(f"---Myopic: {ant_task}: {ant_cost} {myopic_ex_cost}---\n")

        exhausted = 0
        found = False
        for aug_pred in aug_predicates:
            if exhausted >= 2:
                break
            start_time = time.time()
            restaurant.update_container_props(init_state)
            ant_task_pred = f'(and {aug_pred} {task})'
            c_plan, c_cost = self.myopic_planner.get_cost_and_state_from_task(restaurant, ant_task_pred)
            if time.time() - start_time >= 100:
                exhausted += 1
            if c_plan is None:
                continue
            can_help_stat = taskplan_multi.utils.get_status_of_asking_help(c_plan)
            if can_help_stat > myopic_help_stat:
                continue
            can_state = restaurant.get_final_state_from_plan(c_plan)
            restaurant.update_container_props(can_state)
            can_ant_cost = self.get_anticipated_cost(restaurant) + c_cost
            if can_ant_cost < myopic_ant_cost:
                found = True
                ant_state = copy.deepcopy(can_state)
                ant_cost = c_cost
                plan = c_plan
                myopic_ant_cost = can_ant_cost
                ant_task = ant_task_pred
                myopic_help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
            restaurant.update_container_props(init_state)

        with open(save_file, "a+") as f:
            if found:
                f.write(f"{self.concern} -----{ant_task}: {myopic_ant_cost}---\n")
            else:
                f.write(f"{self.concern}---A.P.: Same as Myopic---\n")

        if plan_only:
            return plan, ant_cost
        return ant_state, ant_cost, ant_task, myopic_help_stat

    def get_prepared_state_by_cleaner(self, restaurant, n_iterations=1000):
        def safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                return 0.0

        total_budget = 1000
        self.concern = 'joint'
        prepared_state = copy.deepcopy(restaurant.get_current_object_state())
        int_cost = self.get_anticipated_cost(restaurant)
        i = 0
        temp = 100
        cooling_rate = 0.95
        conts = [c for c in KITCHEN_CONTAINERS + SERVING_ROOM_CONTAINERS if c not in CLEANER_BOT_RESTRICT]
        assets = [
            obj['assetId']
            for c in conts
            for obj in restaurant.get_objects_by_container_name(c)
            if 'washable' in obj and obj.get('dirty') == 1
        ]

        restaurant.active_robot = 'cook_bot'
        while i < n_iterations and total_budget > 0:
            restaurant.update_container_props(prepared_state)
            i += 1
            best_cost = 0
            current_prep = copy.deepcopy(prepared_state)
            for item in assets:
                restaurant.update_container_props(prepared_state)
                plan, cost = self.myopic_planner.get_cost_and_state_from_task(
                    restaurant, taskplan_multi.pddl.task.clean_something(item))
                actual_cost = max(cost - 500, 0)
                if plan is None:
                    continue
                candidate_state = restaurant.get_final_state_from_plan(plan)
                restaurant.update_container_props(candidate_state)
                can_exp_cost = self.get_anticipated_cost(restaurant)
                delta = can_exp_cost - int_cost
                exp = safe_exp(-delta / temp)
                if (delta < 0 or random.uniform(0, 1) < exp) and (total_budget - actual_cost) >= 0:
                    current_prep = copy.deepcopy(candidate_state)
                    int_cost = can_exp_cost
                    best_cost = actual_cost
                temp = max(temp * cooling_rate, 1)
            prepared_state = copy.deepcopy(current_prep)
            total_budget -= best_cost
            if best_cost == 0:
                break
        return prepared_state

    def get_prepared_state(self, restaurant, n_iterations=1000):
        def safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                return 0.0

        prepared_state = copy.deepcopy(restaurant.get_current_object_state())
        int_cost = self.get_anticipated_cost(restaurant)
        i = 0
        temp = 100
        cooling_rate = 0.95
        restaurant.active_all = True
        while i < n_iterations:
            restaurant.update_container_props(prepared_state)
            i += 1
            candidate_state = change_state(restaurant)
            restaurant.update_container_props(candidate_state)
            can_exp_cost = self.get_anticipated_cost(restaurant)
            delta = can_exp_cost - int_cost
            exp = safe_exp(-delta / temp)
            if delta < 0 or random.uniform(0, 1) < exp:
                prepared_state = copy.deepcopy(candidate_state)
                int_cost = can_exp_cost
            temp = max(temp * cooling_rate, 1)
        return prepared_state
