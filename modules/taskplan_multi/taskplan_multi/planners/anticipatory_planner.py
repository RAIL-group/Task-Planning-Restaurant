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

class AntcipatoryPlanner:
    def __init__(self, args, domain=taskplan_multi.pddl.domain.get_domain()):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.domain = domain
        self.eval_net_tall = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
            network_file=args.tall_network, device=device)
        self.eval_net_tiny = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
            network_file=args.tiny_network, device=device)
        self.myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner(
            self.domain)
        self.concern = 'joint'

    def get_anticipated_cost(self, restaurant):
        current_active = restaurant.active_robot
        restaurant.active_robot = 'agent_tall'
        whole_graph = taskplan_multi.utils.get_graph(restaurant)
        anticipated_cost_tall = self.eval_net_tall(whole_graph)
        restaurant.active_robot = 'agent_tiny'
        whole_graph = taskplan_multi.utils.get_graph(restaurant)
        anticipated_cost_tiny = self.eval_net_tiny(whole_graph)
        restaurant.active_robot = current_active
        if self.concern == 'self':
            if restaurant.active_robot == 'agent_tall':
                return anticipated_cost_tall
            else:
                return anticipated_cost_tiny
        elif self.concern == 'other':
            if restaurant.active_robot == 'agent_tall':
                return anticipated_cost_tiny
            else:
                return anticipated_cost_tall
        else:
            return anticipated_cost_tiny + anticipated_cost_tall
    
    def get_seq_cost(self, args, restaurant, task_seq, seq_num, ap_concern='joint'):
        costs = list()
        file_name = 'ap_' + ap_concern + '.txt'
        self.concern = ap_concern
        logfile = os.path.join(args.save_dir, file_name)
        for idx, item in enumerate(task_seq):
            active_agent = item[0]
            task = item[1]
            restaurant.active_robot = active_agent
            new_state, cost = self.get_anticipatory_plan(restaurant, task)
            costs.append(cost)
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
            return init_state, 10000
        if cost == 0:
            return init_state, cost
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
        return ant_state, ant_cost

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