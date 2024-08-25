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
import itertools

IMPACT_FACTOR = 5


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def change_state(proc_data):
    state = None
    while (state is None):
        obj = random.choice(taskplan.environments.sampling.load_movables())
        cont_list = [v for (c, v) in proc_data.get_container_pos_list()]
        container = random.choice(cont_list)
        if 'jar' in obj:
            state = proc_data.fill_up_jar_n_place(obj, container)
        elif 'washable' in obj and 'dirty' in obj:
            state = proc_data.place_n__clean_object(obj, container)
        else:
            state = proc_data.place_object(obj, container)
    return state


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

    def get_prepared_state(self, proc_data, n_iterations=1000):
        def safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                # If the argument is too large in magnitude, return an approximation
                return 0.0
        save_file = '/data/figs/learned_prep' + str(
            proc_data.seed) + '.txt'

        prepared_state = copy.deepcopy(proc_data.get_current_object_state())
        int_cost = self.get_anticipated_cost(proc_data)
        with open(save_file, "a+") as f:
            f.write(
                f"| Initial State"
                f"| Expected Cost: {int_cost}\n")
        i = 0
        temp = 1000
        cooling_rate = 0.99
        while (i < n_iterations):
            proc_data.update_container_props(prepared_state[0],
                                             prepared_state[1],
                                             prepared_state[2])
            i += 1
            # task_to_solve = random.choice(task_list)
            # plan, _ = (
            #         self.myopic_planner.get_cost_and_state_from_task(
            #             proc_data, task_to_solve))
            # # candidate_state = proc_data.randomize_objects()
            # if plan is None:
            #     continue

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

    def get_manual_prepared_state_anneal(self, proc_data,
                                         task_list, n_iterations=1000):
        def safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                # If the argument is too large in magnitude, return an approximation
                return 0.0

        save_file = '/data/figs/computed_prep' + str(
            proc_data.seed) + '.txt'
        prepared_state = copy.deepcopy(proc_data.get_current_object_state())
        int_cost = self.myopic_planner.get_expected_cost(proc_data, task_list)
        with open(save_file, "a+") as f:
            f.write(
                f"| Initial State"
                f"| Expected Cost: {int_cost}\n")
        i = 0
        temp = 1000
        cooling_rate = 0.99
        while (i < n_iterations):
            proc_data.update_container_props(prepared_state[0],
                                             prepared_state[1])
            i += 1
            task_to_solve = random.choice(task_list)
            plan, _ = self.myopic_planner.get_cost_and_state_from_task(
                proc_data, task_to_solve)
            if plan is None:
                continue
            candidate_state = proc_data.get_final_state_from_plan(plan)
            proc_data.update_container_props(candidate_state[0],
                                             candidate_state[1])
            can_exp_cost = self.myopic_planner.get_expected_cost(proc_data,
                                                                 task_list)
            if can_exp_cost is None:
                with open(save_file, "a+") as f:
                    f.write(f"| whatta satte: {candidate_state}")
                continue
            delta = can_exp_cost-int_cost
            exp = safe_exp(-delta/temp)
            if delta < 0 or random.uniform(0, 1) < exp:
                prepared_state = copy.deepcopy(candidate_state)
                int_cost = can_exp_cost
                with open(save_file, "a+") as f:
                    f.write(
                        f"| idx: {i}"
                        f"| task: {task_to_solve}"
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

    def get_anticipatory_state(self, proc_data, task, desc):
        init_state = copy.deepcopy(proc_data.get_current_object_state())
        plan, cost = (
            self.myopic_planner.get_cost_and_state_from_task(proc_data, task)
        )
        # print(task)
        # print(plan)
        # save_file = '/data/figs/anticipatory_debug' + str(
        #     proc_data.seed) + '.txt'
        if plan is None:
            return None, None
        term_state = proc_data.get_final_state_from_plan(plan)
        if cost == 0:
            return term_state, cost
        proc_data.update_container_props(term_state[0], term_state[1],
                                         term_state[2])
        myopic_ex_cost = (self.get_anticipated_cost(proc_data))
        myopic_ant_cost = (myopic_ex_cost * IMPACT_FACTOR) + cost
        # with open(save_file, "a+") as f:
        #     f.write(
        #         f"| Task: {task}"
        #         f"| Myopic Cost: {cost}"
        #         f"| Myopic EC: {myopic_ex_cost}"
        #         f"| Myopic ANTC: {myopic_ant_cost}\n")
        # print(desc)
        ant_task_desc = desc
        # print(f"myopic_ex_cost, cost: {myopic_ex_cost} {cost}")
        ant_cost = cost
        ant_state = copy.deepcopy(term_state)
        conts = [c for (c, v) in proc_data.get_container_pos_list()]
        if 'clear_the' in desc:
            proc_data.update_container_props(init_state[0],
                                             init_state[1],
                                             init_state[2])
            if 'servingtable1' in task:
                ant_objects = proc_data.get_objects_by_container_name('servingtable1')
                conts.remove('servingtable1')
            elif 'servingtable2' in task:
                ant_objects = proc_data.get_objects_by_container_name('servingtable2')
                conts.remove('servingtable2')
            elif 'servingtable3' in task:
                ant_objects = proc_data.get_objects_by_container_name('servingtable3')
                conts.remove('servingtable3')
            elif 'coffeemachine' in task:
                ant_objects = proc_data.get_objects_by_container_name('coffeemachine')
                conts.remove('coffeemachine')
            elif 'dishwasher' in task:
                ant_objects = proc_data.get_objects_by_container_name('dishwasher')
                conts.remove('dishwasher')
            else:
                ant_objects = proc_data.get_objects_by_container_name('countertop')
                conts.remove('countertop')
            if len(ant_objects) == 0:
                return ant_state, ant_cost

            cont_combos = list(itertools.product(conts,
                                                 repeat=len(ant_objects)))
            random.shuffle(cont_combos)
            if len(cont_combos) > 50:
                if 'servingtable1' in conts:
                    conts.remove('servingtable1')
                if 'servingtable2' in conts:
                    conts.remove('servingtable2')
                if 'servingtable3' in conts:
                    conts.remove('servingtable3')
                cont_combos = list(itertools.product(conts,
                                                 repeat=len(ant_objects)))
            all_tasks = list()
            for comb_tuple in cont_combos:
                if len(comb_tuple) != len(ant_objects):
                    continue
                tt = '(and'
                for idx, item in enumerate(comb_tuple):
                    t1 = taskplan.pddl.task.place_something(ant_objects[idx]['assetId'], item)
                    tt += t1
                    pos = proc_data.get_container_pos(item)
                    if pos is not None:
                        new_state = list()
                        new_state = proc_data.place_object(ant_objects[idx],
                                                           pos)
                        proc_data.update_container_props(new_state[0],
                                                         new_state[1],
                                                         new_state[2])

                tt += ')'
                if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                    all_tasks.append(tt)
                proc_data.update_container_props(init_state[0],
                                                 init_state[1],
                                                 init_state[2])
            if len(all_tasks) > 50:
                # with open(save_file, "a+") as f:
                #     f.write(f"| Original Tasks: {len(all_tasks)}\n")
                all_tasks = random.sample(all_tasks, 50)
            # with open(save_file, "a+") as f:
            #     f.write(f"| Starting...: {len(all_tasks)}\n")
            for ant_task_pred in all_tasks:
                plan, c_cost = (
                    self.myopic_planner.get_cost_and_state_from_task(
                        proc_data, ant_task_pred)
                )
                if plan is None:
                    continue
                can_state = proc_data.get_final_state_from_plan(plan)
                proc_data.update_container_props(can_state[0],
                                                 can_state[1],
                                                 can_state[2])
                can_ex_cost = self.get_anticipated_cost(proc_data)
                can_ant_cost = (can_ex_cost * IMPACT_FACTOR) + (c_cost)
                # with open(save_file, "a+") as f:
                #     f.write(
                #         f"| Task: {ant_task_pred}"
                #         f"| Can Cost: {c_cost}"
                #         f"| Can EC: {can_ex_cost}"
                #         f"| Can ANTC: {can_ant_cost}\n")
                if can_ex_cost < myopic_ex_cost and can_ant_cost < myopic_ant_cost:
                    ant_state = copy.deepcopy(can_state)
                    ant_cost = c_cost
                    myopic_ant_cost = can_ant_cost
                    ant_task_desc = ant_task_pred
                proc_data.update_container_props(init_state[0],
                                                 init_state[1],
                                                 init_state[2])
        elif desc == 'serve_water' or desc == 'serve_coffee':
            proc_data.update_container_props(init_state[0],
                                             init_state[1],
                                             init_state[2])
            ant_task = list()
            conts = [c for (c, v) in proc_data.get_container_pos_list()]
            if 'jar' not in task:
                for c in conts:
                    ant_task.append(taskplan.pddl.task.serve_water(
                        c, 'jar1'))
                    ant_task.append(taskplan.pddl.task.serve_water(
                        c, 'jar2'))
            # with open(save_file, "a+") as f:
            #     f.write(f"| Starting...: {len(ant_task)}\n")
            for td in ant_task:
                ant_task_pred = f'(and {td} {task})'
                plan, c_cost = (
                    self.myopic_planner.get_cost_and_state_from_task(
                        proc_data, ant_task_pred))
                if plan is None:
                    continue
                can_state = proc_data.get_final_state_from_plan(plan)
                proc_data.update_container_props(can_state[0],
                                                 can_state[1],
                                                 can_state[2])
                can_ex_cost = self.get_anticipated_cost(proc_data)
                can_ant_cost = (can_ex_cost * IMPACT_FACTOR) + (c_cost)
                # with open(save_file, "a+") as f:
                #     f.write(
                #         f"| Task: {ant_task_pred}"
                #         f"| Can Cost: {c_cost}"
                #         f"| Can EC: {can_ex_cost}"
                #         f"| Can ANTC: {can_ant_cost}\n")
                if can_ex_cost < myopic_ex_cost and can_ant_cost < myopic_ant_cost:
                    ant_state = copy.deepcopy(can_state)
                    ant_cost = c_cost
                    myopic_ant_cost = can_ant_cost
                    ant_task_desc = ant_task_pred
                proc_data.update_container_props(init_state[0],
                                                 init_state[1],
                                                 init_state[2])
        elif desc == 'serve_sandwich' or desc == 'serve_fruit_bowl':
            proc_data.update_container_props(init_state[0],
                                             init_state[1],
                                             init_state[2])

            if 'servingtable1' in task:
                ant_objects = proc_data.get_objects_by_container_name('servingtable1')
            if 'servingtable2' in task:
                ant_objects = proc_data.get_objects_by_container_name('servingtable2')
            if 'servingtable3' in task:
                ant_objects = proc_data.get_objects_by_container_name('servingtable3')

            proc_data.update_container_props(term_state[0],
                                             term_state[1],
                                             term_state[2])
            kn_obj = proc_data.get_object_props_by_name('knife1')
            if kn_obj is not None:
                ant_objects.append(kn_obj)
            ant_task = list()
            for ob in ant_objects:
                for (c, v) in proc_data.get_container_pos_list():
                    new_state = proc_data.place_object(ob, v)
                    proc_data.update_container_props(new_state[0],
                                                     new_state[1],
                                                     new_state[2])
                    if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                        ant_task.append(taskplan.pddl.task.place_something(
                            ob['assetId'], c))
                    proc_data.update_container_props(term_state[0],
                                                     term_state[1],
                                                     term_state[2])
                    if 'dirty' in ob and ob['dirty'] == 1:
                        new_state = proc_data.place_n__clean_object(ob, v)
                        proc_data.update_container_props(new_state[0],
                                                         new_state[1],
                                                         new_state[2])
                        if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                            ant_task.append(taskplan.pddl.task.clean_and_place(
                                ob['assetId'], c))
                        proc_data.update_container_props(term_state[0],
                                                         term_state[1],
                                                         term_state[2])

            proc_data.update_container_props(init_state[0],
                                             init_state[1],
                                             init_state[2])
            if len(ant_task) > 50:
                # with open(save_file, "a+") as f:
                #     f.write(f"| Original Task...: {len(ant_task)}\n")
                ant_task = random.sample(ant_task, 50)
            # with open(save_file, "a+") as f:
            #     f.write(f"| Starting...: {len(ant_task)}\n")
            for td in ant_task:
                ant_task_pred = f'(and {td} {task})'
                plan, c_cost = (
                    self.myopic_planner.get_cost_and_state_from_task(
                        proc_data, ant_task_pred))
                if plan is None:
                    continue
                can_state = proc_data.get_final_state_from_plan(plan)
                proc_data.update_container_props(can_state[0],
                                                 can_state[1],
                                                 can_state[2])
                can_ex_cost = self.get_anticipated_cost(proc_data)
                can_ant_cost = (can_ex_cost * IMPACT_FACTOR) + (c_cost)
                # with open(save_file, "a+") as f:
                #     f.write(
                #         f"| Task: {ant_task_pred}"
                #         f"| Can Cost: {c_cost}"
                #         f"| Can EC: {can_ex_cost}"
                #         f"| Can ANTC: {can_ant_cost}\n")
                if can_ex_cost < myopic_ex_cost and can_ant_cost < myopic_ant_cost:
                    ant_state = copy.deepcopy(can_state)
                    ant_cost = c_cost
                    myopic_ant_cost = can_ant_cost
                    ant_task_desc = ant_task_pred
                proc_data.update_container_props(init_state[0],
                                                 init_state[1],
                                                 init_state[2])

        elif 'clean' in desc:
            proc_data.update_container_props(init_state[0],
                                             init_state[1],
                                             init_state[2])
            ant_task = list()
            conts = [c for (c, v) in proc_data.get_container_pos_list()]
            if 'mug1' in task:
                target_obj = proc_data.get_object_props_by_name('mug1')
                for (c, v) in proc_data.get_container_pos_list():
                    new_state = proc_data.place_n__clean_object(target_obj, v)
                    proc_data.update_container_props(new_state[0],
                                                     new_state[1],
                                                     new_state[2])
                    if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                        ant_task.append(taskplan.pddl.task.place_something(
                            target_obj['assetId'], c))
                    proc_data.update_container_props(init_state[0],
                                                     init_state[1],
                                                     init_state[2])
            if 'mug2' in task:
                target_obj = proc_data.get_object_props_by_name('mug2')
                for (c, v) in proc_data.get_container_pos_list():
                    new_state = proc_data.place_n__clean_object(target_obj, v)
                    proc_data.update_container_props(new_state[0],
                                                     new_state[1],
                                                     new_state[2])
                    if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                        ant_task.append(taskplan.pddl.task.place_something(
                            target_obj['assetId'], c))
                    proc_data.update_container_props(init_state[0],
                                                     init_state[1],
                                                     init_state[2])
            if 'bowl1' in task:
                target_obj = proc_data.get_object_props_by_name('bowl1')
                for (c, v) in proc_data.get_container_pos_list():
                    new_state = proc_data.place_n__clean_object(target_obj, v)
                    proc_data.update_container_props(new_state[0],
                                                     new_state[1],
                                                     new_state[2])
                    if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                        ant_task.append(taskplan.pddl.task.place_something(
                            target_obj['assetId'], c))
                    proc_data.update_container_props(init_state[0],
                                                     init_state[1],
                                                     init_state[2])
            if 'cup1' in task:
                target_obj = proc_data.get_object_props_by_name('cup1')
                for (c, v) in proc_data.get_container_pos_list():
                    new_state = proc_data.place_n__clean_object(target_obj, v)
                    proc_data.update_container_props(new_state[0],
                                                     new_state[1],
                                                     new_state[2])
                    if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                        ant_task.append(taskplan.pddl.task.place_something(
                            target_obj['assetId'], c))
                    proc_data.update_container_props(init_state[0],
                                                     init_state[1],
                                                     init_state[2])
            if 'cup2' in task:
                target_obj = proc_data.get_object_props_by_name('cup2')
                for (c, v) in proc_data.get_container_pos_list():
                    new_state = proc_data.place_n__clean_object(target_obj, v)
                    proc_data.update_container_props(new_state[0],
                                                     new_state[1],
                                                     new_state[2])
                    if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                        ant_task.append(taskplan.pddl.task.place_something(
                            target_obj['assetId'], c))
                    proc_data.update_container_props(init_state[0],
                                                     init_state[1],
                                                     init_state[2])
            if 'knife1' in task:
                target_obj = proc_data.get_object_props_by_name('knife1')
                for (c, v) in proc_data.get_container_pos_list():
                    new_state = proc_data.place_n__clean_object(target_obj, v)
                    proc_data.update_container_props(new_state[0],
                                                     new_state[1],
                                                     new_state[2])
                    if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                        ant_task.append(taskplan.pddl.task.place_something(
                            target_obj['assetId'], c))
                    proc_data.update_container_props(init_state[0],
                                                     init_state[1],
                                                     init_state[2])

            # with open(save_file, "a+") as f:
            #     f.write(f"| Starting...: {len(ant_task)}\n")
            for td in ant_task:
                ant_task_pred = f'(and {td} {task})'
                plan, c_cost = (
                    self.myopic_planner.get_cost_and_state_from_task(
                        proc_data, ant_task_pred))
                if plan is None:
                    continue
                can_state = proc_data.get_final_state_from_plan(plan)
                proc_data.update_container_props(can_state[0],
                                                 can_state[1],
                                                 can_state[2])
                can_ex_cost = self.get_anticipated_cost(proc_data)
                can_ant_cost = (can_ex_cost * IMPACT_FACTOR) + (c_cost)
                # with open(save_file, "a+") as f:
                #     f.write(
                #         f"| Task: {ant_task_pred}"
                #         f"| Can Cost: {c_cost}"
                #         f"| Can EC: {can_ex_cost}"
                #         f"| Can ANTC: {can_ant_cost}\n")
                if can_ex_cost < myopic_ex_cost and can_ant_cost < myopic_ant_cost:
                    ant_state = copy.deepcopy(can_state)
                    ant_cost = c_cost
                    myopic_ant_cost = can_ant_cost
                    ant_task_desc = ant_task_pred
                proc_data.update_container_props(init_state[0],
                                                 init_state[1],
                                                 init_state[2])

        else:
            ant_task = list()
            # ant_task.append(taskplan.pddl.task.move_robot('initial_robot_pose'))
            ant_objects = proc_data.get_objects_by_container_name(proc_data.rob_at)
            reachable_conts = []
            for item in conts:
                if item == proc_data.rob_at:
                    continue
                # if proc_data.known_cost[proc_data.rob_at][item] < 100:
                    # temp = proc_data.get_objects_by_container_name(item)
                # ant_task.append(taskplan.pddl.task.move_robot(item))
                reachable_conts.append(item)
                # if temp:
                #     ant_objects.extend(temp)
            if len(reachable_conts) > 0:
                for nm in reachable_conts:
                    for obj in ant_objects:
                        if obj['assetId'] in task:
                            continue
                        if 'pickable' in obj:
                            pos = proc_data.get_container_pos(nm)
                            if pos is not None:
                                new_state = list()
                                new_state = proc_data.place_object(obj, pos)
                                proc_data.update_container_props(new_state[0],
                                                                 new_state[1],
                                                                 new_state[2])
                                if self.get_anticipated_cost(proc_data) < myopic_ex_cost:
                                    ant_task.append(taskplan.pddl.task.place_something(
                                            obj['assetId'], nm))
                        proc_data.update_container_props(init_state[0],
                                                         init_state[1],
                                                         init_state[2])
            # with open(save_file, "a+") as f:
            #     f.write(f"| Starting...: {len(ant_task)}\n")
            for td in ant_task:
                ant_task_pred = f'(and {td} {task})'
                plan, c_cost = (
                    self.myopic_planner.get_cost_and_state_from_task(
                        proc_data, ant_task_pred))
                if plan is None:
                    continue
                can_state = proc_data.get_final_state_from_plan(plan)
                proc_data.update_container_props(can_state[0],
                                                 can_state[1],
                                                 can_state[2])
                can_ex_cost = self.get_anticipated_cost(proc_data)
                can_ant_cost = (can_ex_cost * IMPACT_FACTOR) + (c_cost)
                # with open(save_file, "a+") as f:
                #     f.write(
                #         f"| Task: {ant_task_pred}"
                #         f"| Can Cost: {c_cost}"
                #         f"| Can EC: {can_ex_cost}"
                #         f"| Can ANTC: {can_ant_cost}\n")
                if can_ex_cost < myopic_ex_cost and can_ant_cost < myopic_ant_cost:
                    ant_state = copy.deepcopy(can_state)
                    ant_cost = c_cost
                    myopic_ant_cost = can_ant_cost
                    ant_task_desc = ant_task_pred
                proc_data.update_container_props(init_state[0],
                                                 init_state[1],
                                                 init_state[2])
        return ant_state, ant_cost

    def get_seq_cost(self, args, proc_data, task_seq,
                     seq_num,
                     prep=False):
        file_name = 'no_prep_ap/beta_' + args.logfile_name
        if prep:
            file_name = 'prep_ap/beta_' + args.logfile_name
        logfile = os.path.join(args.save_dir, file_name)
        total_cost = 0
        start_time = time.time()
        for idx, task in enumerate(task_seq):
            exp_bef = self.get_anticipated_cost(proc_data)
            key = list(task.keys())[0]
            val = task[key]
            new_state, cost = (
                self.get_anticipatory_state(proc_data, val, key)
            )
            if new_state is None or cost is None:
                return None, None
            total_cost += cost
            proc_data.update_container_props(new_state[0], new_state[1],
                                             new_state[2])
            # exp_aft = self.get_anticipated_cost(proc_data)
            with open(logfile, "a+") as f:
                f.write(
                    f" | seq: S{seq_num}"
                    f" | desc: {key}"
                    f" | num: T{idx+1}"
                    f" | cost: {cost:0.4f}"
                    f" | ec_est: {exp_bef:0.4f}\n"
                    # f" | ec_aft: {exp_aft:0.4f}\n"
                )
        return total_cost, time.time() - start_time
