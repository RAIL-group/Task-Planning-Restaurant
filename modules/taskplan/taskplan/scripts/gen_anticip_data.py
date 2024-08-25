import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import taskplan
import copy
from collections import Counter
import os

MAX_DATA_PER_MAP = 500

# cup12, mug12, knife12, spreads(2)
# Dirty: mug2, cup2,


def change_state(proc_data, myopic_planner):
    conts = [c for (c, v) in proc_data.get_container_pos_list()]
    other_states = list()
    jars = ['jar1', 'jar2']
    for j in jars:
        for c in conts:
            tsk = taskplan.pddl.task.serve_water(c, j)
            plan, _ = (
                myopic_planner.get_cost_and_state_from_task(proc_data, tsk))
            if plan is None:
                continue
            can_state = proc_data.get_final_state_from_plan(plan)
            other_states.append(can_state)
    return other_states


def get_tasks():
    t1 = taskplan.pddl.task_distribution.cleaning_task()
    t1.extend(taskplan.pddl.task_distribution.organizing_task())
    t1.extend(taskplan.pddl.task_distribution.clear_task())
    tasks = list()
    for task in t1:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(val)
    return tasks


def get_serve_solid():
    t1 = taskplan.pddl.task_distribution.serve_solid()
    tasks = list()
    for task in t1:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(val)
    return tasks


def get_service_task():
    t1 = taskplan.pddl.task_distribution.service_tasks()
    tasks = list()
    for task in t1:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(val)
    return tasks


def gen_data_main(args):
    # Get restaurant data for a send and extract initial object states
    proc_data = taskplan.environments.restaurant.RESTAURANT(
        seed=args.current_seed)
    whole_graph = taskplan.utils.get_graph(proc_data)
    file_name = 'non_qualified_logs/log.txt'
    logfile = os.path.join(args.save_dir, file_name)
    map_counter = 0
    myopic_planner = taskplan.planners.myopic_planner.MyopicPlanner()
    tasks_1 = get_tasks()
    tasks_2 = get_service_task()
    tasks_3 = get_serve_solid()
    # alternates = list()
    #####
    save_file = '/data/figs/grid' + str(args.current_seed) + '.png'
    grid = proc_data.grid
    # occupied_cells = np.argwhere(grid == 1)
    plt.clf()
    plt.imshow(grid, cmap='gray_r')
    # plt.scatter(occupied_cells[:, 1], occupied_cells[:, 0], c='red', label='Occupied Cells')
    for pose in proc_data.accessible_poses:
        x, y = proc_data.accessible_poses[pose]
        plt.text(y, x, pose, fontsize=6)
    plt.savefig(save_file, dpi=1200)
    plt.clf()
    obj_states = change_state(proc_data, myopic_planner)
    while (map_counter < MAX_DATA_PER_MAP):
        exp_cost_1 = myopic_planner.get_expected_cost(
            proc_data, tasks_1,
            return_alt_goals=False)
        exp_cost_2 = myopic_planner.get_expected_cost(
            proc_data, tasks_2,
            return_alt_goals=False)
        exp_cost_3 = myopic_planner.get_expected_cost(
            proc_data, tasks_3,
            return_alt_goals=False)
        if exp_cost_1 is None or exp_cost_2 is None or exp_cost_3 is None:
            with open(logfile, "a+") as f:
                f.write(
                    f" | Seed No: {args.current_seed} \n"
                    f" | Map No: {map_counter} \n"
                )
            # raise NotImplementedError
            break
        exp_cost = (exp_cost_1 + (exp_cost_2 * 10) + (exp_cost_3 * 3))/14
        whole_graph['label'] = exp_cost
        taskplan.utils.write_datum_to_file(args, whole_graph, map_counter)
        map_counter += 1
        if (map_counter < 10 or map_counter % 100 == 1):
            image_graph = taskplan.utils.graph_formatting(whole_graph)
            img = image_graph['graph_image']
            title = f'Map: {map_counter}: Exp: {exp_cost}'
            plt.clf()
            plt.title(title)
            plt.imshow(img)
            plt.savefig(f'/data/figs/{args.data_file_base_name}_{args.current_seed}_{map_counter}.png')
            plt.clf()

        if len(obj_states) > 0:
            object_state = obj_states.pop()
        else:
            if random.random() > 0.5:
                object_state = proc_data.get_random_state(no_dirty=True)
            else:
                object_state = proc_data.get_random_state()
        proc_data.update_container_props(object_state[0],
                                         object_state[1],
                                         object_state[2])
        whole_graph = taskplan.utils.get_graph(proc_data)

    image_graph = taskplan.utils.graph_formatting(whole_graph)
    img = image_graph['graph_image']
    title = 'Complete Data'
    plt.clf()
    plt.title(title)
    plt.imshow(img)
    plt.savefig(f'{args.save_dir}data_completion_logs/{args.data_file_base_name}_{args.current_seed}.png')
    plt.clf()


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation"
    )
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--data_file_base_name', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    gen_data_main(args)
