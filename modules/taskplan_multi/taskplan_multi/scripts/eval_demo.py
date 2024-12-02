import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import taskplan_multi
import copy
from collections import Counter
import os
from skimage.morphology import erosion
import gridmap
from shapely import geometry

COLLISION_VAL = 1
FREE_VAL = 0
UNOBSERVED_VAL = -1
assert (COLLISION_VAL > FREE_VAL)
assert (FREE_VAL > UNOBSERVED_VAL)
OBSTACLE_THRESHOLD = 0.5 * (FREE_VAL + COLLISION_VAL)


FOOT_PRINT = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])

EVAL_NO = 5

def make_plotting_grid(grid_map):
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= OBSTACLE_THRESHOLD
    # Take one pixel boundary of the region collision
    thinned = erosion(collision, footprint=FOOT_PRINT)
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < OBSTACLE_THRESHOLD, grid_map >= FREE_VAL)
    grid[:, :, 0][free] = 1
    grid[:, :, 1][free] = 1
    grid[:, :, 2][free] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid

def get_tasks(restaurant):
    clean_items = list()
    dirty_items = list()
    for container in restaurant.containers:
        children = container.get('children')
        if children is None:
            continue
        for child in children:
            if 'dirty' in child and child['dirty'] == 1:
                dirty_items.append(child['assetId'])
            else:
                clean_items.append(child['assetId'])
    
    tall_tasks = taskplan_multi.pddl.task_distribution.tall_robots_tasks(clean_items)
    tiny_tasks = taskplan_multi.pddl.task_distribution.tiny_robots_tasks(dirty_items)
    if len(tall_tasks) > 0 and len(tiny_tasks):
        tasks = list()
        for task in tall_tasks:
            key = list(task.keys())[0]
            val = task[key]
            tasks.append(('agent_tall', val))
        for task in tiny_tasks:
            key = list(task.keys())[0]
            val = task[key]
            tasks.append(('agent_tiny', val))
        random.shuffle(tasks)
        return tasks
    return None


def eval_main(args):
    # Get restaurant data for a send and extract initial object states
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=args.current_seed)
    task_sequence = get_tasks(restaurant)
    for i in range(EVAL_NO):
        restaurant.roll_back_to_init()
        myopic_planner.get_seq_cost(args, restaurant, task_sequence, i)
        restaurant.roll_back_to_init()
        ant_planner.get_seq_cost(args, restaurant, task_sequence, i, ap_concern='joint')
        restaurant.roll_back_to_init()
        ant_planner.get_seq_cost(args, restaurant, task_sequence, i, ap_concern='self')
        restaurant.roll_back_to_init()
        ant_planner.get_seq_cost(args, restaurant, task_sequence, i, ap_concern='other')
        random.shuffle(task_sequence)

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation"
    )
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--tall_network', type=str, required=False)
    parser.add_argument('--tiny_network', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # print(args)
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    eval_main(args)