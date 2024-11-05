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
    
    if restaurant.active_robot == 'agent_tall':
        t1 = taskplan_multi.pddl.task_distribution.tall_robots_tasks(clean_items)
    else:
        t1 = taskplan_multi.pddl.task_distribution.tiny_robots_tasks(dirty_items)
    tasks = list()
    for task in t1:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(val)
    return tasks


def gen_data_main(args):
    # Get restaurant data for a send and extract initial object states
    map_counter = 0
    if args.agent:
        active_agent = args.agent
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=args.current_seed, active=active_agent)
    tasks = get_tasks(restaurant)
    exp_cost = None
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    if len(tasks) > 0:
        exp_cost = myopic_planner.get_expected_cost(
            restaurant, tasks)
        whole_graph['label'] = exp_cost
        taskplan_multi.utils.write_datum_to_file(args, whole_graph, map_counter)
    image_graph = taskplan_multi.utils.graph_formatting(whole_graph)
    plt.clf()
    plt.title(f'Seed: {args.current_seed} : ExC: {exp_cost}')
    plt.imshow(image_graph['graph_image'])
    plt.savefig(f'{args.save_dir}data_completion_logs/{args.data_file_base_name}_{args.current_seed}.png', dpi=600)


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation"
    )
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--agent', type=str, required=False)
    parser.add_argument('--data_file_base_name', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    gen_data_main(args)