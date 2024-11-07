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


def eval_main(args):
    # Get restaurant data for a send and extract initial object states
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net_tall = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file=args.tall_network, device=device)
    eval_net_tiny = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file=args.tiny_network, device=device)
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    active_agent = 'tiny'
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=args.current_seed, active=active_agent)
    tiny_tasks = get_tasks(restaurant)
    restaurant.active_robot = 'agent_tall'
    tall_tasks = get_tasks(restaurant)
    restaurant.active_robot = 'agent_tiny'
    tasks = [
        taskplan_multi.pddl.task.place_something('bowl2', 'dishwasher'),
        taskplan_multi.pddl.task.place_something('mug1', 'cabinet'),
        taskplan_multi.pddl.task.place_something('mug2', 'cabinet')
    ]
    plan, cost = myopic_planner.get_cost_and_state_from_task(
        restaurant, tasks[0])
    new_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(new_state)
    restaurant.active_robot = 'agent_tiny'
    exp_cost_tiny = myopic_planner.get_expected_cost(
            restaurant, tiny_tasks)
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ant_cost_tiny = eval_net_tiny(whole_graph)
    restaurant.active_robot = 'agent_tall'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ant_cost_tall = eval_net_tall(whole_graph)
    exp_cost_tall = myopic_planner.get_expected_cost(
            restaurant, tall_tasks)
    print(f"Myopic Cost: For Tiny Robot {cost}")
    print(f"Myopic Exp Cost (Computed): Tiny {exp_cost_tiny} & Tall : {exp_cost_tall}")
    print(f"Myopic Exp Cost (Learned): Tiny {ant_cost_tiny} & Tall : {ant_cost_tall}")
    restaurant.roll_back_to_init()
    task_extra = taskplan_multi.pddl.task.place_something('mug1', 'countertop') # this should come from a process, this probably is our contribution.
    comb_task = f'(and {tasks[0]} {task_extra})'
    restaurant.active_robot = 'agent_tiny'
    plan, cost = myopic_planner.get_cost_and_state_from_task(
        restaurant, comb_task)
    new_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(new_state)
    restaurant.active_robot = 'agent_tiny'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ant_cost_tiny = eval_net_tiny(whole_graph)
    exp_cost_tiny = myopic_planner.get_expected_cost(
            restaurant, tiny_tasks)
    restaurant.active_robot = 'agent_tall'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ant_cost_tall = eval_net_tall(whole_graph)
    exp_cost_tall = myopic_planner.get_expected_cost(
            restaurant, tall_tasks)
    print(f"AP Cost: For Tiny Robot {cost}")
    print(f"AP Exp Cost (Computed): Tiny {exp_cost_tiny} & Tall : {exp_cost_tall}")
    print(f"AP Exp Cost (Learned): Tiny {ant_cost_tiny} & Tall : {ant_cost_tall}")
    raise NotImplementedError

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