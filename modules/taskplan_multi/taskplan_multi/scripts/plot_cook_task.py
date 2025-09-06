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
from taskplan_multi.environments.restaurant import world_to_grid
import json
import ast

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

EVAL_NO = 20
MAX_TASK_COOK = 20
MAX_TASK_SERVER = 30
MAX_TASK_CLEANER = 30
TASK_PER_SEQ = 15

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

def get_tasks():
    tasks_cook = taskplan_multi.pddl.task_distribution.tasks_for_cook()
    tasks = list()
    for task in tasks_cook:
        val = task[1]
        tasks.append(('cook_bot', val))
    selected_tasks = random.sample(tasks, TASK_PER_SEQ)
    return selected_tasks


def manual_tasks():
    tasks = list()
    tasks.append(
        ('cook_bot', taskplan_multi.pddl.task.prep_item('pasta', 'bowl', 'stove'))
    )
    return tasks

def plot_state(restaurant, args, image_name='init', title='None'):
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    plt.clf()
    for container in restaurant.containers:
        _x, _y = world_to_grid(
            container['position']['x'], container['position']['z'],
            restaurant.grid_min_x, restaurant.grid_min_z, restaurant.grid_res)
        assetId = container['assetId']
        plt.text(_x, _y, assetId, fontsize=8, color='blue')
        children = container.get('children', [])
        for i, child in enumerate(children):
            name = child['assetId']
            cl = 'green'
            if 'dirty' in child and child['dirty'] == 1:
                name = 'dirty ' + name
                cl = 'red'
            if 'cooked' in child and child['cooked'] == 1:
                name = 'cooked ' + name
                cl = 'gold'
            plt.text(_x, _y + (i + 1) * 1.2, name, fontsize=5, color=cl)  # Slight offset
        plt.imshow(img, cmap='gray_r', alpha=0.5)
    # plt.axis('off')  # Hides the axis
    plt.title(title)
    plt.savefig(f'{args.save_dir}/{image_name}_{args.current_seed}.png', dpi=600)

def eval_main(args):
    # Get restaurant data for a send and extract initial object states
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
    agents = ['cook_bot', 'cleaner_bot', 'server_bot']
    active_agent = 'cook_bot'
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=args.current_seed, agents=agents, active=active_agent)
    restaurant.active_all = False
    plan_file = 'myopic_plan.txt'
    plan_log = os.path.join(args.save_dir, plan_file)

    no_prep_state = copy.deepcopy(restaurant.get_current_object_state())
    init_exp_cost = ant_planner.get_anticipated_cost(restaurant)
    plot_state(restaurant, args, image_name='no-prep-init', title=f'Not Prepared State (INIT) \n Exp Cost: {init_exp_cost}')
    
    sampled_task = manual_tasks()
    for idx, item in enumerate(sampled_task):
        active_agent = item[0]
        task = item[1]
        restaurant.active_robot = active_agent
        restaurant.asked_help = False
        myopic_plan, myopic_cost = myopic_planner.get_cost_and_state_from_task(restaurant, task)
        with open(plan_log, "a+") as f:
            f.write(
                f"mp-plan: {myopic_plan}\n"
                f"mp-cost: {myopic_cost}\n"
            )
        ant_planner.concern = 'self'
        ap_plan, ap_cost = ant_planner.get_anticipatory_plan(restaurant, task, last_task=False, plan_only=True)
        with open(plan_log, "a+") as f:
            f.write(
                f"ap-plan: {ap_plan}\n"
                f"ap-cost: {ap_cost}\n"
            )
        restaurant.update_container_props(no_prep_state)
        ant_planner.concern = 'joint'
        joint_plan, joint_cost = ant_planner.get_anticipatory_plan(restaurant, task, last_task=False, plan_only=True)
        with open(plan_log, "a+") as f:
            f.write(
                f"joint-plan: {joint_plan}\n"
                f"joint-cost: {joint_cost}\n"
            )
        
    plt.clf()
    plt.title(f'Seed: {args.current_seed}: Eval Done')
    plt.savefig(f'/data/exp-v0/figure/eval_{args.current_seed}.png', dpi=100)

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation"
    )
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--cook_network', type=str, required=False)
    parser.add_argument('--server_network', type=str, required=False)
    parser.add_argument('--cleaner_network', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # print(args)
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    eval_main(args)