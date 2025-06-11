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
from taskplan_multi.utilities.primitives import (
    ROOMS_KEY, DOORS_KEY, OBJECTS_KEY, POLYGON, ASSET_ID, ROOM_1, ROOM_2, BOT_1, BOT_2)

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

EVAL_NO = 1
MAX_TASK_COOK = 20
MAX_TASK_SERVER = 20
MAX_TASK_CLEANER = 20
TASK_PER_SEQ = 50

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


def load_prepared_state(args):
    file_name = ''
    root = args.save_dir
    for path, _, files in os.walk(root):
        for name in files:
            if 'prep_state_learned' in name:
                file_name = os.path.join(path, name)
                break
    # print(file_name)
    # Open and read the content of the file
    with open(file_name, 'r') as file:
        file_content = file.read()

    # Convert the string content to a Python list
    data_list = ast.literal_eval(file_content)

    # Now data_list is a Python list containing your data
    # print(data_list)
    # print(type(data_list))
    return data_list

def get_tasks(restaurant):
    food_items = list()
    utensils = list()
    cooked_items = list()
    uncooked_items =  list()
    clean_items = list()
    dirty_items = list()
    items_on_bus = list()
    for container in restaurant.containers:
        children = container.get('children')
        if children is None:
            continue
        for child in children:
            if container.get('assetId') == 'bussingcart':
                items_on_bus.append(child['assetId'])
            if 'washable' in child:
                utensils.append(child['assetId'])
                if 'dirty' in child and child['dirty'] == 1:
                    dirty_items.append(child['assetId'])
                else:
                    clean_items.append(child['assetId'])
            if 'cookable' in child:
                food_items.append(child['assetId'])
                if 'cooked' in child and child['cooked'] == 1:
                    cooked_items.append(child['assetId'])
                else:
                    uncooked_items.append(child['assetId'])
    
    tasks_cook = taskplan_multi.pddl.task_distribution.tasks_for_cook(food_items)
    tasks_server = taskplan_multi.pddl.task_distribution.tasks_for_server()
    tasks_cleaner = taskplan_multi.pddl.task_distribution.tasks_for_cleaner(utensils)

    if len(tasks_cook) >= MAX_TASK_COOK:
        tasks_cook = random.sample(tasks_cook, MAX_TASK_COOK)
    else:
        rem = MAX_TASK_COOK - len(tasks_cook)
        temp = random.choices(tasks_cook, k=rem)
        tasks_cook.extend(temp)
    
    if len(tasks_server) >= MAX_TASK_SERVER:
        tasks_server = random.sample(tasks_server, MAX_TASK_SERVER)
    else:
        rem = MAX_TASK_SERVER - len(tasks_server)
        temp = random.choices(tasks_server, k=rem)
        tasks_server.extend(temp)
    
    if len(tasks_cleaner) >= MAX_TASK_CLEANER:
        tasks_cleaner = random.sample(tasks_cleaner, MAX_TASK_CLEANER)
    else:
        rem = MAX_TASK_CLEANER - len(tasks_cleaner)
        temp = random.choices(tasks_cleaner, k=rem)
        tasks_cleaner.extend(temp)
    
    tasks = list()
    for task in tasks_cook:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(('cook_bot', val))
    for task in tasks_server:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(('server_bot', val))
    for task in tasks_cleaner:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(('cleaner_bot', val))
    random.shuffle(tasks)
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
            if 'bad' in child and child['bad'] == 1:
                name = 'bad ' + name
                cl = 'red'
            if 'cooked' in child and child['cooked'] == 1:
                name = 'cooked ' + name
                cl = 'gold'
            plt.text(_x, _y + (i + 1) * 1.2, name, fontsize=5, color=cl)  # Slight offset
        plt.imshow(img, cmap='gray_r', alpha=0.5)
    # plt.axis('off')  # Hides the axis
    plt.title(title)
    plt.savefig(f'{args.save_dir}/{image_name}_{args.current_seed}.png', dpi=600)


def tasks_distr():
    tasks = list()
    tasks.append(
        ('tire_bot', '(and (not (is-bad tire1)) (is-at tire1 tirerack))')
    )
    tasks.append(
        ('tire_bot', '(and (not (is-bad tire2)) (is-at tire2 tirerack))')
    )
    tasks.append(
        ('tire_bot', '(and (not (is-bad tire3)) (is-at tire3 tirerack))')
    )
    tasks.append(
        ('mirror_bot', '(and (not (is-bad mirror1)) (is-at mirror1 mirrorrack))')
    )
    tasks.append(
        ('mirror_bot', '(and (not (is-bad mirror2)) (is-at mirror2 mirrorrack))')
    )
    tasks.append(
        ('mirror_bot', '(and (not (is-bad mirror3)) (is-at mirror3 mirrorrack))')
    )
    return tasks

def eval_main(args):
    agents = [BOT_1, BOT_2]
    # agents = [BOT_1]
    active_agent = random.choice(agents)
    domain = taskplan_multi.pddl.workshop_domain.get_domain()
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner(domain=domain)
    computed_planner = taskplan_multi.planners.computed_ap_planner.AntcipatoryPlanner(args, domain=domain)
    workshop = taskplan_multi.environments.workshop.WORKSHOP(seed=args.current_seed, agents=agents, active=active_agent)
    task_sequence = tasks_distr()
    no_prep_state = workshop.get_current_object_state()
    # print(workshop)
    plot_state(workshop, args, image_name='workshop', title='wth')
    for i in range(EVAL_NO):
        # random.shuffle(task_sequence)
        myopic_planner.get_seq_cost(args, workshop, task_sequence, i, no_prep_state=no_prep_state, prep_state=None)
        computed_planner.get_seq_cost(args, workshop, task_sequence, i, no_prep_state=no_prep_state, prep_state=None, ap_concern='self')
        # computed_planner.get_seq_cost(args, workshop, task_sequence, i, no_prep_state=no_prep_state, prep_state=None, ap_concern='joint')
        # ant_planner.get_seq_cost(args, restaurant, task_sequence, i, no_prep_state=None, prep_state=prep_state, ap_concern='self')
        # ant_planner.get_seq_cost(args, restaurant, task_sequence, i, no_prep_state=None, prep_state=prep_state, ap_concern='other')

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