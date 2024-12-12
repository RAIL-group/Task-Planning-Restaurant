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

EVAL_NO = 10
MAX_TASK_COOK = 10
MAX_TASK_SERVER = 10
MAX_TASK_CLEANER = 10

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
                if 'dirty' in child and child['dirty'] == 1:
                    dirty_items.append(child['assetId'])
                else:
                    clean_items.append(child['assetId'])
            if 'cookable' in child:
                if 'cooked' in child and child['cooked'] == 1:
                    cooked_items.append(child['assetId'])
                else:
                    uncooked_items.append(child['assetId'])
    
    tasks_cook = taskplan_multi.pddl.task_distribution.tasks_for_cook(uncooked_items)
    tasks_server = taskplan_multi.pddl.task_distribution.tasks_for_server(clean_items)
    tasks_cleaner = taskplan_multi.pddl.task_distribution.available_tasks_for_cleaner(dirty_items, items_on_bus)

    # print(len(tasks_cook))
    # print(len(tasks_server))
    # print(len(tasks_cleaner))

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


def eval_main(args):
    # Get restaurant data for a send and extract initial object states
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
    agents = ['cook_bot', 'cleaner_bot', 'server_bot']
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    active_agent = random.choice(agents)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=args.current_seed, agents=agents, active=active_agent)
    task_sequence = get_tasks(restaurant)
    prep_state = ant_planner.get_prepared_state(restaurant, task_sequence, n_iterations=1000)
    # if len(task_sequence) >= MAX_TASK:
    #     sampled_task = random.sample(task_sequence, MAX_TASK)
    # else:
    #     rem = MAX_TASK - len(task_sequence)
    #     sampled_task = random.sample(task_sequence, rem)
    #     sampled_task.extend(task_sequence)
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    image_graph = taskplan_multi.utils.get_image_for_data(whole_graph)
    plt.clf()
    plt.imshow(image_graph)
    plt.savefig(f'{args.save_dir}/graph_{args.current_seed}.png', dpi=600)
    init_state = restaurant.get_current_object_state()
    # raise NotImplementedError
    # task_sequence = [
    #     ('cleaner_bot', taskplan_multi.pddl.task.clean_something('bowl1'))
    # ]
    for i in range(EVAL_NO):
        if len(task_sequence) == 0:
            break
        # init_state = restaurant.randomize_objects_state(randomness=[random.randint(0, 1), random.choice(random_choices), random.choice(random_choices)])
        restaurant.update_container_props(init_state)
        myopic_planner.get_seq_cost(args, restaurant, task_sequence, i)
        restaurant.update_container_props(init_state)
        ant_planner.get_seq_cost(args, restaurant, task_sequence, i, ap_concern='joint')
        # restaurant.update_container_props(init_state)
        # ant_planner.get_seq_cost(args, restaurant, task_sequence, i, ap_concern='self')
        # restaurant.update_container_props(init_state)
        # ant_planner.get_seq_cost(args, restaurant, task_sequence, i, ap_concern='max')
        # restaurant.update_container_props(init_state)
        # ant_planner.get_seq_cost(args, restaurant, task_sequence, i, ap_concern='min')
        random.shuffle(task_sequence)

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