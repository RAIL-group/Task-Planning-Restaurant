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

EVAL_NO = 50
MAX_TASK_COOK = 30
MAX_TASK_SERVER = 200
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


# def load_prepared_state(args):
#     file_name = ''
#     root = args.save_dir
#     for path, _, files in os.walk(root):
#         for name in files:
#             if 'prep_state_learned' in name:
#                 file_name = os.path.join(path, name)
#                 break
#     # print(file_name)
#     # Open and read the content of the file
#     with open(file_name, 'r') as file:
#         file_content = file.read()

#     # Convert the string content to a Python list
#     data_list = ast.literal_eval(file_content)

#     # Now data_list is a Python list containing your data
#     # print(data_list)
#     # print(type(data_list))
#     return data_list

def get_tasks():
    # tasks_cook = taskplan_multi.pddl.task_distribution.tasks_for_cook()
    # tasks_server = taskplan_multi.pddl.task_distribution.tasks_for_server()
    tasks_cleaner = taskplan_multi.pddl.task_distribution.tasks_for_cleaner()

    # if len(tasks_cook) >= MAX_TASK_COOK:
    #     tasks_cook = random.sample(tasks_cook, MAX_TASK_COOK)
    # else:
    #     rem = MAX_TASK_COOK - len(tasks_cook)
    #     temp = random.choices(tasks_cook, k=rem)
    #     tasks_cook.extend(temp)
    
    # if len(tasks_server) >= MAX_TASK_SERVER:
    #     tasks_server = random.sample(tasks_server, MAX_TASK_SERVER)
    # else:
    #     rem = MAX_TASK_SERVER - len(tasks_server)
    #     temp = random.choices(tasks_server, k=rem)
    #     tasks_server.extend(temp)
    
    # if len(tasks_cleaner) >= MAX_TASK_CLEANER:
    #     tasks_cleaner = random.sample(tasks_cleaner, MAX_TASK_CLEANER)
    # else:
    #     rem = MAX_TASK_CLEANER - len(tasks_cleaner)
    #     temp = random.choices(tasks_cleaner, k=rem)
    #     tasks_cleaner.extend(temp)
    
    tasks = list()
    # for task in tasks_cook:
    #     val = task[1]
    #     tasks.append(('cook_bot', val))
    # for task in tasks_server:
    #     val = task[1]
    #     tasks.append(('server_bot', val))
    for task in tasks_cleaner:
        val = task[1]
        tasks.append(('cleaner_bot', val))
    # print(len(tasks))
    selected_tasks = random.sample(tasks, TASK_PER_SEQ)
    # random.shuffle(tasks)
    return selected_tasks

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


def manual_tasks():
    tasks = list()
    tasks.append(
        ('server_bot', taskplan_multi.pddl.task.serve_pasta('servingtable2'))
    )
    tasks.append(
        ('cook_bot', taskplan_multi.pddl.task.cook_something('pasta'))
    )
    return tasks

def load_prepared_state(args):
    file_name = ''
    root = args.save_dir
    for path, _, files in os.walk(root):
        for name in files:
            if 'prep_state_' + str(args.current_seed) in name:
                file_name = os.path.join(path, name)
                datum = json.load(open(file_name))
                return datum
    return None

def eval_main(args):
    # Get restaurant data for a send and extract initial object states
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
    agents = ['cleaner_bot']
    # random_choices = [0, 0.25, 0.5, 0.75, 1]
    active_agent = random.choice(agents)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=args.current_seed, agents=agents, active=active_agent)
    # task_sequence = get_tasks()
    
    no_prep_state = copy.deepcopy(restaurant.get_current_object_state())
    init_exp_cost = ant_planner.get_anticipated_cost(restaurant)
    plot_state(restaurant, args, image_name='no-prep-init', title=f'Not Prepared State (INIT) \n Exp Cost: {init_exp_cost}')
    
    prep_state = None
    prep_state = load_prepared_state(args)
    if prep_state is not None:
        # prep_state = ant_planner.get_prepared_state(restaurant, n_iterations=2000)
        restaurant.update_container_props(prep_state)
        prep_exp_cost = ant_planner.get_anticipated_cost(restaurant)
        plot_state(restaurant, args, image_name='prep', title=f'Prepared State \n Exp Cost: {prep_exp_cost}')
    # logfile_prep_learned = os.path.join(args.save_dir, f'prep_state_{args.current_seed}.txt')
    # with open(logfile_prep_learned, "w+") as f:
    #     f.write(json.dumps(prep_state))
    # failed_tasks = list()
    # success_tasks = list()
    for i in range(EVAL_NO):
        restaurant.active_all = False
        sampled_task = get_tasks()
        myopic_planner.get_seq_cost(args, restaurant, sampled_task, i, no_prep_state=no_prep_state, prep_state=prep_state)
        ant_planner.get_seq_cost(args, restaurant, sampled_task, i, no_prep_state=no_prep_state, prep_state=prep_state, ap_concern='self')
        # ant_planner.get_seq_cost(args, restaurant, sampled_task, i, no_prep_state=no_prep_state, prep_state=prep_state, ap_concern='joint')
        # ant_planner.get_seq_cost(args, restaurant, sampled_task, i, no_prep_state=None, prep_state=prep_state, ap_concern='other')
        # random.shuffle(sampled_task)
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