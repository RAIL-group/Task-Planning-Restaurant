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

EVAL_NO = 10
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
    # tasks.append(
    #     ('cook_bot', taskplan_multi.pddl.task.make_pasta())
    # )
    return tasks

def eval_main(args):
    # Get restaurant data for a send and extract initial object states
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
    agents = ['cook_bot', 'cleaner_bot', 'server_bot']
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    active_agent = random.choice(agents)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=args.current_seed, agents=agents, active=active_agent)
    init_exp_cost = ant_planner.get_anticipated_cost(restaurant)
    plot_state(restaurant, args, image_name='no-prep-init', title=f'Not Prepared State (INIT) \n Exp Cost: {init_exp_cost}')
    
    no_prep_state = restaurant.randomize_objects_state(randomness=[random.randint(0, 1), random.choice(random_choices), random.choice(random_choices)])
    restaurant.update_container_props(no_prep_state)
    init_exp_cost = ant_planner.get_anticipated_cost(restaurant)
    plot_state(restaurant, args, image_name='no-prep-random', title=f'Not Prepared State \n Exp Cost: {init_exp_cost}')
    
    # # whole_graph = taskplan_multi.utils.get_graph(restaurant)
    # # image_graph = taskplan_multi.utils.get_image_for_data(whole_graph)
    
    # # raise NotImplementedError
    task_sequence = get_tasks(restaurant)
    prep_state = None
    prep_state = ant_planner.get_prepared_state(restaurant, task_sequence, n_iterations=1000)
    # # # # prep_state = load_prepared_state(args)
    restaurant.update_container_props(prep_state)
    prep_exp_cost = ant_planner.get_anticipated_cost(restaurant)
    plot_state(restaurant, args, image_name='prep', title=f'Prepared State \n Exp Cost: {prep_exp_cost}')
    # logfile_prep_learned = os.path.join(args.save_dir, 'prep_state_learned_0.txt')
    # with open(logfile_prep_learned, "w+") as f:
    #     f.write(json.dumps(prep_state))
    # failed_tasks = list()
    # success_tasks = list()

    # for idx, item in enumerate(task_sequence):
    #     active_agent = item[0]
    #     task = item[1]
    #     restaurant.active_robot = active_agent
    #     plan, cost = (
    #         myopic_planner.get_cost_and_state_from_task(restaurant, task)
    #     )
    #     if plan is None:
    #         failed_tasks.append(item)
    #     else:
    #         success_tasks.append(item)
    
    # if len(task_sequence) > TASK_PER_SEQ:
    #     sampled_task = random.sample(task_sequence, TASK_PER_SEQ)
    # else:
    #     need = TASK_PER_SEQ - len(task_sequence)
    #     sampled_task = random.choices(task_sequence, k=need)
    #     sampled_task.extend(task_sequence)
    # assert len(sampled_task) == TASK_PER_SEQ
    # file_name = 'no_prep_oracle.txt'
    # logfile_np = os.path.join(args.save_dir, file_name)
    # file_name = 'prep_oracle.txt'
    # logfile_prep = os.path.join(args.save_dir, file_name)
    for i in range(EVAL_NO):
        restaurant.active_all = False
        # restaurant.update_container_props(no_prep_state)
        # fail_perc = myopic_planner.get_oracle_failure_ratio(restaurant, sampled_task)
        # with open(logfile_np, "a+") as f:
        #     f.write(
        #         f" | seq: S{i}"
        #         f" | failed: {fail_perc}\n"
        #     )
        # restaurant.update_container_props(prep_state)
        # fail_perc = myopic_planner.get_oracle_failure_ratio(restaurant, sampled_task)
        # with open(logfile_prep, "a+") as f:
        #     f.write(
        #         f" | seq: S{i}"
        #         f" | failed: {fail_perc}\n"
        #     )
        sampled_task = random.sample(task_sequence, TASK_PER_SEQ)
        # s_tasks = random.sample(failed_tasks, 10)
        # sampled_task.extend(s_tasks)
        myopic_planner.get_seq_cost(args, restaurant, sampled_task, i, no_prep_state=None, prep_state=prep_state)
        ant_planner.get_seq_cost(args, restaurant, sampled_task, i, no_prep_state=None, prep_state=prep_state, ap_concern='joint')
        ant_planner.get_seq_cost(args, restaurant, sampled_task, i, no_prep_state=None, prep_state=prep_state, ap_concern='self')
        ant_planner.get_seq_cost(args, restaurant, sampled_task, i, no_prep_state=None, prep_state=prep_state, ap_concern='other')
        # random.shuffle(sampled_task)

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