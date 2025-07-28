from pddlstream.algorithms.search import solve_from_pddl
import taskplan_multi
import random
import matplotlib.pyplot as plt
import os
import argparse
import time
import copy
import numpy as np
from taskplan_multi.environments.restaurant import world_to_grid
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
    grid[:, :, 0][boundary] = 0.3
    grid[:, :, 1][boundary] = 0.3
    grid[:, :, 2][boundary] = 0.3

    return grid


def plot_state(restaurant, save_path='/data/figs/data-grid.png', title='None'):
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
            if 'empty' in child and child['empty'] == 1:
                name = 'empty ' + name
                cl = 'red'
            plt.text(_x, _y + (i + 1) * 1.2, name, fontsize=5, color=cl)  # Slight offset
        plt.imshow(img, cmap='gray_r', alpha=0.5)
    # plt.axis('off')  # Hides the axis
    plt.title(title)
    plt.savefig(save_path, dpi=300)


def plot_state_plain(restaurant):
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    plt.clf()
    for agent in restaurant.agent_list:
        _x, _y = world_to_grid(
            restaurant.restaurant[agent]['position']['x'], restaurant.restaurant[agent]['position']['z'],
            restaurant.grid_min_x, restaurant.grid_min_z, restaurant.grid_res)
        assetId = restaurant.restaurant[agent]['name'].split('_')[0]
        if 'cook' in assetId:
            c = 'fuchsia'
        if 'server' in assetId:
            c = 'blueviolet'
        if 'cleaner' in assetId:
            c = 'teal'
        plt.text(_x, _y, assetId, fontsize=5, color=c)

    for container in restaurant.containers:
        _x, _y = world_to_grid(
            container['position']['x'], container['position']['z'],
            restaurant.grid_min_x, restaurant.grid_min_z, restaurant.grid_res)
        assetId = container['assetId']
        plt.text(_x, _y, assetId, fontsize=5, color='blue')
        children = container.get('children', [])
        for i, child in enumerate(children):
            name = child['assetId']
            cl = 'green'
            if 'dirty' in child and child['dirty'] == 1:
                name = 'dirty ' + name
                cl = 'red'
            if 'empty' in child and child['empty'] == 1:
                name = 'empty ' + name
                cl = 'red'
            plt.text(_x, _y + (i + 1) * 1.2, name, fontsize=3, color=cl)  # Slight offset
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    # plt.axis('off')  # Hides the axis
    # plt.title(title)
    # plt.savefig(save_path, dpi=300)

def plot_task_plan(plan, restaurant, tot_cost, num=0):
    move_plans = [p for p in plan if p.name == "move"]
    cook_moves = [(move.args[1], move.args[2]) for move in move_plans if move.args[0] == 'cook_bot']
    server_moves = [(move.args[1], move.args[2]) for move in move_plans if move.args[0] == 'server_bot']
    cleaner_moves = [(move.args[1], move.args[2]) for move in move_plans if move.args[0] == 'cleaner_bot']
    
    cook_move_poses = list()
    server_move_poses = list()
    cleaner_move_poses = list()
    
    for move in cook_moves:
        src, target = (move)
        if src == 'base_cook_bot':
            pos1 = restaurant.accessible_poses['base_cook_bot']
        else:
            pos1 = restaurant.accessible_poses[src]

        if target == 'base_cook_bot':
            pos2 = restaurant.accessible_poses['base_cook_bot']
        else:
            pos2 = restaurant.accessible_poses[target]
        
        cook_move_poses.append((pos1, pos2))
    
    for pos in cook_move_poses:
        src, target = (pos)
        occupancy_grid = gridmap.utils.inflate_grid(restaurant.grid, 1)
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            occupancy_grid, start = [src[0], src[1]], )
        cost = cost_grid[target[0], target[1]]
        did_plan, path = get_path([target[0], target[1]])
        path_points = [(path[0][idx], path[1][idx])
                    for idx in range(len(path[0]))]
        path_line = geometry.LineString(path_points)
        x, y = path_line.xy
        plt.plot(x, y, color='fuchsia')
    

    for move in server_moves:
        src, target = (move)
        if src == 'base_server_bot':
            pos1 = restaurant.accessible_poses['base_server_bot']
        else:
            pos1 = restaurant.accessible_poses[src]

        if target == 'base_server_bot':
            pos2 = restaurant.accessible_poses['base_server_bot']
        else:
            pos2 = restaurant.accessible_poses[target]
        
        server_move_poses.append((pos1, pos2))
    
    for pos in server_move_poses:
        src, target = (pos)
        occupancy_grid = gridmap.utils.inflate_grid(restaurant.grid, 1)
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            occupancy_grid, start = [src[0], src[1]], )
        cost = cost_grid[target[0], target[1]]
        did_plan, path = get_path([target[0], target[1]])
        path_points = [(path[0][idx], path[1][idx])
                    for idx in range(len(path[0]))]
        path_line = geometry.LineString(path_points)
        x, y = path_line.xy
        plt.plot(x, y, color='blueviolet')
    
    for move in cleaner_moves:
        src, target = (move)
        if src == 'base_cleaner_bot':
            pos1 = restaurant.accessible_poses['base_cleaner_bot']
        else:
            pos1 = restaurant.accessible_poses[src]

        if target == 'base_cleaner_bot':
            pos2 = restaurant.accessible_poses['base_cleaner_bot']
        else:
            pos2 = restaurant.accessible_poses[target]
        
        cleaner_move_poses.append((pos1, pos2))
    
    for pos in cleaner_move_poses:
        src, target = (pos)
        occupancy_grid = gridmap.utils.inflate_grid(restaurant.grid, 1)
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            occupancy_grid, start = [src[0], src[1]], )
        cost = cost_grid[target[0], target[1]]
        did_plan, path = get_path([target[0], target[1]])
        path_points = [(path[0][idx], path[1][idx])
                    for idx in range(len(path[0]))]
        path_line = geometry.LineString(path_points)
        x, y = path_line.xy
        plt.plot(x, y, color='teal')
    
    save_path = f'/data/figs/data-after-task-{num}.png'
    plt.title(f'Cost:{tot_cost:0.2f}')
    plt.savefig(save_path, dpi=300)


# def run_pddl(args):
#     # preparing pddl as input to the solver
#     seed = 0
#     pddl = {}
#     save_file = '/data/figs/data-init-state-0.png'  
#     random.seed(seed)
#     restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cook_bot')
#     myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
#     ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
#     plot_state(restaurant, save_path = save_file, title='Grid Map')
#     tasks_cook = [
#         taskplan_multi.pddl.task.prep_item('cereal', 'bowl', 'stove')
#     ]
#     tasks_server = [
#         taskplan_multi.pddl.task.serve_item('cereal', 'bowl', 'servingtable1')
#     ]
#     tasks_cleaner = [
#         taskplan_multi.pddl.task.clean_both_item('mug')
#     ]
#     task = tasks_cook[0]
#     plan, cost = (myopic_planner.get_cost_and_state_from_task(restaurant, task))
#     # plan, cost = (ant_planner.get_anticipatory_plan(restaurant, task, last_task=False, plan_only=True))
#     if plan:
#         plot_state_plain(restaurant)
#         plot_task_plan(plan, restaurant, cost, num=1)
    
#     new_state = restaurant.get_final_state_from_plan(plan)
#     restaurant.update_container_props(new_state)
#     task = tasks_cleaner[0]
#     plan, cost = (myopic_planner.get_cost_and_state_from_task(restaurant, task))
    
#     if plan:
#         plot_state_plain(restaurant)
#         plot_task_plan(plan, restaurant, cost, num=2)
    
#     new_state = restaurant.get_final_state_from_plan(plan)
#     restaurant.update_container_props(new_state)
#     task = tasks_server[0]
#     plan, cost = (myopic_planner.get_cost_and_state_from_task(restaurant, task))
    
#     if plan:
#         plot_state_plain(restaurant)
#         plot_task_plan(plan, restaurant, cost, num=3)
    
#     plt.title(f'Myopic')
#     plt.savefig(os.path.join(args.output_image_file), dpi=2000)
#     raise NotImplementedError

def run_pddl(args):
    # preparing pddl as input to the solver
    seed = 0
    pddl = {}
    save_file = '/data/figs/data-init-state-0.png'  
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cook_bot')
    init_state = copy.deepcopy(restaurant.get_current_object_state())
    
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
    
    tasks_cook = [
        taskplan_multi.pddl.task.prep_item('cereal', 'bowl', 'stove')
    ]
    tasks_server = [
        taskplan_multi.pddl.task.serve_item('cereal', 'bowl', 'servingtable1')
    ]
    tasks_cleaner = [
        taskplan_multi.pddl.task.clean_both_item('mug')
    ]

    restaurant.update_container_props(init_state)
    plot_state(restaurant, save_path = save_file, title='Grid Map')
    task = tasks_cook[0]
    plan, cost = (myopic_planner.get_cost_and_state_from_task(restaurant, task))
    if plan:
        plot_state_plain(restaurant)
        plot_task_plan(plan, restaurant, cost, num=1)
    
    new_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(new_state)
    task = tasks_cleaner[0]
    plan, cost = (myopic_planner.get_cost_and_state_from_task(restaurant, task))
    
    if plan:
        plot_state_plain(restaurant)
        plot_task_plan(plan, restaurant, cost, num=2)
    
    new_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(new_state)
    task = tasks_server[0]
    plan, cost = (myopic_planner.get_cost_and_state_from_task(restaurant, task))
    
    if plan:
        plot_state_plain(restaurant)
        plot_task_plan(plan, restaurant, cost, num=3)
    
    # plt.title(f'Myopic')
    # plt.savefig(os.path.join(args.output_image_file), dpi=2000)

    # plt.clf()

    restaurant.update_container_props(init_state)
    # save_file = '/data/figs/data-init-state-11.png'  
    # plot_state(restaurant, save_path = save_file, title='Grid vap')
    plot_state_plain(restaurant)
    task = tasks_cook[0]
    plan, cost = (ant_planner.get_anticipatory_plan(restaurant, task, last_task=False, plan_only=True))
    if plan:
        print(plan)
        plot_task_plan(plan, restaurant, cost, num=11)
    
    new_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(new_state)
    plot_state_plain(restaurant)
    task = tasks_cleaner[0]
    plan, cost = (ant_planner.get_anticipatory_plan(restaurant, task, last_task=False, plan_only=True))
    if plan:
        print(plan)
        plot_task_plan(plan, restaurant, cost, num=12)
    
    new_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(new_state)
    plot_state_plain(restaurant)
    task = tasks_server[0]
    plan, cost = (ant_planner.get_anticipatory_plan(restaurant, task, last_task=True, plan_only=True))
    if plan:
        print(plan)
        plot_task_plan(plan, restaurant, cost, num=13)
    

    restaurant.update_container_props(init_state)
    # save_file = '/data/figs/data-init-state-11.png'  
    # plot_state(restaurant, save_path = save_file, title='Grid vap')
    ant_planner.concern == 'self'
    plot_state_plain(restaurant)
    task = tasks_cook[0]
    plan, cost = (ant_planner.get_anticipatory_plan(restaurant, task, last_task=False, plan_only=True))
    if plan:
        print(plan)
        plot_task_plan(plan, restaurant, cost, num=21)
    
    new_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(new_state)
    plot_state_plain(restaurant)
    task = tasks_cleaner[0]
    plan, cost = (ant_planner.get_anticipatory_plan(restaurant, task, last_task=False, plan_only=True))
    if plan:
        print(plan)
        plot_task_plan(plan, restaurant, cost, num=22)
    
    new_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(new_state)
    plot_state_plain(restaurant)
    task = tasks_server[0]
    plan, cost = (ant_planner.get_anticipatory_plan(restaurant, task, last_task=True, plan_only=True))
    if plan:
        print(plan)
        plot_task_plan(plan, restaurant, cost, num=23)
    
    # plt.title(f'Proactive A.P')
    # plt.savefig(os.path.join(args.output_image_file), dpi=2000)
    raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TAMP Example Planner.."
    )
    parser.add_argument("--output_image_file", type=str, default="/results/")
    parser.add_argument('--cook_network', type=str, required=False)
    parser.add_argument('--server_network', type=str, required=False)
    parser.add_argument('--cleaner_network', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    args = parser.parse_args()
    start_time = time.time()
    run_pddl(args)
    # run_pddl_anticip(args) 
    # draw_map(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.2f} seconds to finish.")