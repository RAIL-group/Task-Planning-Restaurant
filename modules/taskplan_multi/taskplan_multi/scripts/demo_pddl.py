from pddlstream.algorithms.search import solve_from_pddl
import taskplan_multi
import random
import matplotlib.pyplot as plt
import os
import argparse
import time
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
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid

def plot_plan(plan, cost):
    # Add a text block
    textstr = ''
    for p in plan:
        textstr += str(p) + '\n'
    # textstr = 'This is a text block.\nYou can add multiple lines of text.'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Add labels and title
    # plt.title(f'Plan in PDDL, cost : {cost}', fontsize=6)

    # Place a text box in upper left in axes coords
    plt.text(0, .7, textstr, transform=plt.gca().transAxes, fontsize=5,
             verticalalignment='top', bbox=props)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

def run_pddl(args):
    # preparing pddl as input to the solver
    seed = 0
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cleaner_bot'], active='cleaner_bot')
    ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
    object_state = restaurant.get_current_object_state()
    print(object_state)
    # raise NotImplementedError
    random_state = restaurant.randomize_objects_state(randomness=[0, 2, 0])
    restaurant.update_container_props(random_state)
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    for container_name, container_pos in restaurant.get_container_pos_list():
        _x, _z = restaurant.accessible_poses[container_name]
        plt.text(_x + 2, _z + 2, container_name, color='black', fontsize=6, rotation=45)

    # for obj_state in object_state:
    #     _x, _z = world_to_grid(obj_state['position']['x'], obj_state['position']['z'],
    #                                      restaurant.grid_min_x,
    #                                      restaurant.grid_min_z, restaurant.grid_res)
    #     plt.scatter(_x, _z, c='blue')
    #     plt.text(_x, _z, obj_state['assetId'], color='black', fontsize=6, rotation=45)
        
    for agent in restaurant.agent_list:
        tall_x, tall_z = restaurant.accessible_poses['base_' + agent]
        plt.text(tall_x, tall_z, agent, color='red', fontsize=6, rotation=45)
        plt.scatter(tall_x, tall_z, c='red')
    # print(restaurant.containers)
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar'
    # task = taskplan_multi.pddl.task.move_robot('agent_tall', 'servingtable1')
    # task = taskplan_multi.pddl.task.place_something('cup', 'stove')
    # task = taskplan_multi.pddl.task.clean_something('mug')
    # task = taskplan_multi.pddl.task.clear_surface('bussingcart', 'mug')
    # task = taskplan_multi.pddl.task.clear_surface('bussingcart', 'mug')
    task = taskplan_multi.pddl.task.clear_surface('bussingcart', 'mug')
    # t2 = taskplan_multi.pddl.task.clear_surface('bussingcart', 'bowl')
    # t3 = taskplan_multi.pddl.task.clear_surface('bussingcart', 'pan')
    # t3 = taskplan_multi.pddl.task.clear_surface('bussingcart', 'milk')
    # task = f'(and {t1} {t2} {t3})'
    # task = taskplan_multi.pddl.task.serve_oats('milk', 'servingtable1')
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=60)
    # tot_cost = 0
    print(plan)
    print(plan_cost)
    raise NotImplementedError
    if plan:
        for p in plan:
            print(p)
        move_plans = [p for p in plan if p.name == "move"]
        move_poses = list()
        for move in move_plans:
            if move.args[1] == 'init_tall':
                pos1 = restaurant.accessible_poses['init_tall']
            elif move.args[1] == 'init_tiny':
                pos1 = restaurant.accessible_poses['init_tiny']
            else:
                pos1 = restaurant.accessible_poses[move.args[1]]
            if move.args[2] == 'init_tall':
                pos2 = restaurant.accessible_poses['agent_tall']
            elif move.args[2] == 'init_tiny':
                pos2 = restaurant.accessible_poses['init_tiny']
            else:
                pos2 = restaurant.accessible_poses[move.args[2]]
            
            move_poses.append((pos1, pos2))
        for pos in move_poses:
            src, target = (pos)
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                restaurant.grid, start = [src[0], src[1]], )
            cost = cost_grid[target[0], target[1]]
            print(cost)
            did_plan, path = get_path([target[0], target[1]])
            path_points = [(path[0][idx], path[1][idx])
                        for idx in range(len(path[0]))]
            path_line = geometry.LineString(path_points)
            x, y = path_line.xy
            plt.plot(x, y, color='orange')
    
    # plt.subplot(122)
    # plt.title('Plan in PDDL')
    plot_plan(plan, plan_cost)
    plt.title(f'Task1: \n move dirty cup to dishwasher (Tiny Robot) \n Cost: {plan_cost}')
    # raise NotImplementedError
    final_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(final_state)
    restaurant.active_robot = 'agent_tall'
    task = taskplan_multi.pddl.task.place_something('mug2', 'cabinet')
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plt.subplot(122)
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    for container_name, container_pos in restaurant.get_container_pos_list():
        _x, _z = restaurant.accessible_poses[container_name]
        plt.text(_x + 2, _z + 2, container_name, color='black', fontsize=6, rotation=45)
        

    tall_x, tall_z = restaurant.accessible_poses['init_tall']
    plt.text(tall_x, tall_z, 'tall robot', color='red', fontsize=6, rotation=45)
    tiny_x, tiny_z = restaurant.accessible_poses['init_tiny']
    plt.text(tiny_x, tiny_z, 'tiny robot', color='green', fontsize=6, rotation=45)
    plt.scatter(tall_x, tall_z, c='red')
    plt.scatter(tiny_x, tiny_z, c='green')
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=60)
    if plan:
        for p in plan:
            print(p)
        print(cost)
        move_plans = [p for p in plan if p.name == "move"]
        move_poses = list()
        for move in move_plans:
            if move.args[1] == 'init_tall':
                pos1 = restaurant.accessible_poses['init_tall']
            elif move.args[1] == 'init_tiny':
                pos1 = restaurant.accessible_poses['init_tiny']
            else:
                pos1 = restaurant.accessible_poses[move.args[1]]
            if move.args[2] == 'init_tall':
                pos2 = restaurant.accessible_poses['agent_tall']
            elif move.args[2] == 'init_tiny':
                pos2 = restaurant.accessible_poses['init_tiny']
            else:
                pos2 = restaurant.accessible_poses[move.args[2]]
            
            move_poses.append((pos1, pos2))
        for pos in move_poses:
            src, target = (pos)
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                restaurant.grid, start = [src[0], src[1]], )
            cost = cost_grid[target[0], target[1]]
            print(cost)
            did_plan, path = get_path([target[0], target[1]])
            path_points = [(path[0][idx], path[1][idx])
                        for idx in range(len(path[0]))]
            path_line = geometry.LineString(path_points)
            x, y = path_line.xy
            plt.plot(x, y, color='orange')
    
    # plt.subplot(122)
    # plt.title('Plan in PDDL')
    if plan is None:
        plt.title(f'Task2: \n move clean mug to cabinet (Tall Robot) \n No Plan Cost: INF')
    else:
        plot_plan(plan, cost)
        plt.title(f'Task2: \n move clean mug to cabinet (Tall Robot) \n Cost: {plan_cost}')
    
    plt.savefig(os.path.join(args.output_image_file), dpi=2000)



def run_pddl_anticip(args):
    # preparing pddl as input to the solver
    seed = 3
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, active='tiny')
    ant_planner = taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner(args)
    object_state = restaurant.get_current_object_state()
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    for container_name, container_pos in restaurant.get_container_pos_list():
        _x, _z = restaurant.accessible_poses[container_name]
        plt.text(_x + 2, _z + 2, container_name, color='black', fontsize=6, rotation=45)

    # for obj_state in object_state:
    #     _x, _z = world_to_grid(obj_state['position']['x'], obj_state['position']['z'],
    #                                      restaurant.grid_min_x,
    #                                      restaurant.grid_min_z, restaurant.grid_res)
    #     plt.scatter(_x, _z, c='blue')
    #     plt.text(_x, _z, obj_state['assetId'], color='black', fontsize=6, rotation=45)
        

    tall_x, tall_z = restaurant.accessible_poses['init_tall']
    plt.text(tall_x, tall_z, 'tall robot', color='red', fontsize=6, rotation=45)
    tiny_x, tiny_z = restaurant.accessible_poses['init_tiny']
    plt.text(tiny_x, tiny_z, 'tiny robot', color='green', fontsize=6, rotation=45)
    plt.scatter(tall_x, tall_z, c='red')
    plt.scatter(tiny_x, tiny_z, c='green')
    # print(restaurant.containers)
    # pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    # pddl['planner'] = 'ff-astar'
    # task = taskplan_multi.pddl.task.move_robot('agent_tall', 'servingtable1')
    task = taskplan_multi.pddl.task.place_something('cup1', 'dishwasher')
    # pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, plan_cost = ant_planner.get_ap_plan_only(restaurant, task)
    # tot_cost = 0
    if plan:
        for p in plan:
            print(p)
        # print(cost)
        move_plans = [p for p in plan if p.name == "move"]
        move_poses = list()
        for move in move_plans:
            if move.args[1] == 'init_tall':
                pos1 = restaurant.accessible_poses['init_tall']
            elif move.args[1] == 'init_tiny':
                pos1 = restaurant.accessible_poses['init_tiny']
            else:
                pos1 = restaurant.accessible_poses[move.args[1]]
            if move.args[2] == 'init_tall':
                pos2 = restaurant.accessible_poses['agent_tall']
            elif move.args[2] == 'init_tiny':
                pos2 = restaurant.accessible_poses['init_tiny']
            else:
                pos2 = restaurant.accessible_poses[move.args[2]]
            
            move_poses.append((pos1, pos2))
        for pos in move_poses:
            src, target = (pos)
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                restaurant.grid, start = [src[0], src[1]], )
            cost = cost_grid[target[0], target[1]]
            # print(cost)
            did_plan, path = get_path([target[0], target[1]])
            path_points = [(path[0][idx], path[1][idx])
                        for idx in range(len(path[0]))]
            path_line = geometry.LineString(path_points)
            x, y = path_line.xy
            plt.plot(x, y, color='orange')
    
    # plt.subplot(122)
    # plt.title('Plan in PDDL')
    plot_plan(plan, plan_cost)
    plt.title(f'Task1: \n move dirty cup to dishwasher (Tiny Robot) \n Cost: {plan_cost}')
    # raise NotImplementedError
    final_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(final_state)
    restaurant.active_robot = 'agent_tall'
    task = taskplan_multi.pddl.task.place_something('mug2', 'cabinet')
    # pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plt.subplot(122)
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    for container_name, container_pos in restaurant.get_container_pos_list():
        _x, _z = restaurant.accessible_poses[container_name]
        plt.text(_x + 2, _z + 2, container_name, color='black', fontsize=6, rotation=45)
        

    tall_x, tall_z = restaurant.accessible_poses['init_tall']
    plt.text(tall_x, tall_z, 'tall robot', color='red', fontsize=6, rotation=45)
    tiny_x, tiny_z = restaurant.accessible_poses['init_tiny']
    plt.text(tiny_x, tiny_z, 'tiny robot', color='green', fontsize=6, rotation=45)
    plt.scatter(tall_x, tall_z, c='red')
    plt.scatter(tiny_x, tiny_z, c='green')
    plan, plan_cost = ant_planner.get_ap_plan_only(restaurant, task)
    if plan:
        for p in plan:
            print(p)
        # print(cost)
        move_plans = [p for p in plan if p.name == "move"]
        move_poses = list()
        for move in move_plans:
            if move.args[1] == 'init_tall':
                pos1 = restaurant.accessible_poses['init_tall']
            elif move.args[1] == 'init_tiny':
                pos1 = restaurant.accessible_poses['init_tiny']
            else:
                pos1 = restaurant.accessible_poses[move.args[1]]
            if move.args[2] == 'init_tall':
                pos2 = restaurant.accessible_poses['agent_tall']
            elif move.args[2] == 'init_tiny':
                pos2 = restaurant.accessible_poses['init_tiny']
            else:
                pos2 = restaurant.accessible_poses[move.args[2]]
            
            move_poses.append((pos1, pos2))
        for pos in move_poses:
            src, target = (pos)
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                restaurant.grid, start = [src[0], src[1]], )
            cost = cost_grid[target[0], target[1]]
            # print(cost)
            did_plan, path = get_path([target[0], target[1]])
            path_points = [(path[0][idx], path[1][idx])
                        for idx in range(len(path[0]))]
            path_line = geometry.LineString(path_points)
            x, y = path_line.xy
            plt.plot(x, y, color='orange')
    
    # plt.subplot(122)
    # plt.title('Plan in PDDL')
    if plan is None:
        plt.title(f'Task2: \n move clean mug to cabinet (Tall Robot) \n No Plan Cost: INF')
    else:
        plot_plan(plan, cost)
        plt.title(f'Task2: \n move clean mug to cabinet (Tall Robot) \n Cost: {plan_cost}')
    
    plt.savefig(os.path.join(args.output_image_file), dpi=2000)

def draw_map(args):
    # preparing pddl as input to the solver
    seed = 3
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, active='tall')
    object_state = restaurant.get_current_object_state()
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Occupancy Grid")
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    for container_name, container_pos in restaurant.get_container_pos_list():
        _x, _z = restaurant.accessible_poses[container_name]
        plt.text(_x + 2, _z + 2, container_name, color='black', fontsize=6, rotation=45)

    for obj_state in object_state:
        _x, _z = world_to_grid(obj_state['position']['x'], obj_state['position']['z'],
                                         restaurant.grid_min_x,
                                         restaurant.grid_min_z, restaurant.grid_res)
        plt.scatter(_x, _z, c='blue')
        plt.text(_x, _z, obj_state['assetId'], color='black', fontsize=6, rotation=45)
        

    tall_x, tall_z = restaurant.accessible_poses['init_tall']
    plt.text(tall_x, tall_z, 'tall robot', color='red', fontsize=6, rotation=45)
    tiny_x, tiny_z = restaurant.accessible_poses['init_tiny']
    plt.text(tiny_x, tiny_z, 'tiny robot', color='green', fontsize=6, rotation=45)
    plt.scatter(tall_x, tall_z, c='red', s=100)
    plt.scatter(tiny_x, tiny_z, c='green', s=100)
    plt.subplot(122)
    plt.title("ProcGraph (Assuming the tall robot is active now)")
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    image_graph = taskplan_multi.utils.graph_formatting(whole_graph)
    plt.imshow(image_graph['graph_image'])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_image_file), dpi=2000)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TAMP Example Planner.."
    )
    parser.add_argument("--output_image_file", type=str, default="/results/")
    parser.add_argument('--tall_network', type=str, required=False)
    parser.add_argument('--tiny_network', type=str, required=False)
    args = parser.parse_args()
    start_time = time.time()
    run_pddl(args)
    # run_pddl_anticip(args) 
    # draw_map(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.2f} seconds to finish.")