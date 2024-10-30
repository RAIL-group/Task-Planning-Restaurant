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

def run_pddl(args):
    # preparing pddl as input to the solver
    seed = 23
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed)
    object_state = restaurant.get_current_object_state()
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    for container_name, container_pos in restaurant.get_container_pos_list():
        _x, _z = world_to_grid(container_pos['x'], container_pos['z'],
                                         restaurant.grid_min_x,
                                         restaurant.grid_min_z, restaurant.grid_res)
        plt.text(_x + 2, _z + 2, container_name, color='black', fontsize=6, rotation=45)

    for obj_state in object_state:
        _x, _z = world_to_grid(obj_state['position']['x'], obj_state['position']['z'],
                                         restaurant.grid_min_x,
                                         restaurant.grid_min_z, restaurant.grid_res)
        plt.scatter(_x, _z, c='blue')
        plt.text(_x, _z, obj_state['assetId'], color='black', fontsize=6, rotation=45)
        

    tall_x, tall_z = world_to_grid(restaurant.agent_tall['position']['x'], restaurant.agent_tall['position']['z'],
                                         restaurant.grid_min_x,
                                         restaurant.grid_min_z, restaurant.grid_res)
    plt.text(tall_x, tall_z, 'tall robot', color='red', fontsize=6, rotation=45)
    tiny_x, tiny_z = world_to_grid(restaurant.agent_tiny['position']['x'], restaurant.agent_tiny['position']['z'],
                                         restaurant.grid_min_x,
                                         restaurant.grid_min_z, restaurant.grid_res)
    plt.text(tiny_x, tiny_z, 'tiny robot', color='green', fontsize=6, rotation=45)
    plt.scatter(tall_x, tall_z, c='red', s=100)
    plt.scatter(tiny_x, tiny_z, c='green', s=100)

    # print(restaurant.containers)
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar'
    # task = taskplan_multi.pddl.task.move_robot('agent_tall', 'servingtable1')
    task = taskplan_multi.pddl.task.place_something('bowl2', 'countertop')
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=60)
    # tot_cost = 0
    if plan:
        for p in plan:
            print(p)
        print(cost)
        move_plans = [p for p in plan if p.name == "move"]
        move_poses = list()
        for move in move_plans:
            if move.args[1] == 'init_tall':
                pos1 = restaurant.agent_tall['position']
            elif move.args[1] == 'init_tiny':
                pos1 = restaurant.agent_tiny['position']
            else:
                pos1 = restaurant.get_container_pos(move.args[1])
            if move.args[2] == 'init_tall':
                pos2 = restaurant.agent_tall['position']
            elif move.args[2] == 'init_tiny':
                pos2 = restaurant.agent_tiny['position']
            else:
                pos2 = restaurant.get_container_pos(move.args[2])
            _x1, _z1 = world_to_grid(pos1['x'], pos1['z'],
                                         restaurant.grid_min_x,
                                         restaurant.grid_min_z, restaurant.grid_res)
            _x2, _z2 = world_to_grid(pos2['x'], pos2['z'],
                                         restaurant.grid_min_x,
                                         restaurant.grid_min_z, restaurant.grid_res)
            move_poses.append(((_x1, _z1), (_x2, _z2)))
        print(move_poses)
        for pos in move_poses:
            src, target = (pos)
            print(src)
            print(target)
            _, get_path = gridmap.planning.compute_cost_grid_from_position(
                grid, [src[0], src[1]])
            did_plan, path = get_path([target[0], target[1]])
            print(path)
            print(did_plan)
            path_points = [(path[0][idx], path[1][idx])
                        for idx in range(len(path[0]))]
            print(path_points)
            path_line = geometry.LineString(path_points)
            x, y = path_line.xy
            plt.plot(x, y, color='orange')
        
    plt.savefig(os.path.join(args.output_image_file), dpi=2000)
    # raise NotImplementedError
    # final_state = restaurant.get_random_state()
    # restaurant.update_container_props(final_state[0],
    #                                   final_state[1],
    #                                   final_state[2])
    # pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    # plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
    #                              max_planner_time=300)
    # # print(plan)
    # if plan:
    #     for p in plan:
    #         print(p)
    #     print(cost)
    # task = taskplan.pddl.task.serve_water('servingtable2', 'mug2')
    # pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    # plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
    #                              max_planner_time=300)
    # # print(plan)
    # if plan:
    #     for p in plan:
    #         print(p)
    #     print(cost)
    #     final_state = restaurant.get_final_state_from_plan(plan)
    #     restaurant.update_container_props(final_state[0],
    #                                       final_state[1],
    #                                       final_state[2])
    # print(restaurant.get_objects_by_container_name('servingtable2'))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TAMP Example Planner.."
    )
    parser.add_argument("--output_image_file", type=str, default="/results/")
    args = parser.parse_args()
    start_time = time.time()
    run_pddl(args) 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.2f} seconds to finish.")