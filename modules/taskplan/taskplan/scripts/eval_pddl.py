import os
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl

import taskplan


def evaluate_main(args):
    # Get environment data
    # TODO make generation possible for variable seed
    restaurant = taskplan.environments.restaurant.RESTAURANT(seed=1)

    # Get graph data and robot start pose from restaurant data
    grid = restaurant.grid
    graph = taskplan.utils.get_graph(restaurant)
    start = taskplan.utils.get_robot_pose(restaurant)
    whole_graph = taskplan.utils.graph_formatting(graph)

    # Initialize the PartialMap with whole graph
    partial_map = taskplan.core.PartialMap(whole_graph, grid)

    # initialize pddl related contents
    pddl = {}
    pddl['domain'] = taskplan.pddl.domain.get_domain()
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant)
    pddl['planner'] = 'ff-astar2'  # 'max-astar'

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])

    # append to robot_pose list
    known_space_poses, find_start = taskplan.utils. \
        get_poses_from_plan(plan, partial_map)

    # Intialize logfile
    logfile = os.path.join(args.save_dir, args.logfile_name)
    with open(logfile, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")

    cost_str = 'pddl'

    path = [start] + known_space_poses
    dist, trajectory = taskplan.core.compute_path_cost(partial_map.grid, path)

    print(f"Planning cost: {dist}")
    with open(logfile, "a+") as f:
        # err_str = '' if did_succeed else '[ERR]'
        f.write(f"[Learn] s: {args.current_seed:4d}"
                f" | {cost_str}: {dist:0.3f}\n"
                f"  Steps: {len(path)-1:3d}\n")

    # Plot the results
    taskplan.plotting.plot_result(partial_map, whole_graph,
                                  plan, path, cost_str)
    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=800)


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation arguments for task planning in restaurant.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--image_filename', type=str, required=True)
    parser.add_argument('--logfile_name', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
