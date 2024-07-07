import os
import time
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pddlstream.algorithms.search import solve_from_pddl

import taskplan
from taskplan.planners.planner import ClosestActionPlanner
# from lsp_tp.planners import LearnedPlanner
# from lsp_tp.utils.pddl_helper import get_learning_informed_plan


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
    partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)

    # initialize pddl related contents
    pddl = {}
    pddl['domain'] = taskplan.pddl.domain.get_domain()
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant)
    pddl['planner'] = 'ff-astar2'  # 'max-astar'

    # # 5: 'fridge', 6: 'garbagecan', 8: 'countertop',
    # # 9: 'sofa', 10: 'chair', 11: 'diningtable'
    # init_subgoals = [5, 6, 8, 9, 10, 11]

    if args.logfile_name == 'naive_logfile.txt':
        plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
    # elif args.logfile_name == 'pddl_learned_logfile.txt':
    #     plan, cost = get_learning_informed_plan(
    #         pddl=pddl, partial_map=partial_map,
    #         subgoals=init_subgoals, robot_pose=start, args=args)

    # append to robot_pose list
    known_space_poses, find_start = taskplan.utils. \
        get_poses_from_plan(plan, partial_map)

    # use the latest robot pose of known space plan before finding object
    if find_start == -1:
        init_robot_pose = start
    else:
        init_robot_pose = known_space_poses[find_start]

    # Initialize what object to find
    partial_map.target_obj = taskplan.utils. \
        get_object_to_find_from_plan(plan, partial_map)

    # Intialize logfile
    logfile = os.path.join(args.save_dir, args.logfile_name)
    with open(logfile, "a+") as f:
        f.write(f"LOG: {args.current_seed}\n")
    if args.logfile_name == 'naive_logfile.txt':
        planner = ClosestActionPlanner(args, partial_map, verbose=True)
        cost_str = 'naive_lsp'
    # elif args.logfile_name == 'pddl_learned_logfile.txt':
    #     planner = LearnedPlanner(args, partial_map, verbose=True)
    #     cost_str = 'learned_lsp'

    if partial_map.target_obj is not None:
        planning_loop = taskplan.planners.planning_loop.PlanningLoop(
            partial_map=partial_map, robot=init_robot_pose, args=args,
            verbose=True)
        # # update the subgoals of the planning loop
        # planning_loop.subgoals = init_subgoals.copy()
        # print(planning_loop.subgoals)
        # for k in whole_graph['nodes']:
        #     print(k, whole_graph['nodes'][k]['id'])

        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            s_time = time.time()
            planner.update(
                step_data['graph'],
                step_data['subgoals'],
                step_data['robot_pose'])
            print(f"Time taken to update: {time.time() - s_time}")

            # Compute the next subgoal and set to the planning loop
            s_time = time.time()
            chosen_subgoal = planner.compute_selected_subgoal()
            print(f"Time taken to choose subgoal: {time.time() - s_time}")
            planning_loop.set_chosen_subgoal(chosen_subgoal)

        # Generate the complete path combining move and find actions
        if start == init_robot_pose:
            path = planning_loop.robot + known_space_poses
        else:
            path = [start] + known_space_poses[:find_start] + \
                planning_loop.robot
            if len(known_space_poses) - len(known_space_poses[:find_start]) > 1:
                path += known_space_poses[find_start+1:]
        args.robot_path = planning_loop.robot
        plot_args = args
    else:
        path = [start] + known_space_poses
        plot_args = None

    dist, trajectory = taskplan.core.compute_path_cost(partial_map.grid, path)

    print(f"Planning cost: {dist}")
    with open(logfile, "a+") as f:
        # err_str = '' if did_succeed else '[ERR]'
        f.write(f"[Learn] s: {args.current_seed:4d}"
                f" | {cost_str}: {dist:0.3f}\n"
                f"  Steps: {len(path)-1:3d}\n")

    # Plot the results
    taskplan.plotting.plot_result(partial_map, whole_graph,
                                  plan, path, cost_str, plot_args)
    plt.savefig(f'{args.save_dir}/{args.image_filename}', dpi=800)


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation arguments for task planning in restaurant.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--current_seed', type=int, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--image_filename', type=str, required=True)
    parser.add_argument('--logfile_name', type=str, required=True)
    # parser.add_argument('--data_file_base_name', type=str, required=True)
    # parser.add_argument('--network_file', type=str, required=True)
    # parser.add_argument('--learning_rate', type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
