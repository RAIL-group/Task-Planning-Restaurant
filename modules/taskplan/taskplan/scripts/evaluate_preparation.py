import os
import time
import torch
import random
import numpy as np
import taskplan
import copy
import matplotlib.pyplot as plt
import argparse

TASKS_PER_SEQUENCE = 20
MAX_EVAL = 1000
TASKS_IN_DISTR = 100


def write_props_from_Plan(proc_data, planner, selected_tasks, init_state,
                          logfile, sequence_num):
    # curr_state = copy.deepcopy(init_state)
    proc_data.update_container_props(init_state)
    for idx, task in enumerate(selected_tasks):
        returned_task, curr_state, cost = planner.get_state_n_cost(
            proc_data, task)
        proc_data.update_container_props(curr_state)
        with open(logfile, "a+") as f:
            f.write(
                f" | seq: S{sequence_num}"
                f" | desc: {returned_task}"
                f" | num: T{idx+1}"
                f" | cost: {cost:0.4f}\n"
            )


def plot_state(args, proc_data, name):
    whole_graph = taskplan.utils.get_graph(proc_data)
    image_graph = taskplan.utils.graph_formatting(whole_graph)
    file_name = name + args.image_filename
    image_path = os.path.join(args.save_dir, file_name)
    graph_img = image_graph['graph_image']
    grid = proc_data.occupancy_grid
    grid_img = np.transpose(grid)
    plt.clf()
    text = f'ProcTHOR State: {name}'
    plt.title(text)
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=1000)
    axs[0].imshow(graph_img, cmap='viridis')
    axs[0].set_title('Graph')

    axs[1].imshow(grid_img, cmap='viridis')
    axs[1].set_title('Grid')

    locs = proc_data.get_container_pos_list()
    robot = proc_data.agent['position']
    locs.append(("robot", robot))
    dist = np.zeros((len(locs)+1, len(locs)+1))
    # print(dist)
    i = 0
    for (k1, v1) in locs:
        axs[1].text(v1[0]+1, v1[1]+1, k1, fontsize=6, color='white')
        # print(v1)
        j = 0
        for (k2, v2) in locs:
            # print(v2)
            cost = proc_data.get_cost_from_occupancy_grid(v1[0],
                                                          v1[1],
                                                          v2[0],
                                                          v2[1])
            j += 1
            dist[i, j] = cost
        i += 1
    plt.tight_layout()
    plt.savefig(image_path)


def get_tasks():
    return [
        taskplan.pddl.task.serve_water('servingtable1'),
        taskplan.pddl.task.serve_water('servingtable2'),
        taskplan.pddl.task.serve_water('servingtable3'),
        taskplan.pddl.task.serve_coffee('servingtable1'),
        taskplan.pddl.task.serve_coffee('servingtable2'),
        taskplan.pddl.task.serve_coffee('servingtable3'),
        taskplan.pddl.task.fill_coffeemachine_with_water(),
        taskplan.pddl.task.serve_sandwich('servingtable1'),
        taskplan.pddl.task.serve_sandwich('servingtable2'),
        taskplan.pddl.task.serve_sandwich('servingtable3'),
        taskplan.pddl.task.clear_surface('servingtable1'),
        taskplan.pddl.task.clear_surface('servingtable2'),
        taskplan.pddl.task.clear_surface('servingtable3'),
        taskplan.pddl.task.clear_surface('coffeemachine'),
        taskplan.pddl.task.clear_surface('countertop'),
        # taskplan.pddl.task.clean_everything(),
        taskplan.pddl.task.clean_something('bowl1'),
        taskplan.pddl.task.clean_something('bowl2'),
        taskplan.pddl.task.clean_something('cup2'),
        taskplan.pddl.task.clean_something('mug2'),
        taskplan.pddl.task.clean_something('knife2'),
        taskplan.pddl.task.clean_something('plate2'),
        taskplan.pddl.task.place_something('mug3', 'shelf3'),
        taskplan.pddl.task.place_something('cup2', 'shelf2'),
        taskplan.pddl.task.place_something('plate2', 'shelf6'),
        taskplan.pddl.task.place_something('knife2', 'shelf4'),
        taskplan.pddl.task.clean_and_place('mug3', 'shelf3'),
        taskplan.pddl.task.clean_and_place('cup2', 'shelf2'),
        taskplan.pddl.task.clean_and_place('plate2', 'shelf6'),
        taskplan.pddl.task.clean_and_place('knife2', 'shelf4'),
        taskplan.pddl.task.place_something('knife2', 'countertop'),
    ]


def evaluate_main(args):
    proc_data = taskplan.environments.restaurant.RESTAURANT(
        seed=args.current_seed)
    task_distribution = get_tasks()
    # plot_state(args, proc_data, 'Initial Map')
    # print(proc_data.containers)
    myopic_planner = taskplan.planners.myopic_planner.MyopicPlanner()
    ant_planner = taskplan.planners.anticipatory_planner.AntcipatoryPlanner(
        args)
    # Prepared State From Task Distribution using A.P
    prepared_state = ant_planner. \
        get_prepared_state(proc_data, task_distribution,
                           n_iterations=500)
    proc_data.roll_back_to_init()
    # proc_data.update_container_props(prepared_state)
    # plot_state(args, proc_data, 'Anneal_Shuffle_Prepared_Map')
    raise NotImplementedError

    logfile_cost = os.path.join(args.save_dir, 'prep_cost_' + args.logfile_name)
    logfile_ex_time = os.path.join(args.save_dir, 'prep_ext_' + args.logfile_name)

    planners = {
        'np_myopic': {
            'cost': list(),
            'ext': list(),
            'pt': list()
        },
        'ideal_prep': {
            'cost': list(),
            'ext': list(),
            'pt': list()
        },
    }

    selected_sequence = task_distribution

    for i in range(MAX_EVAL):
        random.shuffle(selected_sequence)
        start_time = time.time()
        seq_cost, seq_time = (
            myopic_planner.get_seq_cost(args, proc_data,
                                        selected_sequence,
                                        non_prepared_state,
                                        seq_num=i,
                                        prep=False, rand_state=rnd_state))
        if seq_cost is None:
            break

        pt = time.time()-start_time

        planners['np_myopic']['cost'].append(seq_cost)
        planners['np_myopic']['ext'].append(seq_time)
        planners['np_myopic']['pt'].append(pt)

        # # N.L Prep Myopic
        proc_data.update_container_props(prepared_state)
        start_time = time.time()
        seq_cost, seq_time = (
            myopic_planner.get_seq_cost(args, proc_data,
                                        selected_sequence,
                                        prepared_state,
                                        seq_num=i,
                                        prep=True, rand_state=rnd_state))

        if seq_cost is None:
            break

        pt = time.time()-start_time

        planners['ideal_prep']['cost'].append(seq_cost)
        planners['ideal_prep']['ext'].append(seq_time)
        planners['ideal_prep']['pt'].append(pt)

        with open(logfile_cost, "a+") as f:
            f.write(f"eval: {i}"
                    f"| np_myopic: {planners['np_myopic']['cost'][i]}"
                    f"| ideal_prep: {planners['ideal_prep']['cost'][i]}\n")
        with open(logfile_ex_time, "a+") as f:
            f.write(f"eval: {i}"
                    f"| np_myopic: {planners['np_myopic']['ext'][i]}"
                    f"| ideal_prep: {planners['ideal_prep']['ext'][i]}\n")
        # with open(logfile_plan_time, "a+") as f:
        #     f.write(f"eval: {i}"
        #             f"| np_myopic: {planners['np_myopic']['pt'][i]}"
        #             f"| prep_myopic: {planners['prep_myopic']['pt'][i]}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation using ProcTHOR for LOMDP"
    )
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--save_dir', type=str, required=False)
    parser.add_argument('--logfile_name', type=str, default='logfile.txt')
    parser.add_argument('--image_filename', type=str, required=False)
    parser.add_argument(
        '--network_file', type=str, required=False,
        help='Directory with the name of the conditional gcn model')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    evaluate_main(args)
