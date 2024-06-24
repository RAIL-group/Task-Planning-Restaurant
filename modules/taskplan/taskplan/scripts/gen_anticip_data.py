import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import taskplan
import copy
from collections import Counter
import os

MAX_DATA_PER_MAP = 200

# cup12, mug123, knife12, spreads, plate12, bowl12 


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


def gen_data_main(args):
    # Get restaurant data for a send and extract initial object states
    proc_data = taskplan.environments.restaurant.RESTAURANT(
        seed=args.current_seed)
    whole_graph = taskplan.utils.get_graph(proc_data)
    file_name = 'non_qualified_logs/log.txt'
    logfile = os.path.join(args.save_dir, file_name)
    map_counter = 0
    myopic_planner = taskplan.planners.myopic_planner.MyopicPlanner()
    tasks = get_tasks()
    alternates = list()
    while (map_counter < MAX_DATA_PER_MAP):
        alt_states, exp_cost = myopic_planner.get_expected_cost(
            proc_data, tasks,
            return_alt_goals=True)
        if exp_cost is None:
            with open(logfile, "a+") as f:
                f.write(
                    f" | State: {proc_data.get_current_object_poses()}"
                    f" | Map No: {map_counter} \n"
                )
            break
        whole_graph['label'] = exp_cost
        taskplan.utils.write_datum_to_file(args, whole_graph, map_counter)
        map_counter += 1
        if alt_states or alternates:
            alternates.extend(alt_states)
            object_state = alternates.pop()
            proc_data.update_container_props(object_state)
            whole_graph = taskplan.utils.get_graph(proc_data)
        else:
            break

    image_graph = taskplan.utils.graph_formatting(whole_graph)
    img = image_graph['graph_image']
    title = f'Success Gen: {map_counter}'
    plt.clf()
    plt.title(title)
    plt.imshow(img)
    plt.savefig(f'{args.save_dir}data_completion_logs/{args.data_file_base_name}_{args.current_seed}.png')


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation"
    )
    parser.add_argument('--current_seed', type=int)
    parser.add_argument('--data_file_base_name', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.current_seed)
    np.random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    gen_data_main(args)
