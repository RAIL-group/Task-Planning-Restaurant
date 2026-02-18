import os
import torch
import random
import numpy as np
import taskplan
import argparse
import json

TASKS_PER_SEQUENCE = 40
MAX_EVAL = 200
TASKS_IN_DISTR = 100


def load_prepared_state(args):
    file_name = ''
    root = args.save_dir
    for path, _, files in os.walk(root):
        for name in files:
            if 'learned_' + str(args.current_seed) in name:
                file_name = os.path.join(path, name)
                datum = json.load(open(file_name))
                return datum


def get_tasks():
    sampled_task = list()
    for i in range(20):
        sampled_task.extend(
            random.sample(taskplan.pddl.task_distribution.service_tasks(), 1)
        )
    sampled_task.extend(
        random.sample(taskplan.pddl.task_distribution.serve_solid(), 3)
    )
    sampled_task.extend(
        random.sample(taskplan.pddl.task_distribution.cleaning_task(), 5)
    )
    sampled_task.extend(
        random.sample(taskplan.pddl.task_distribution.clear_task(), 3)
    )
    sampled_task.extend(
        random.sample(taskplan.pddl.task_distribution.organizing_task(), 9)
    )
    # tasks = list()
    # for task in t1:
    #     key = list(task.keys())[0]
    #     val = task[key]
    #     tasks.append(val)
    return sampled_task


def evaluate_main(args):
    proc_data = taskplan.environments.restaurant.RESTAURANT(
        seed=args.current_seed)
    task_distribution = get_tasks()
    # plot_state(args, proc_data, 'Initial Map')
    # print(proc_data.containers)
    myopic_planner = taskplan.planners.myopic_planner.MyopicPlanner(args=args)
    ant_planner = taskplan.planners.anticipatory_planner.AntcipatoryPlanner(
        args)
    # Prepared State From Task Distribution using A.P

    logfile_cost = os.path.join(args.save_dir, 'eval_cost_' + args.logfile_name)
    logfile_ex_time = os.path.join(args.save_dir, 'eval_ext_' + args.logfile_name)
    planners = {
        'np_myopic': {
            'cost': list(),
            'ext': list(),
        },
        'np_ap': {
            'cost': list(),
            'ext': list(),
        },
        'prep_myopic': {
            'cost': list(),
            'ext': list(),
        },
        'prep_ap': {
            'cost': list(),
            'ext': list(),
        },
    }

    object_state = proc_data.get_current_object_state()
    prepared_state = load_prepared_state(args)
    for i in range(MAX_EVAL):
        if i % 20 == 0:
            task_distribution = get_tasks()
        # N.P Myopic
        proc_data.update_container_props(object_state[0],
                                         object_state[1],
                                         object_state[2])
        seq_cost, seq_time = (
            myopic_planner.get_seq_cost(args, proc_data,
                                        task_distribution,
                                        seq_num=i,
                                        prep=False))
        if seq_cost is None:
            print(args.current_seed)
            print("None Cost: NP MYopic")
            break

        planners['np_myopic']['cost'].append(seq_cost)
        planners['np_myopic']['ext'].append(seq_time)

        # Prep Myopic

        proc_data.update_container_props(prepared_state[0],
                                         prepared_state[1],
                                         prepared_state[2])
        seq_cost, seq_time = (
            myopic_planner.get_seq_cost(args, proc_data,
                                        task_distribution,
                                        seq_num=i,
                                        prep=True))
        if seq_cost is None:
            print(args.current_seed)
            print("None Cost: Prep Myopic")
            break

        planners['prep_myopic']['cost'].append(seq_cost)
        planners['prep_myopic']['ext'].append(seq_time)

        # NP Anticipatory Planninh
        proc_data.update_container_props(object_state[0],
                                         object_state[1],
                                         object_state[2])
        seq_cost, seq_time = (
            ant_planner.get_seq_cost(args, proc_data,
                                     task_distribution,
                                     seq_num=i,
                                     prep=False))

        if seq_cost is None:
            print(args.current_seed)
            print("None Cost: NP Antcip")
            break

        planners['np_ap']['cost'].append(seq_cost)
        planners['np_ap']['ext'].append(seq_time)

        # Prepared Anticipatory Planning
        proc_data.update_container_props(prepared_state[0],
                                         prepared_state[1],
                                         prepared_state[2])
        seq_cost, seq_time = (
            ant_planner.get_seq_cost(args, proc_data,
                                     task_distribution,
                                     seq_num=i,
                                     prep=True))

        if seq_cost is None:
            print(args.current_seed)
            print("None Cost: Prep Antcip")
            break

        planners['prep_ap']['cost'].append(seq_cost)
        planners['prep_ap']['ext'].append(seq_time)
        # object_state = proc_data.get_random_state()
        random.shuffle(task_distribution)

        with open(logfile_cost, "a+") as f:
            f.write(f"eval: {i}"
                    f"| np_myopic: {planners['np_myopic']['cost'][i]}"
                    f"| np_ap: {planners['np_ap']['cost'][i]}"
                    f"| prep_myopic: {planners['prep_myopic']['cost'][i]}"
                    f"| prep_ap: {planners['prep_ap']['cost'][i]}\n")
        with open(logfile_ex_time, "a+") as f:
            f.write(f"eval: {i}"
                    f"| np_myopic: {planners['np_myopic']['ext'][i]}"
                    f"| np_ap: {planners['np_ap']['ext'][i]}"
                    f"| prep_myopic: {planners['prep_myopic']['ext'][i]}"
                    f"| prep_ap: {planners['prep_ap']['ext'][i]}\n")


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
