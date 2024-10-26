from pddlstream.algorithms.search import solve_from_pddl
import matplotlib.pyplot as plt
import taskplan
import numpy as np
import torch
import pandas as pd
import learning
import os
import random
from skimage.morphology import erosion

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


def are_lists_equal(list1, list2):
    # Sort the dictionaries within each list
    sorted_list1 = sorted([sorted(d.items()) for d in list1])
    sorted_list2 = sorted([sorted(d.items()) for d in list2])
    # print(sorted_list1)
    # print(sorted_list2)
    # Compare the sorted lists
    return sorted_list1 == sorted_list2


def test_antplan_model_output():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net_name = 'ap_beta-v2.pt'
    eval_net = taskplan.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/restaurant/logs/v0/' + net_name,
        device=device
    )
    root = "/data/restaurant/"
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    # print(json_files)
    # raise NotImplementedError
    # json_files = ["./data/restaurant/data_training_0.csv"]
    # json_files = ["./data/antplan_procthor/true_set_training_803.csv"]
    # print(json_files)
    true_costs = list()
    exp_cost = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = "/data/restaurant/"+pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            true_costs.append(x['label'])
            # print(x.keys())
            anticipated_cost = eval_net(x)
            # print(anticipated_cost)
            # print(x['label'])
            exp_cost.append(anticipated_cost)
    # print(exp_cost)
    # print(true_costs)
    plt.clf()
    plt.scatter(true_costs, exp_cost, alpha=0.1)

    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(exp_cost))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # Labeling the axes
    plt.xlabel('True Costs')
    plt.ylabel('Learned Costs')
    plt.title('Costs Scatter Plot with Line from Origin (On Training)')
    save_file = '/data/figs/' + net_name + '-compare-2.png'
    plt.savefig(save_file, dpi=600)


def test_state(proc_data):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net_name = 'ap_beta-v1.pt'
    eval_net = taskplan.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/restaurant/logs/v0/' + net_name,
        device=device
    )
    whole_graph = taskplan.utils.get_graph(proc_data)
    anticipated_cost = eval_net(whole_graph)
    return anticipated_cost


def test_states(seed):
    save_file = '/data/figs/grid' + str(seed) + '.png'
    restaurant = taskplan.environments.restaurant.RESTAURANT(seed=seed)
    grid = restaurant.grid
    # occupied_cells = np.argwhere(grid == 1)
    plt.clf()
    plt.imshow(grid, cmap='gray_r')
    # plt.scatter(occupied_cells[:, 1], occupied_cells[:, 0], c='red', label='Occupied Cells')
    for pose in restaurant.accessible_poses:
        x, y = restaurant.accessible_poses[pose]
        plt.text(y, x, pose, fontsize=6)
    plt.savefig(save_file, dpi=1200)
    cost = test_state(restaurant)
    print(cost)


def get_tasks():
    t1 = taskplan.pddl.task_distribution.cleaning_task()
    t1.extend(taskplan.pddl.task_distribution.organizing_task())
    tasks = list()
    for task in t1:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(val)
    return tasks


def get_service_task():
    t1 = taskplan.pddl.task_distribution.service_tasks()
    tasks = list()
    for task in t1:
        key = list(task.keys())[0]
        val = task[key]
        tasks.append(val)
    return tasks


def test_pddl():
    seed = 3
    # save_file = '/data/figs/grid' + str(seed) + '.png'
    proc_data = taskplan.environments.restaurant.RESTAURANT(seed=seed)
    myopic_planner = taskplan.planners.myopic_planner.MyopicPlanner()
    service_tasks = get_service_task()
    exp_cost = myopic_planner.get_expected_cost(
        proc_data, service_tasks,
        return_alt_goals=False)
    print(exp_cost)
    task1 = taskplan.pddl.task.pour_water('jar1')
    task2 = taskplan.pddl.task.place_something('jar1', 'servingtable2')
    task3 = taskplan.pddl.task.serve_water('servingtable2', 'mug2')
    task = f'(and {task2} {task1} {task3})'
    pddl = {}
    pddl['domain'] = taskplan.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    pddl['problem'] = taskplan.pddl.problem.get_problem(proc_data, task)
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=60)
    if plan:
        for p in plan:
            print(p)
        print(cost)
        final_state = proc_data.get_final_state_from_plan(plan)
        proc_data.update_container_props(final_state[0],
                                         final_state[1],
                                         final_state[2])
        exp_cost = myopic_planner.get_expected_cost(
            proc_data, service_tasks,
            return_alt_goals=False)
        print(exp_cost)
    raise NotImplementedError
    other_tasks = get_tasks()
    exp_cost = myopic_planner.get_expected_cost(
        proc_data, other_tasks,
        return_alt_goals=False)
    print(exp_cost)


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

def run_pddl():
    # preparing pddl as input to the solver
    seed = 19
    random.seed(seed)
    plt.figure(figsize=(10, 10))
    save_file = '/data/figs/grid' + str(seed) + '.png'
    pddl = {}
    restaurant = taskplan.environments.restaurant.RESTAURANT(seed=seed)
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    plt.axis('off')  # Hides the axis
    plt.savefig(save_file, dpi=1200)
    pddl['domain'] = taskplan.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar'
    task = taskplan.pddl.task.serve_water('servingtable1', 'cup1')
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=60)
    tot_cost = 0
    if plan:
        for p in plan:
            print(p)
        print(cost)
    raise NotImplementedError
    final_state = restaurant.get_random_state()
    restaurant.update_container_props(final_state[0],
                                      final_state[1],
                                      final_state[2])
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=300)
    # print(plan)
    if plan:
        for p in plan:
            print(p)
        print(cost)
    task = taskplan.pddl.task.serve_water('servingtable2', 'mug2')
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=300)
    # print(plan)
    if plan:
        for p in plan:
            print(p)
        print(cost)
        final_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(final_state[0],
                                          final_state[1],
                                          final_state[2])
    print(restaurant.get_objects_by_container_name('servingtable2'))


if __name__ == "__main__":
    # test_antplan_model_output()
    run_pddl()
    # test_states(10)
    # test_pddl()
