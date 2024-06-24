from pddlstream.algorithms.search import solve_from_pddl
import matplotlib.pyplot as plt
import taskplan
import numpy as np
import torch
import pandas as pd
import learning
import os


def test_antplan_model_output():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net_name = 'ap_beta-v0.pt'
    eval_net = taskplan.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/restaurant/logs/v0/' + net_name,
        device=device
    )
    root = "/data/restaurant/"
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training' in name and ".csv" in name:
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
    save_file = '/data/figs/' + net_name + '-compare-0.png'
    plt.savefig(save_file, dpi=600)


def run_pddl():
    # preparing pddl as input to the solver
    seed = 1
    save_file = '/data/figs/grid' + str(seed) + '.png'
    pddl = {}
    restaurant = taskplan.environments.restaurant.RESTAURANT(seed=seed)
    raise NotImplementedError
    grid = restaurant.grid
    # occupied_cells = np.argwhere(grid == 1)
    # print(occupied_cells)
    plt.clf()
    plt.imshow(grid, cmap='gray_r')
    # plt.scatter(occupied_cells[:, 1], occupied_cells[:, 0], c='red', label='Occupied Cells')
    # print(restaurant.accessible_poses)
    # print(restaurant.known_cost)
    un_oc_cells = list(restaurant.accessible_poses.values())
    for x, y in un_oc_cells:
        # print(x, y, grid[x, y])
        plt.scatter(y, x, c='green')
    plt.savefig(save_file, dpi=1200)
    task = taskplan.pddl.task.clean_everything()
    pddl['domain'] = taskplan.pddl.domain.get_domain()
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    pddl['planner'] = 'ff-astar2'

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=300)
    # print(plan)
    # print(restaurant.containers)
    if plan:
        for p in plan:
            print(p)
        print(cost)
        # print(restaurant.get_current_object_state())
        final_state = restaurant.get_final_state_from_plan(plan)
        # print(final_state)
        restaurant.update_container_props(final_state)
        # print(restaurant.get_current_object_state())
    # print(restaurant.containers)
    for container in restaurant.containers:
        if container.get('assetId') == 'countertop':
            print(container)
    task = taskplan.pddl.task.make_sandwich()
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=300)
    if plan:
        for p in plan:
            print(p)
        print(cost)
        final_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(final_state)
    for container in restaurant.containers:
        if container.get('assetId') == 'countertop':
            print(container)

    task = taskplan.pddl.task.place_something('bread', 'coffeemachine')
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=300)
    if plan:
        for p in plan:
            print(p)
        print(cost)
        final_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(final_state)


if __name__ == "__main__":
    test_antplan_model_output()
    # run_pddl()
