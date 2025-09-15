import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import taskplan_multi
import numpy as np
import os
import learning
from pddlstream.algorithms.search import solve_from_pddl
import random
import argparse
import time
from taskplan_multi.environments.restaurant import world_to_grid
from skimage.morphology import erosion
import gridmap
from shapely import geometry
import copy
import math
from itertools import product

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


def test_ma_plot_grid():
    seed = 111
    fig = plt.figure(figsize=(10, 10), dpi=1000)
    save_file = f'/data/figures/data-grid-{seed}.png'
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cook_bot')
    grid = np.transpose(restaurant.occupancy_grid)
    img = make_plotting_grid(grid)
    plt.subplot(221)
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    containers = restaurant.get_container_pos_list()
    for (name, val) in containers:
        _x, _y = val
        plt.text(_x, _y, name, fontsize=8, color='blue')
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    plt.subplot(222)
    plt.imshow(restaurant.get_top_down_image())
    plt.savefig(save_file, dpi=1000)


def test_ma_plot_path():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['path.simplify'] = False

    seed = 341 #111 #341
    fig = plt.figure(figsize=(10, 10), dpi=1000)

    fname = '2-robot-court'
    save_file = f'/data/figures/res-exp-{fname}.png'
    save_pdf  = f'/data/figures/res-exp-{fname}.pdf'

    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(
        seed=seed, agents=['cook_bot','cleaner_bot','server_bot'], active='cook_bot')

    # Right image
    ax2 = plt.subplot(221)
    ax2.imshow(restaurant.get_top_down_image())

    # Left image + paths
    # Yellow
    ax1 = plt.subplot(222)
    # # Myopic
    # start_hex = "#a2d6e3"  # light # differen shades:  #86e7ff #3ad8ff
    # end_hex   = "#1f5967"  # dark
    
    # Selfish
    # start_hex = "#e7f8ef"  # light
    # end_hex   = "#24814e"  # dark # 248165 #2a7b62
    
    # Courteous
    start_hex = "#fef4d1"  # light
    end_hex   = "#ddae07"  # dark # f1b72b f1862b
    
    
    
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_blue", [start_hex, end_hex], N=256
    )
    grid = np.transpose(restaurant.occupancy_grid)
    img = make_plotting_grid(grid)
    ax1.imshow(img, origin='upper')  # or 'lower' if your coords expect it
    #CAP
    # moves = [('base','pantry'), ('pantry','stove'), ('stove','countertop'), ('countertop', 'stove'), ('stove','shelf'), ('shelf','stove'), ('stove','bussingcart')]
    #SAP-1    
    # moves = [('base','pantry'), ('pantry','stove'), ('stove','bussingcart'), ('bussingcart','countertop')]
     #SAP    
    # moves = [('base','stove'), ('stove','countertop')]
    moves = [('base','countertop'), ('countertop','tvstand')]
    # moves = [('base2','sofa'), ('sofa','stool'), ('stool','tvstand')]
    # moves = [('base','fridge'), ('fridge','table'), ('table','fridge')]
    # moves = [('base','countertop'), ('countertop','table2'), ('table2', 'stove'), ('stove', 'fridge')]
    full_xy = []
    for k, (name1, name2) in enumerate(moves):
        src = restaurant.get_agent_pos() if name1 == 'base' else restaurant.get_container_pos(name1)
        target = restaurant.get_agent_pos() if name2 == 'base' else restaurant.get_container_pos(name2)
        cost, path = restaurant.get_cost_from_occupancy_grid(
            src[0], src[1], target[0], target[1], return_path=True)

        pts = list(zip(path[0], path[1]))          # [(x0,y0), (x1,y1), ...]
        if k > 0 and pts:                          # avoid duplicating the join vertex
            pts = pts[1:]
        full_xy.extend(pts)

    # Safety: need at least 2 points
    if len(full_xy) >= 2:
        full_xy = np.asarray(full_xy, dtype=float)
        x, y = full_xy[:, 0], full_xy[:, 1]

        # Segments for a single LineCollection
        points = full_xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize cumulative distance to [0, 1] for a global gradient
        d = np.hypot(np.diff(x), np.diff(y))
        t = np.insert(np.cumsum(d), 0, 0.0)
        if t[-1] > 0:
            t = t / t[-1]

        lc = LineCollection(
            segments,
            cmap=custom_cmap,                      # your hex-based cmap
            norm=mpl.colors.Normalize(0, 1),
            linewidths=2.0,
            zorder=5
        )
        lc.set_array(t[:-1])                       # one value per segment
        # Optional aesthetics:
        lc.set_capstyle('round')
        lc.set_joinstyle('round')

        ax1.add_collection(lc)
        ax1.autoscale_view()
        ax1.set_aspect('equal', adjustable='box')
    
    # Second Robot Moves (Cleaner)
    # start_hex = "#e7f8ef"  # light
    # end_hex   = "#f1b72b"  # dark
    # custom_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    #     "custom_blue", [start_hex, end_hex], N=256
    # )
    # SAP
    # moves = [('pantry', 'servingtable2'), ('servingtable2', 'sink')]
    # moves = [('pantry','bussingcart'), ('bussingcart','sink'), ('sink','servingtable2'), ('servingtable2','sink')]
    moves = [('base2','sofa')]
    full_xy = []
    for k, (name1, name2) in enumerate(moves):
        src = restaurant.get_agent_pos() if name1 == 'base' else restaurant.get_container_pos(name1)
        target = restaurant.get_agent_pos() if name2 == 'base' else restaurant.get_container_pos(name2)
        cost, path = restaurant.get_cost_from_occupancy_grid(
            src[0], src[1], target[0], target[1], return_path=True)

        pts = list(zip(path[0], path[1]))          # [(x0,y0), (x1,y1), ...]
        if k > 0 and pts:                          # avoid duplicating the join vertex
            pts = pts[1:]
        full_xy.extend(pts)

    # Safety: need at least 2 points
    if len(full_xy) >= 2:
        full_xy = np.asarray(full_xy, dtype=float)
        x, y = full_xy[:, 0], full_xy[:, 1]

        # Segments for a single LineCollection
        points = full_xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize cumulative distance to [0, 1] for a global gradient
        d = np.hypot(np.diff(x), np.diff(y))
        t = np.insert(np.cumsum(d), 0, 0.0)
        if t[-1] > 0:
            t = t / t[-1]

        lc = LineCollection(
            segments,
            cmap=custom_cmap,                      # your hex-based cmap
            norm=mpl.colors.Normalize(0, 1),
            linewidths=2.0,
            zorder=5
        )
        lc.set_array(t[:-1])                       # one value per segment
        # Optional aesthetics:
        lc.set_capstyle('round')
        lc.set_joinstyle('round')

        ax1.add_collection(lc)
        ax1.autoscale_view()
        ax1.set_aspect('equal', adjustable='box')
    
    # # moves = [('countertop','shelf'), ('shelf','servingtable1'), ('servingtable1','sink'), ('sink','servingtable1')]
    # moves = [('countertop','servingtable1'), ('servingtable1','sink'), ('sink','servingtable1')]
    # # moves = [('base2','sofa')]
    # # moves = [('base2','sofa')]
    # full_xy = []
    # for k, (name1, name2) in enumerate(moves):
    #     src = restaurant.get_agent_pos() if name1 == 'base' else restaurant.get_container_pos(name1)
    #     target = restaurant.get_agent_pos() if name2 == 'base' else restaurant.get_container_pos(name2)
    #     cost, path = restaurant.get_cost_from_occupancy_grid(
    #         src[0], src[1], target[0], target[1], return_path=True)

    #     pts = list(zip(path[0], path[1]))          # [(x0,y0), (x1,y1), ...]
    #     if k > 0 and pts:                          # avoid duplicating the join vertex
    #         pts = pts[1:]
    #     full_xy.extend(pts)

    # # Safety: need at least 2 points
    # if len(full_xy) >= 2:
    #     full_xy = np.asarray(full_xy, dtype=float)
    #     x, y = full_xy[:, 0], full_xy[:, 1]

    #     # Segments for a single LineCollection
    #     points = full_xy.reshape(-1, 1, 2)
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)

    #     # Normalize cumulative distance to [0, 1] for a global gradient
    #     d = np.hypot(np.diff(x), np.diff(y))
    #     t = np.insert(np.cumsum(d), 0, 0.0)
    #     if t[-1] > 0:
    #         t = t / t[-1]

    #     lc = LineCollection(
    #         segments,
    #         cmap=custom_cmap,                      # your hex-based cmap
    #         norm=mpl.colors.Normalize(0, 1),
    #         linewidths=2.0,
    #         zorder=5
    #     )
    #     lc.set_array(t[:-1])                       # one value per segment
    #     # Optional aesthetics:
    #     lc.set_capstyle('round')
    #     lc.set_joinstyle('round')

    #     ax1.add_collection(lc)
    #     ax1.autoscale_view()
    #     ax1.set_aspect('equal', adjustable='box')

    #remove axes (do this AFTER creating subplots)
    for ax in fig.get_axes():
        ax.set_axis_off()
        ax.set_frame_on(False)

    # Tight save without borders
    plt.subplots_adjust(0,0,1,1, wspace=0.02, hspace=0.02)
    fig.savefig(save_file, dpi=1000, bbox_inches='tight', pad_inches=0, transparent=True)
    fig.savefig(save_pdf,  format='pdf', bbox_inches='tight', pad_inches=0, transparent=True)

def test_ma_plan_both():
    seed = 5
    plt.clf()
    random.seed(seed)
    pddl = {}
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cook_bot')
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    for container in restaurant.containers:
        _x, _y = world_to_grid(
            container['position']['x'], container['position']['z'],
            restaurant.grid_min_x, restaurant.grid_min_z, restaurant.grid_res)
        assetId = container['assetId']
        plt.text(_x, _y, assetId, fontsize=8, color='blue')
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    # Plan 
    task = tau[1]
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                max_planner_time=120)
    move_plans = [p for p in plan if p.name == "move"]
    move_poses = list()
    offset = 0.2
    for move in move_plans:
        if move.args[1] == 'base':
            pos1 = proc_data.agent['position']
        else:
            pos1 = proc_data.get_container_pos_by_name(move.args[1])
        if move.args[2] == 'base':
            pos2 = proc_data.agent['position']
        else:
            pos2 = proc_data.get_container_pos_by_name(move.args[2])
        move_poses.append((pos1, pos2))
    
    paths = list()
    for pos in move_poses:
        src, targate = (pos)
        cost, path = proc_data.get_cost_from_occupancy_grid(
            src[0], src[1], targate[0], targate[1], return_path=True)
        path_points = [(path[0][idx], path[1][idx])
                       for idx in range(len(path[0]))]
        path = geometry.LineString(path_points)
        x, y = path.xy
        plt.plot(x, y, color='blue')
        path = path.buffer(15)
        paths.append(path)
    
    merged_polygon = None
    for poly in paths:
        if merged_polygon is None:
            merged_polygon = poly
        else:
            merged_polygon = unary_union([merged_polygon, poly])

    x, y = merged_polygon.exterior.xy
    # plt.plot(x, y, color='blue')
    plt.fill(x, y, color='lightblue', alpha=0.5)

    plt.title(title)
    plt.savefig(save_path, dpi=300)

def test_ma_cook_tasks():
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cook_bot')
    # random_state = restaurant.randomize_objects_state(randomness=[random.randint(0, 1), random.choice(random_choices), random.choice(random_choices)], bias=True)
    # restaurant.update_container_props(random_state)
    save_file = '/data/figs/data-grid.png'
    plot_state(restaurant, save_path = save_file, title='Cook Map')
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    count = 0
    for tau in taskplan_multi.pddl.task_distribution.tasks_for_cook():
        print(f"Task: {tau[0]}")
        task = tau[1]
        pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
        plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                    max_planner_time=120)
        # tot_cost = 0
        print(plan)
        print(plan_cost)
        break
        assert plan is not None
        if plan is None:
            count+=1
    print(count)


def test_ma_server_tasks():
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='server_bot')
    # random_state = restaurant.randomize_objects_state(randomness=[random.randint(0, 1), random.choice(random_choices), random.choice(random_choices)], bias=True)
    # restaurant.update_container_props(random_state)
    save_file = '/data/figs/data-grid.png'
    plot_state(restaurant, save_path = save_file, title='Cook Map')
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    count = 0
    for tau in taskplan_multi.pddl.task_distribution.tasks_for_server():
        print(f"Task: {tau[0]}")
        task = tau[1]
        pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
        plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                    max_planner_time=120)
        # tot_cost = 0
        # print(plan)
        print(plan_cost)
        assert plan is not None
        if plan is None:
            count+=1
    print(count)


def test_ma_cleaner_tasks():
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cleaner_bot')
    # random_state = restaurant.randomize_objects_state(randomness=[random.randint(0, 1), random.choice(random_choices), random.choice(random_choices)], bias=True)
    # restaurant.update_container_props(random_state)
    save_file = '/data/figs/data-grid.png'
    plot_state(restaurant, save_path = save_file, title='Cook Map')
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    count = 0
    for tau in taskplan_multi.pddl.task_distribution.tasks_for_cleaner():
        print(f"Task: {tau[0]}")
        task = tau[1]
        pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
        plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                    max_planner_time=120)
        # tot_cost = 0
        # print(plan)
        print(plan_cost)
        assert plan is not None
        if plan is None:
            count+=1
    print(count)

def test_ma_inspect_text():
    root = "/data/server-agent/"
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    true_costs = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = root+pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            for count, node_key in enumerate(x['nodes']):
                print(x['nodes'][node_key]['name'])
    # plt.clf()
    # plt.scatter(true_costs, true_costs, alpha=0.1)
    # # Draw a line from the origin to the farthest point
    # max_value = max(max(true_costs), max(true_costs))
    # plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # # Labeling the axes
    # plt.xlabel('True Costs')
    # plt.ylabel('True Costs')
    # plt.title('Costs Scatter Plot with Line from Origin (On Training)')
    # save_file = '/data/figs/data-viz-cost-server.png'
    # plt.savefig(save_file, dpi=600)


def test_ma_inspect_data():
    root = "/data/agent-server/"
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    true_costs = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = root+pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            true_costs.append(x['label'])
    plt.clf()
    plt.scatter(true_costs, true_costs, alpha=0.1)
    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(true_costs))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # Labeling the axes
    plt.xlabel('True Costs')
    plt.ylabel('True Costs')
    plt.title('Costs Scatter Plot with Line from Origin')
    save_file = '/data/figs/data-viz-true-cost-server.png'
    plt.savefig(save_file, dpi=600)

def test_ma_model_output():
    root = "/data/server-agent/"
    net_name = 'ap_server.pt'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file= root + 'logs/beta-v0/' + net_name,
        device=device
    )
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    true_costs = list()
    exp_cost = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = root + pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            true_costs.append(x['label'])
            anticipated_cost = eval_net(x)
            exp_cost.append(anticipated_cost)
    plt.clf()
    plt.scatter(true_costs, exp_cost, alpha=0.1)
    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(exp_cost))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # Labeling the axes
    plt.xlabel('True Costs')
    plt.ylabel('Learned Costs')
    plt.title('Costs Scatter Plot with Line from Origin (On Training)')
    save_file = '/data/figs/' + net_name + '-compare-cost.png'
    plt.savefig(save_file, dpi=600)


def test_ma_model_output_cleaner():
    root = "/data/agent-cleaner/"
    net_name = 'ap_cleaner.pt'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file= root + 'logs/beta-v0/' + net_name,
        device=device
    )
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    true_costs = list()
    exp_cost = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = root + pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            true_costs.append(x['label'])
            anticipated_cost = eval_net(x)
            exp_cost.append(anticipated_cost)
    plt.clf()
    plt.scatter(true_costs, exp_cost, alpha=0.1)
    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(exp_cost))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # Labeling the axes
    plt.xlabel('True Costs')
    plt.ylabel('Learned Costs')
    plt.title('Cleaner Robot True vs Learned Cost Comparison')
    save_file = '/data/figs/' + net_name + '-learned-cost.png'
    plt.savefig(save_file, dpi=600)

def test_ma_model_output_cook():
    root = "/data/agent-cook/"
    net_name = 'ap_cook.pt'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file= root + 'logs/beta-v0/' + net_name,
        device=device
    )
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    true_costs = list()
    exp_cost = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = root + pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            true_costs.append(x['label'])
            anticipated_cost = eval_net(x)
            exp_cost.append(anticipated_cost)
    plt.clf()
    plt.scatter(true_costs, exp_cost, alpha=0.1)
    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(exp_cost))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # Labeling the axes
    plt.xlabel('True Costs')
    plt.ylabel('Learned Costs')
    plt.title('Cook Robot True vs Learned Cost Comparison')
    save_file = '/data/figs/' + net_name + '-learned-cost.png'
    plt.savefig(save_file, dpi=600)

def test_ma_model_output_server():
    root = "/data/agent-server/"
    net_name = 'ap_server.pt'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file= root + 'logs/beta-v0/' + net_name,
        device=device
    )
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    true_costs = list()
    exp_cost = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = root + pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            true_costs.append(x['label'])
            anticipated_cost = eval_net(x)
            exp_cost.append(anticipated_cost)
    plt.clf()
    plt.scatter(true_costs, exp_cost, alpha=0.1)
    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(exp_cost))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red
    plt.xlabel('True Costs')
    plt.ylabel('Learned Costs')
    plt.title('Server Robot True vs Learned Cost Comparison')
    save_file = '/data/figs/' + net_name + '-learned-cost.png'
    plt.savefig(save_file, dpi=600)

def test_ma_anticipatory_cost_cleaner():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    server_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/new-server/logs/beta-v0/ap_server.pt',
        device=device
    )
    cleaner_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/new-cleaner/logs/beta-v0/ap_cleaner.pt',
        device=device
    )
    cook_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/new-cook/logs/beta-v0/ap_cook.pt',
        device=device
    )
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='server_bot')
    save_file = '/data/figs/data-grid-before.png'
    plot_state(restaurant, save_path = save_file, title='Before Map')
    print("Before Plan")
    ###
    restaurant.active_robot = 'cook_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ap_cost_cook = cook_net(whole_graph)
    print(f"Cook: {ap_cost_cook}")
    ###
    restaurant.active_robot = 'server_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ap_cost_server = server_net(whole_graph)
    print(f"Server: {ap_cost_server}")
    restaurant.active_robot = 'cleaner_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ap_cost_cleaner = cleaner_net(whole_graph)
    print(f"Cleaner: {ap_cost_cleaner}")
    
    ### After Plan
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    # task = f'''(and {taskplan_multi.pddl.task.clean_something('mug1')} {taskplan_multi.pddl.task.clean_something('mug1')})'''
    task = taskplan_multi.pddl.task.serve_item('pasta', 'bowl', 'servingtable1', garnish='sauce')
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                max_planner_time=120)
    print(plan)
    print(plan_cost)
    
    print("After Plan")
    term_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(term_state)
    save_file = '/data/figs/data-grid-after.png'
    plot_state(restaurant, save_path = save_file, title='After Map')
    ###
    restaurant.active_robot = 'cook_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ap_cost_cook = cook_net(whole_graph)
    print(f"Cook: {ap_cost_cook}")
    ###
    restaurant.active_robot = 'server_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ap_cost_server = server_net(whole_graph)
    print(f"Server: {ap_cost_server}")
    restaurant.active_robot = 'cleaner_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    ap_cost_cleaner = cleaner_net(whole_graph)
    print(f"Cleaner: {ap_cost_cleaner}")

    assert True

def get_anticipated_cost(restaurant):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    server_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/agent-server/logs/beta-v0/ap_server.pt',
        device=device
    )
    cleaner_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/agent-cleaner/logs/beta-v0/ap_cleaner.pt',
        device=device
    )
    cook_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file='/data/agent-cook/logs/beta-v0/ap_cook.pt',
        device=device
    )
    prev_active = restaurant.active_robot
    
    cost = 0
    restaurant.active_robot = 'cook_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    cost += cook_net(whole_graph)
    
    restaurant.active_robot = 'server_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    cost += server_net(whole_graph)
    
    restaurant.active_robot = 'cleaner_bot'
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    cost += cleaner_net(whole_graph)
    
    restaurant.active_robot = prev_active
    return cost

def get_combo(item1, item2):
    combinations = [list(zip(item1, p)) for p in product(item2, repeat=len(item1))]
    return combinations

def aug_predicates_for_cleaner(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=set()):
    aug_predicates = list()
    current_state = copy.deepcopy(restaurant.get_current_object_state())

    conts = ['countertop', 'cabinet', 'fridge', 'pantry', 'sink', 'servingtable1', 'servingtable2', 'bussingcart', 'shelf']
    assets = ['mug1', 'mug2', 'pan1', 'pan2', 'bowl1', 'bowl2', 'pasta', 'cereal', 'milk', 'oats', 'sauce', 'saltshaker']
    
    prior_combos = list()
    selected_conts = set()
    selected_assets = set()
    
    for item1 in used_containers:
        for item2 in conts:
            if restaurant.known_cost[item1][item2] < 20:
                selected_conts.add(item2)
                ant_objects = restaurant.get_objects_by_container_name(item2)
                for obj in ant_objects:
                    if obj['assetId'] not in used_items:
                        selected_assets.add(obj['assetId'])

    print(used_items)
    print(used_containers)
    print(selected_assets)
    print(selected_conts)
    if len(used_items) >= 4:
        sampled_conts = random.sample(list(selected_conts), 2)
        prior_combos = get_combo(list(used_items), sampled_conts)
    else:
        for idx, take_one in enumerate(selected_assets):
            temp = [take_one]
            temp.extend(list(used_items))
            max_size = min(len(selected_conts), 4)
            sampled_conts = random.sample(list(selected_conts), max_size)
            temp_combos = get_combo(temp, sampled_conts)
            prior_combos.extend(temp_combos)

    print(len(prior_combos))
    max_size = min(len(prior_combos), 50)
    prior_combos = random.sample(prior_combos, max_size)
    print(len(prior_combos))
    
    for item_list in prior_combos:
        restaurant.update_container_props(current_state)
        tt = '(and'
        for (obj_name, cnt_name) in item_list:
            obj = restaurant.get_object_props_by_name(obj_name)
            val = restaurant.get_container_pos(cnt_name)
            if 'washable' in obj and 'dirty' in obj and obj['dirty'] == 1:
                candidate_state = restaurant.place_washables(obj, val, dirty=0)
                t1 = taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], cnt_name)
            else:
                candidate_state = restaurant.place_object(obj, val)
                t1 = taskplan_multi.pddl.task.place_something(obj['assetId'], cnt_name)
            restaurant.update_container_props(candidate_state)
            tt += t1
        tt += ')'
        can_exp_cost = get_anticipated_cost(restaurant)
        if can_exp_cost < myopic_ex_cost:
            # print(f"Org: {can_exp_cost} : {tt}")
            aug_predicates.append(tt)
    # obj = item
    # raise NotImplementedError
    # for cnt in conts:
    #     restaurant.update_container_props(current_state)
    #     ant_objects = restaurant.get_objects_by_container_name(cnt)
    #     for obj in ant_objects:
    #         for (oth_cnt, val) in restaurant.get_container_pos_list():
    #             if oth_cnt not in conts:
    #                 continue
    #             candidate_state = restaurant.place_object(obj, val)
    #             restaurant.update_container_props(candidate_state)
    #             can_exp_cost = get_anticipated_cost(restaurant)
    #             if can_exp_cost < myopic_ex_cost:
    #                 aug_predicates.append(taskplan_multi.pddl.task.place_something(obj['assetId'], oth_cnt))
    #             if 'washable' in obj:
    #                 if 'dirty' in obj and obj['dirty'] == 1:
    #                     restaurant.update_container_props(current_state)
    #                     candidate_state = restaurant.place_washables(obj, val, dirty=0)
    #                     restaurant.update_container_props(candidate_state)
    #                     can_exp_cost = get_anticipated_cost(restaurant)
    #                     if can_exp_cost < myopic_ex_cost:
    #                         aug_predicates.append(taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], oth_cnt))
                        
    return aug_predicates


def aug_predicates_for_cook(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=set()):
    aug_predicates = list()
    current_state = copy.deepcopy(restaurant.get_current_object_state())

    conts = ['stove', 'countertop', 'cabinet', 'fridge', 'pantry', 'sink', 'bussingcart', 'shelf']
    assets = ['mug1', 'mug2', 'pan1', 'pan2', 'bowl1', 'bowl2', 'pasta', 'cereal', 'milk', 'oats', 'sauce', 'saltshaker']
    item_to_remove = ['pasta', 'cereal', 'milk', 'oats']

    if len(other_bots) > 0:
        conts.append('servingtable1')
        conts.append('servingtable2')

    
    prior_combos = list()
    selected_conts = set()
    selected_assets = set()
    item_to_use = set()

    for itm in item_to_remove:
        if itm in used_items:
            used_items.remove(itm)
            item_to_use.add(itm)
    
    
    for item1 in used_containers:
        for item2 in conts:
            if restaurant.known_cost[item1][item2] < 20:
                selected_conts.add(item2)
                ant_objects = restaurant.get_objects_by_container_name(item2)
                for obj in ant_objects:
                    if obj['assetId'] not in item_to_use:
                        selected_assets.add(obj['assetId'])

    print(used_items)
    print(item_to_use)
    print(used_containers)
    print(selected_assets)
    print(selected_conts)
    if len(item_to_use) >= 4:
        sampled_conts = random.sample(list(selected_conts), 2)
        prior_combos = get_combo(list(item_to_use), sampled_conts)
    else:
        for idx, take_one in enumerate(selected_assets):
            temp = [take_one]
            temp.extend(list(item_to_use))
            max_size = min(len(selected_conts), 4)
            sampled_conts = random.sample(list(selected_conts), max_size)
            temp_combos = get_combo(temp, sampled_conts)
            prior_combos.extend(temp_combos)

    print(len(prior_combos))
    max_size = min(len(prior_combos), 1000)
    prior_combos = random.sample(prior_combos, max_size)
    print(len(prior_combos))
    
    for item_list in prior_combos:
        print(item_list)
        restaurant.update_container_props(current_state)
        tt = '(and'
        for (obj_name, cnt_name) in item_list:
            obj = restaurant.get_object_props_by_name(obj_name)
            val = restaurant.get_container_pos(cnt_name)
            if 'cleaner_bot' in other_bots and 'washable' in obj and 'dirty' in obj and obj['dirty'] == 1:
                candidate_state = restaurant.place_washables(obj, val, dirty=0)
                t1 = taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], cnt_name)
            elif 'server_bot' in other_bots and 'food' in obj and 'empty' in obj and obj['empty'] == 1:
                candidate_state = restaurant.place_food_items(obj, val, empty=0)
                t1 = taskplan_multi.pddl.task.stock_and_place_something(obj['assetId'], cnt_name)
            else:
                candidate_state = restaurant.place_object(obj, val)
                t1 = taskplan_multi.pddl.task.place_something(obj['assetId'], cnt_name)
            restaurant.update_container_props(candidate_state)
            tt += t1
        tt += ')'
        can_exp_cost = get_anticipated_cost(restaurant)
        if can_exp_cost < myopic_ex_cost:
            aug_predicates.append(tt)
    raise NotImplementedError
    return aug_predicates


def aug_predicates_for_server(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=set()):
    aug_predicates = list()
    current_state = copy.deepcopy(restaurant.get_current_object_state())

    conts = ['servingtable1', 'servingtable2', 'countertop', 'cabinet', 'fridge', 'pantry', 'sink', 'bussingcart', 'shelf']
    assets = ['mug1', 'mug2', 'pan1', 'pan2', 'bowl1', 'bowl2', 'pasta', 'cereal', 'milk', 'oats', 'sauce', 'saltshaker']
    item_to_remove = ['pasta', 'cereal', 'oats']

    if len(other_bots) > 0 and 'cook_bot' in other_bots:
        conts.append('stove')

    
    prior_combos = list()
    selected_conts = set()
    selected_assets = set()
    item_to_use = set()

    for itm in item_to_remove:
        if itm in used_items:
            used_items.remove(itm)
            item_to_use.add(itm)
    
    
    for item1 in used_containers:
        for item2 in conts:
            if restaurant.known_cost[item1][item2] < 20:
                selected_conts.add(item2)
                ant_objects = restaurant.get_objects_by_container_name(item2)
                for obj in ant_objects:
                    if obj['assetId'] not in item_to_use:
                        selected_assets.add(obj['assetId'])

    print(used_items)
    print(item_to_use)
    print(used_containers)
    print(selected_assets)
    print(selected_conts)
    if len(item_to_use) >= 4:
        sampled_conts = random.sample(list(selected_conts), 2)
        prior_combos = get_combo(list(item_to_use), sampled_conts)
    else:
        for idx, take_one in enumerate(selected_assets):
            temp = [take_one]
            temp.extend(list(item_to_use))
            max_size = min(len(selected_conts), 4)
            sampled_conts = random.sample(list(selected_conts), max_size)
            temp_combos = get_combo(temp, sampled_conts)
            prior_combos.extend(temp_combos)

    if len(item_to_use) == 1:
            one_list_combo = get_combo(list(item_to_use), conts)
            prior_combos.extend(one_list_combo)
    print(len(prior_combos))
    max_size = min(len(prior_combos), 1000)
    prior_combos = random.sample(prior_combos, max_size)
    print(len(prior_combos))
    # raise NotImplementedError
    print(other_bots)
    # raise NotImplementedError
    
    for item_list in prior_combos:
        print(item_list)
        restaurant.update_container_props(current_state)
        tt = '(and'
        for (obj_name, cnt_name) in item_list:
            obj = restaurant.get_object_props_by_name(obj_name)
            val = restaurant.get_container_pos(cnt_name)
            if 'food' in obj and 'empty' in obj and obj['empty'] == 1:
                candidate_state = restaurant.place_food_items(obj, val, empty=0)
                t1 = taskplan_multi.pddl.task.stock_and_place_something(obj['assetId'], cnt_name)
            elif 'cleaner_bot' in other_bots and 'washable' in obj and 'dirty' in obj and obj['dirty'] == 1:
                candidate_state = restaurant.place_washables(obj, val, dirty=0)
                t1 = taskplan_multi.pddl.task.clean_and_place_something(obj['assetId'], cnt_name)
            else:
                candidate_state = restaurant.place_object(obj, val)
                t1 = taskplan_multi.pddl.task.place_something(obj['assetId'], cnt_name)
            restaurant.update_container_props(candidate_state)
            tt += t1
        tt += ')'
        can_exp_cost = get_anticipated_cost(restaurant)
        if can_exp_cost < myopic_ex_cost:
            aug_predicates.append(tt)
    raise NotImplementedError
    return aug_predicates

def get_anticipatory_plan(restaurant, task):
    init_state = copy.deepcopy(restaurant.get_current_object_state())
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()
    plan, myopic_cost = (
        myopic_planner.get_cost_and_state_from_task(restaurant, task)
    )
    if plan is None:
        return init_state, 10000, task, 0
    
    print(plan)
    
    help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
    other_bots = set()
   
    used_items = set()
    used_containers = set()
    for p in plan:
        if "pick" in p.name:
            used_containers.add(p.args[2])
        if "place" in p.name:
            used_items.add(p.args[1])
            used_containers.add(p.args[2])
        if "wash" in p.name:
            used_items.add(p.args[1])
        if "mix" in p.name:
            used_items.add(p.args[1])
            used_items.add(p.args[2])
        if "serve" in p.name:
            used_items.add(p.args[1])
            used_items.add(p.args[2])
        if "restock" in p.name:
            used_items.add(p.args[1])
        if "ask-help" in p.name:
            other_bots.add(p.args[0])
    print(help_stat)

    term_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(term_state)

    myopic_ex_cost = get_anticipated_cost(restaurant)
    myopic_ant_cost = myopic_ex_cost + myopic_cost

    print(f"Myopic: {myopic_ex_cost} + {myopic_cost}")

    ant_cost = myopic_cost
    ant_state = copy.deepcopy(term_state)
    
    if restaurant.active_robot == 'server_bot':
        aug_predicates = aug_predicates_for_server(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)
    elif restaurant.active_robot == 'cook_bot':
        aug_predicates = aug_predicates_for_cook(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)
    else:
        aug_predicates = aug_predicates_for_cleaner(restaurant, used_items, used_containers, myopic_ex_cost, other_bots=other_bots)

    # if help_stat:
    #     if 'cook_bot' in other_bots:
    #         extra_pred = aug_predicates_for_cook(restaurant, used_items, myopic_ex_cost)
    #         aug_predicates.extend(extra_pred)
    #     if 'server_bot' in other_bots:
    #         extra_pred = aug_predicates_for_server(restaurant, used_items, myopic_ex_cost)
    #         aug_predicates.extend(extra_pred)
    #     if 'cleaner_bot' in other_bots:
    #         extra_pred = aug_predicates_for_cleaner(restaurant, used_items, myopic_ex_cost)
    #         # print(extra_pred)
    #         aug_predicates.extend(extra_pred)

    ant_task = task
    print(len(aug_predicates))
    for aug_pred in aug_predicates:
        restaurant.update_container_props(init_state)
        ant_task_pred = f'(and {aug_pred} {task})'
        plan, c_cost = (
            myopic_planner.get_cost_and_state_from_task(
                restaurant, ant_task_pred))
        if plan is None:
            continue
        can_state = restaurant.get_final_state_from_plan(plan)
        restaurant.update_container_props(can_state)
        can_ex_cost = get_anticipated_cost(restaurant)
        can_ant_cost = can_ex_cost + c_cost
        print(f"Candidate: {can_ex_cost-myopic_ex_cost} + {c_cost-myopic_cost} : {ant_task_pred}")
        if can_ant_cost < myopic_ant_cost:
            ant_state = copy.deepcopy(can_state)
            ant_cost = c_cost
            myopic_ant_cost = can_ant_cost
            ant_task = ant_task_pred
            help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
    return ant_state, ant_cost, ant_task, help_stat


def test_ma_anticipatory_plan_cleaner():
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cleaner_bot')
    save_file = '/data/figs/data-grid-before.png'
    plot_state(restaurant, save_path = save_file, title='Before Map')
    task = taskplan_multi.pddl.task.clean_both_item('bowl')
    get_anticipatory_plan(restaurant, task)
    assert True


def test_ma_anticipatory_plan_cook():
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cook_bot')
    save_file = '/data/figs/data-grid-before.png'
    plot_state(restaurant, save_path = save_file, title='Before Map')
    task = taskplan_multi.pddl.task.prep_item('oats', 'bowl', 'stove')
    get_anticipatory_plan(restaurant, task)
    assert True

def test_ma_anticipatory_plan_server():
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='server_bot')
    # random_state = restaurant.randomize_objects_state(randomness=[random.randint(0, 1), random.choice(random_choices), random.choice(random_choices)], bias=True)
    # restaurant.update_container_props(random_state)
    save_file = '/data/figs/data-grid-before.png'
    plot_state(restaurant, save_path = save_file, title='Before Map')
    task = f"(and {taskplan_multi.pddl.task.serve_item('oats', 'bowl', 'servingtable1')} {taskplan_multi.pddl.task.restock_something('oats')})"
    # taskplan_multi.pddl.task.serve_item('cereal', 'bowl', 'servingtable2', garnish='milk')
    get_anticipatory_plan(restaurant, task)
    assert True


def test_ma_server_combined_task():
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='server_bot')
    save_file = '/data/figs/data-grid.png'
    plot_state(restaurant, save_path = save_file, title='Cook Map')
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    count = 0
    task = f"(and {taskplan_multi.pddl.task.serve_item('oats', 'bowl', 'servingtable1')} {taskplan_multi.pddl.task.restock_something('oats')})"
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                max_planner_time=120)
    print(plan)
    print(plan_cost)
    assert plan is not None


def test_ma_cleaner_ap_cost():
    def load_prepared_state():
        file_name = ''
        root = '/data/exp-v0/results/5'
        for path, _, files in os.walk(root):
            for name in files:
                if 'prep_state_' + str(args.current_seed) in name:
                    file_name = os.path.join(path, name)
                    datum = json.load(open(file_name))
                    return datum
        return None
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='server_bot')
    prep_state = load_prepared_state()
    restaurant.update_container_props(prep_state)
    save_file = '/data/figs/data-grid-prepared-5.png'
    plot_state(restaurant, save_path = save_file, title='Prepared Map')

    task = taskplan_multi.pddl.task.serve_item('oats', 'bowl', 'servingtable1')
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    count = 0
    task = f"(and {taskplan_multi.pddl.task.serve_item('oats', 'bowl', 'servingtable1')} {taskplan_multi.pddl.task.restock_something('oats')})"
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                max_planner_time=120)
    print(plan)
    print(plan_cost)
    assert plan is not None

def change_state(restaurant):
    candidate_state = None
    while (candidate_state is None):
        objs = restaurant.get_current_object_state()
        obj = random.choice(objs)
        cont_list = [v for (c, v) in restaurant.get_container_pos_list()]
        container = random.choice(cont_list)
        # if random.random() > 0.5:
        candidate_state = restaurant.place_object(obj, container)
        # else:
        #     if 'washable' in obj:
        #         # if 'dirty' in obj and obj['dirty'] == 1:
        #         if random.random() > 0.75:
        #             candidate_state = restaurant.place_washables(obj, container, dirty=0)
        #         else:
        #             candidate_state = restaurant.place_washables(obj, container, dirty=1)
        #     elif 'food' in obj:
        #         # if 'empty' in obj and obj['empty'] == 1:
        #         if random.random() > 0.75:
        #             candidate_state = restaurant.place_food_items(obj, container, empty=0)
        #         else:
        #             candidate_state = restaurant.place_food_items(obj, container, empty=1)
        #     else:
        #         candidate_state = restaurant.place_object(obj, container)
    return candidate_state

def get_prepared_state(restaurant, n_iterations=1000):
        def safe_exp(x):
            try:
                return math.exp(x)
            except OverflowError:
                return 0.0
        prepared_state = copy.deepcopy(restaurant.get_current_object_state())
        int_cost = get_anticipated_cost(restaurant)
        print(int_cost)
        i = 0
        temp = 100
        cooling_rate = 0.95
        restaurant.active_all = True
        while (i < n_iterations):
            restaurant.update_container_props(prepared_state)
            i += 1
            candidate_state = change_state(restaurant)
            restaurant.update_container_props(candidate_state)
            can_exp_cost = get_anticipated_cost(restaurant)
            delta = (can_exp_cost-int_cost)
            exp = safe_exp(-delta/temp)
            if delta < 0 or random.uniform(0, 1) < exp:
                prepared_state = copy.deepcopy(candidate_state)
                int_cost = can_exp_cost
            temp = max(temp * cooling_rate, 1)
        print(int_cost)
        return prepared_state

def test_ma_preparation():
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='server_bot')
    # random_state = restaurant.randomize_objects_state(randomness=[random.randint(0, 1), random.choice(random_choices), random.choice(random_choices)], bias=True)
    # restaurant.update_container_props(random_state)
    save_file = '/data/figs/data-grid-before-prep-cook.png'
    init_cost = get_anticipated_cost(restaurant)
    plot_state(restaurant, save_path = save_file, title=f"Initial Map: Exp Cost: {init_cost}")
    prep_state = get_prepared_state(restaurant)
    restaurant.update_container_props(prep_state)
    prep_cost = get_anticipated_cost(restaurant)
    save_file = '/data/figs/data-grid-prep-cook.png'
    plot_state(restaurant, save_path = save_file, title=f"Prep Map (Cook-Only): Exp Cost: {prep_cost}")
    assert True


def test_ma_anticipatory_plan_three_tasks():
    pddl = {}
    seed = 5
    random_choices = [0, 0.25, 0.5, 0.75, 1]
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed, agents=['cook_bot', 'cleaner_bot', 'server_bot'], active='cleaner_bot')
    no_prep_state = copy.deepcopy(restaurant.get_current_object_state())
    save_file = '/data/figs/data-grid-before-task-mp.png'
    plot_state(restaurant, save_path = save_file, title='Before Task')

    total_cost = 0
    start_time = time.time()
    task = taskplan_multi.pddl.task.clean_something('pan2')
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar2'
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                max_planner_time=120)
    print(plan)
    total_cost+=plan_cost
    term_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(term_state)

    restaurant.active_robot = 'cook_bot'
    task = taskplan_multi.pddl.task.cook_something('pasta')
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                max_planner_time=120)
    print(plan)
    total_cost+=plan_cost
    term_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(term_state)

    restaurant.active_robot = 'server_bot'
    task = taskplan_multi.pddl.task.serve_item('pasta', 'bowl', 'servingtable1', garnish='sauce')
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, plan_cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                max_planner_time=120)
    print(plan)
    total_cost+=plan_cost

    print(f"Myopic: {total_cost}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.2f} seconds to finish.")

    term_state = restaurant.get_final_state_from_plan(plan)
    restaurant.update_container_props(term_state)
    save_file = '/data/figs/data-grid-after-task-mp.png'
    plot_state(restaurant, save_path = save_file, title='Aftef Task (Myopic)')

    start_time = time.time()
    total_cost = 0
    restaurant.active_robot = 'cleaner_bot'
    task = taskplan_multi.pddl.task.clean_something('pan2')
    restaurant.update_container_props(no_prep_state)
    save_file = '/data/figs/data-grid-before-task-ap.png'
    plot_state(restaurant, save_path = save_file, title='Before Task (A.P)')
    ant_state, ant_cost, ant_task, help_stat = get_anticipatory_plan(restaurant, task)
    total_cost+=ant_cost
    print(ant_task)
    
    restaurant.update_container_props(ant_state)
    restaurant.active_robot = 'cook_bot'
    task = taskplan_multi.pddl.task.cook_something('pasta')
    ant_state, ant_cost, ant_task, help_stat = get_anticipatory_plan(restaurant, task)
    total_cost+=ant_cost
    print(ant_task)

    restaurant.update_container_props(ant_state)
    restaurant.active_robot = 'server_bot'
    task = taskplan_multi.pddl.task.serve_item('pasta', 'bowl', 'servingtable1', garnish='sauce')
    ant_state, ant_cost, ant_task, help_stat = get_anticipatory_plan(restaurant, task)
    total_cost+=ant_cost
    print(ant_task)

    print(total_cost)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.2f} seconds to finish.")
    restaurant.update_container_props(ant_state)
    save_file = '/data/figs/data-grid-after-task-ap.png'
    plot_state(restaurant, save_path = save_file, title='After Task (A.P)')
    
    # get_anticipatory_plan(restaurant, task)
    assert True