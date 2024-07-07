import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion

import taskplan


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


def plot_plan(plan):
    # Add a text block
    textstr = ''
    for p in plan:
        textstr += str(p) + '\n'
    # textstr = 'This is a text block.\nYou can add multiple lines of text.'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Add labels and title
    plt.title('Plan progression', fontsize=6)

    # Place a text box in upper left in axes coords
    plt.text(0, .7, textstr, transform=plt.gca().transAxes, fontsize=5,
             verticalalignment='top', bbox=props)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])


def plot_result(partial_map, whole_graph,
                plan, path, cost_str, args=None):
    ''' This function plots the result in a meaningful way so that
    what happened during the trial can be easily understood.
    '''
    plt.clf()
    plt.figure(figsize=(10, 5))
    if args:
        what = partial_map.org_node_names[partial_map.target_obj]
        where = [partial_map.org_node_names[goal] for goal in partial_map.target_container]
        plt.suptitle(f"Find {what} from {where} in seed: [{args.current_seed}]", fontsize=9)
    dist, trajectory = taskplan.core.compute_path_cost(
        partial_map.grid, path)

    # plot the plan
    plt.subplot(131)
    plot_plan(plan)

    # plot the underlying graph (except objects) on grid
    plt.subplot(132)
    plt.title("Underlying grid with containers", fontsize=6)
    plot_graph_on_grid(partial_map.grid, whole_graph)
    # plt.axis('equal')
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = path[0]
    plt.text(x, y, '+', color='red', size=6, rotation=45)

    # plot the trajectory
    plt.subplot(133)
    plotting_grid = make_plotting_grid(
        np.transpose(partial_map.grid)
    )
    plt.title(f"{cost_str}: {dist:0.3f}", fontsize=6)
    plt.imshow(plotting_grid)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    x, y = path[0]
    plt.text(x, y, '0 - ROBOT', color='brown', size=4)

    for idx, coords in enumerate(path[1:]):
        # find the node_idx for this pose and use it through
        # graph['node_coords']
        pose = taskplan.utils. \
            get_pos_from_coord(coords, whole_graph['node_coords'])
        x = whole_graph['node_coords'][pose][0]
        y = whole_graph['node_coords'][pose][1]
        name = whole_graph['node_names'][pose]
        plt.text(x, y, f'{idx+1} - {pose}:{name}', color='brown', size=4)

    # Generate colors based on the Viridis color map
    # Create a Viridis color map
    red_cmap = plt.get_cmap('Reds')
    blue_cmap = plt.get_cmap('Blues')
    colors = np.linspace(0, 1, len(trajectory[0]))
    red_colors = red_cmap(colors)
    blue_colors = blue_cmap(colors)
    selected_color = blue_colors
    flag = 'blue'

    # Plot the points with Viridis color gradient
    for idx, x in enumerate(trajectory[0]):
        y = trajectory[1][idx]
        if args:
            if x == args.robot_path[0][0] and y == args.robot_path[0][1] and flag == 'blue':
                selected_color = red_colors
                flag = 'red'
            elif x == args.robot_path[-1][0] and y == args.robot_path[-1][1] and flag == 'red':
                selected_color = blue_colors
                flag = None

        plt.plot(x, y, color=selected_color[idx], marker='.', markersize=2, alpha=0.9)


def plot_graph_on_grid(grid, graph):
    '''Plot the scene graph on the occupancy grid to scale'''
    plotting_grid = make_plotting_grid(np.transpose(grid))
    plt.imshow(plotting_grid)

    # find the room nodes
    room_node_idx = [idx for idx in range(1, graph['cnt_node_idx'][0])]

    rc_idx = room_node_idx + graph['cnt_node_idx']

    # plot the edge connectivity between rooms and their containers only
    filtered_edges = [
        edge
        for edge in graph['edge_index']
        if edge[1] in rc_idx and edge[0] != 0
    ]

    for (start, end) in filtered_edges:
        p1 = graph['nodes'][start]['pos']
        p2 = graph['nodes'][end]['pos']
        x_values = [p1[0], p2[0]]
        y_values = [p1[1], p2[1]]
        plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

    # plot room nodes
    for room in rc_idx:
        room_pos = graph['nodes'][room]['pos']
        room_name = graph['nodes'][room]['name']
        plt.text(room_pos[0], room_pos[1], room_name, color='brown', size=6, rotation=40)


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
