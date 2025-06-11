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
from taskplan_multi.utilities.primitives import (
    ROOMS_KEY, DOORS_KEY, OBJECTS_KEY, POLYGON, ASSET_ID, ROOM_1, ROOM_2, BOT_1, BOT_2)

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



def plot_state_dotted(restaurant, args, image_name='init', title='None'):
    grid = np.transpose(restaurant.grid)
    img = make_plotting_grid(grid)
    plt.clf()
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    plt.title(title)
    plt.savefig(os.path.join(args.output_image_file), dpi=2000)

def plot_state(restaurant, args, image_name='init', title='None'):
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
            if 'bad' in child and child['bad'] == 1:
                name = 'bad ' + name
                cl = 'red'
            # if 'cooked' in child and child['cooked'] == 1:
            #     name = 'cooked ' + name
            #     cl = 'gold'
            plt.text(_x, _y + (i + 1) * 1.2, name, fontsize=5, color=cl)  # Slight offset
        plt.imshow(img, cmap='gray_r', alpha=0.5)
    # plt.axis('off')  # Hides the axis
    plt.title(title)
    plt.savefig(os.path.join(args.output_image_file), dpi=600)

def plot_plan(plan, cost):
    # Add a text block
    textstr = ''
    for p in plan:
        textstr += str(p) + '\n'
    # textstr = 'This is a text block.\nYou can add multiple lines of text.'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Add labels and title
    # plt.title(f'Plan in PDDL, cost : {cost}', fontsize=6)

    # Place a text box in upper left in axes coords
    plt.text(0, .7, textstr, transform=plt.gca().transAxes, fontsize=5,
             verticalalignment='top', bbox=props)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

def plot_graph(restaurant):
    whole_graph = taskplan_multi.utils.get_graph(restaurant)
    image_graph = taskplan_multi.utils.get_image_for_data(whole_graph)
    plt.clf()
    plt.imshow(image_graph)
    plt.savefig(os.path.join(args.output_image_file), dpi=600)

def run_pddl(args):
    # preparing pddl as input to the solver
    seed = 5
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.workshop.WORKSHOP(seed=seed, agents=[BOT_1, BOT_2], active=BOT_1)
    domain = taskplan_multi.pddl.workshop_domain.get_domain()
    task = '(not (is-bad mirror1))'
    # pddl_problem = taskplan_multi.pddl.workshop_problem.get_problem(restaurant, task)
    myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner(domain=domain)
    plan, cost = myopic_planner.get_cost_and_state_from_task(restaurant, task)
    print(plan)
    # plot_graph(restaurant)
    plot_state(restaurant, args, image_name='no-prep-init', title=f'Not Prepared State (INIT)')
    raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TAMP Example Planner.."
    )
    parser.add_argument("--output_image_file", type=str, default="/results/")
    parser.add_argument('--cook_network', type=str, required=False)
    parser.add_argument('--server_network', type=str, required=False)
    parser.add_argument('--cleaner_network', type=str, required=False)
    parser.add_argument('--save_dir', type=str, required=False)
    args = parser.parse_args()
    start_time = time.time()
    run_pddl(args)
    # run_pddl_anticip(args) 
    # draw_map(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.2f} seconds to finish.")