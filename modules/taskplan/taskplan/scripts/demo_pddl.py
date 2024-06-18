from pddlstream.algorithms.search import solve_from_pddl
import matplotlib.pyplot as plt
import taskplan
import numpy as np


def run_pddl():
    # preparing pddl as input to the solver
    seed = 1
    save_file = '/data/figs/grid' + str(seed) + '.png'
    pddl = {}
    restaurant = taskplan.environments.restaurant.RESTAURANT(seed=2)
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
        print(x, y, grid[x, y])
        plt.scatter(y, x, c='green')
    plt.savefig(save_file, dpi=1200)
    pddl['domain'] = taskplan.pddl.domain.get_domain()
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant)
    pddl['planner'] = 'ff-astar2'

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=180)
    print(plan)

    if plan:
        for p in plan:
            print(p)
        print(cost)


if __name__ == "__main__":
    run_pddl()
