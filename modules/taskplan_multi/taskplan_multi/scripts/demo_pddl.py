from pddlstream.algorithms.search import solve_from_pddl
import taskplan_multi
import random
import matplotlib.pyplot as plt
import os
import argparse
import time

def run_pddl(args):
    # preparing pddl as input to the solver
    seed = 10
    pddl = {}
    # random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed)

    grid = restaurant.grid
    # occupied_cells = np.argwhere(grid == 1)
    # print(occupied_cells)
    plt.clf()
    plt.imshow(grid, cmap='gray_r')
    # plt.scatter(occupied_cells[:, 1], occupied_cells[:, 0], c='red', label='Occupied Cells')
    # print(restaurant.accessible_poses)
    # print(restaurant.known_cost)
    # robot_poses = [restaurant.agent_tall["position"], restaurant.agent_tiny["position"]]
 
    plt.scatter(restaurant.agent_tall["position"]["x"], restaurant.agent_tall["position"]["z"], c='red', s=200)
    plt.scatter(restaurant.agent_tiny["position"]["x"], restaurant.agent_tiny["position"]["z"], c='green', s=200)
    plt.savefig(os.path.join(args.output_image_file), dpi=2000)

    # print(restaurant.containers)
    pddl['domain'] = taskplan_multi.pddl.domain.get_domain()
    pddl['planner'] = 'ff-astar'
    # task = taskplan_multi.pddl.task.move_robot('agent_tall', 'servingtable1')
    task = taskplan_multi.pddl.task.place_something('bowl2', 'cabinet')
    pddl['problem'] = taskplan_multi.pddl.problem.get_problem(restaurant, task)
    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
                                 max_planner_time=60)
    # tot_cost = 0
    if plan:
        for p in plan:
            print(p)
        print(cost)
    # raise NotImplementedError
    # final_state = restaurant.get_random_state()
    # restaurant.update_container_props(final_state[0],
    #                                   final_state[1],
    #                                   final_state[2])
    # pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    # plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
    #                              max_planner_time=300)
    # # print(plan)
    # if plan:
    #     for p in plan:
    #         print(p)
    #     print(cost)
    # task = taskplan.pddl.task.serve_water('servingtable2', 'mug2')
    # pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant, task)
    # plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'],
    #                              max_planner_time=300)
    # # print(plan)
    # if plan:
    #     for p in plan:
    #         print(p)
    #     print(cost)
    #     final_state = restaurant.get_final_state_from_plan(plan)
    #     restaurant.update_container_props(final_state[0],
    #                                       final_state[1],
    #                                       final_state[2])
    # print(restaurant.get_objects_by_container_name('servingtable2'))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TAMP Example Planner.."
    )
    parser.add_argument("--output_image_file", type=str, default="/results/")
    args = parser.parse_args()
    start_time = time.time()
    run_pddl(args) 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.2f} seconds to finish.")