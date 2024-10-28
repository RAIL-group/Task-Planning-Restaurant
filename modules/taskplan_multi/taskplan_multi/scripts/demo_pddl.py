from pddlstream.algorithms.search import solve_from_pddl
import taskplan_multi
import random

def run_pddl():
    # preparing pddl as input to the solver
    seed = 19
    pddl = {}
    random.seed(seed)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(seed=seed)
    print(restaurant.containers)
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
    run_pddl()
