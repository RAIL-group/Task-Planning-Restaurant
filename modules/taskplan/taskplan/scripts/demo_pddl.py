from pddlstream.algorithms.search import solve_from_pddl

import taskplan


def run_pddl():
    # preparing pddl as input to the solver
    pddl = {}
    restaurant = taskplan.environments.restaurant.RESTAURANT(seed=1)
    pddl['domain'] = taskplan.pddl.domain.get_domain()
    pddl['problem'] = taskplan.pddl.problem.get_problem(restaurant)
    pddl['planner'] = 'max-astar'

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])
    print(plan)

    if plan:
        for p in plan:
            print(p)
        print(cost)


if __name__ == "__main__":
    run_pddl()
