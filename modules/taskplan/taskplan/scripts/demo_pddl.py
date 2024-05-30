from pddlstream.algorithms.search import solve_from_pddl

import taskplan


def run_pddl():
    # preparing pddl as input to the solver
    pddl = {}
    pddl['domain'] = taskplan.pddl.domain.get_domain()
    pddl['problem'] = taskplan.pddl.problem.get_problem()
    pddl['planner'] = 'max-astar'

    plan, cost = solve_from_pddl(pddl['domain'], pddl['problem'], planner=pddl['planner'])

    if plan:
        for p in plan:
            print(p)
        print(cost)


if __name__ == "__main__":
    run_pddl()
