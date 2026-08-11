from collections import namedtuple

import taskplan_multi.pddl.domain_decentralized_temporal
import taskplan_multi.pddl.problem_decentralized_temporal
import taskplan_multi.planners.tfd_solver as tfd_solver

FAILED_COST = 2000

Action = namedtuple('Action', ['name', 'args'])


def _describe(name, args):
    if name == 'move':
        return f"{args[0]} moves {args[1]} -> {args[2]}"
    if name == 'pick':
        return f"{args[0]} picks {args[1]} from {args[2]}"
    if name == 'place':
        return f"{args[0]} places {args[1]} at {args[2]}"
    if name == 'wash':
        return f"{args[0]} washes {args[1]}"
    if name == 'mix':
        return f"{args[0]} mixes {args[1]} into {args[2]} at {args[3]}"
    if name == 'serve':
        return f"{args[0]} serves {args[1]} in {args[2]} at {args[3]}"
    if name == 'restock':
        return f"{args[0]} restocks {args[1]}"
    if name == 'wait-for':
        return f"{args[0]} waits (in the background) for '{args[1]}' to be released"
    return f"{name} {' '.join(args)}"


class TemporalDecentralizedPlanner:
    """
    Temporal counterpart of DecentralizedPlanner: same decentralized
    protocol (one robot, one single-agent problem, reservations broadcast
    between robots), but solved with Temporal Fast Downward against a
    :durative-action domain instead of classical FF/A* against an
    instantaneous-action one.

    `wait-for` still exists here (the TFD build available doesn't support
    Timed Initial Literals), but it's a fixed-duration durative action with
    no rob-at footprint at all -- see domain_decentralized_temporal's
    docstring. Because nothing marks it mutex with the rest of the plan,
    TFD is free to run it in the background while genuinely concurrent,
    causally-independent physical actions happen alongside it, instead of
    it blocking everything else the way the classical planner's
    total-order `wait-for` tended to (see decentralized_planner
    ._defer_waits, which only partially compensated for that).

    get_timed_plan returns steps whose 't_start'/'t_end' are exactly what
    TFD solved for -- no post-hoc duration bookkeeping needed, unlike the
    classical planner's get_timed_plan.
    """

    def __init__(self, domain=None, max_planner_time=120):
        self.domain = domain or taskplan_multi.pddl.domain_decentralized_temporal.get_domain()
        self.max_planner_time = max_planner_time

    def get_timed_plan(self, restaurant, robot, task, reservations=None, arrival_time=0.0):
        reservations = reservations or {}
        pddl_problem = taskplan_multi.pddl.problem_decentralized_temporal.get_problem(
            restaurant, robot, task, reservations, arrival_time)

        solved = tfd_solver.solve_temporal_pddl(
            self.domain, pddl_problem, max_planner_time=self.max_planner_time)
        if solved is None:
            return None, None, FAILED_COST

        steps = []
        plan = []
        for start, name, args, duration in solved:
            steps.append({
                't_start': start,
                't_end': start + duration,
                'action': name,
                'args': args,
                'description': _describe(name, args),
            })
            plan.append(Action(name, args))

        cost = steps[-1]['t_end'] if steps else 0.0   # makespan, for logging only
        return steps, plan, cost
