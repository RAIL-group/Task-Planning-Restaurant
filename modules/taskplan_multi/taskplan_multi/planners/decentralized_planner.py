import taskplan_multi.pddl.domain_decentralized
import taskplan_multi.pddl.problem_decentralized
from pddlstream.algorithms.search import solve_from_pddl

FAILED_COST = 2000

_FIXED_COST = {
    'pick': 10, 'place': 10, 'mix': 10, 'serve': 10,
    'wash': 30, 'restock': 30,
}


def _defer_waits(plan):
    """
    Move every 'wait-for' to sit immediately before the specific pick it
    unblocks, instead of wherever the classical search happened to place
    it. This is safe because wait-for's only precondition/effect is the
    `reserved` flag — it never touches rob-at/hand-is-free/is-holding, so
    it has no real ordering dependency on anything else in the plan except
    that one pick. Everything else keeps its original relative order.
    """
    plan = list(plan)
    for wait in [a for a in plan if a.name == 'wait-for']:
        plan.remove(wait)
        item = wait.args[1]
        insert_at = next(
            (i for i, a in enumerate(plan) if a.name == 'pick' and a.args[1] == item),
            len(plan))
        plan.insert(insert_at, wait)
    return plan


class DecentralizedPlanner:
    """
    Local, single-agent classical planner used by every robot in the
    decentralized protocol:

        1. robot gets a task
        2. robot plans locally (this class), against only its own
           single-agent PDDL problem — the other robot is never an object
           in it, so nothing here can produce another robot's action
        3. caller broadcasts the resulting timed plan
        4. other robots record the items it touches as reservations
        5. the next robot to plan passes those reservations in here, and
           any `pick` on a still-reserved item is blocked until the plan
           explicitly runs `wait-for` on it

    One instance is reused across robots/calls; `robot` is always passed in
    explicitly rather than being fixed on the planner.
    """

    def __init__(self, domain=None):
        self.domain = domain or taskplan_multi.pddl.domain_decentralized.get_domain()

    def get_cost_and_state_from_task(self, restaurant, robot, task, reservations=None):
        pddl_problem = taskplan_multi.pddl.problem_decentralized.get_problem(
            restaurant, robot, task, reservations)
        plan, cost = solve_from_pddl(
            self.domain,
            pddl_problem,
            planner='ff-astar',
            max_planner_time=300,
        )

        if plan:
            move_cost = sum(
                restaurant.known_cost[action.args[1]][action.args[2]]
                for action in plan if action.name == 'move'
            )
            cost += move_cost

        return plan, cost

    def get_timed_plan(self, restaurant, robot, task, reservations=None, arrival_time=0.0):
        """
        Same shape as MyopicPlanner.get_timed_plan: returns
        (steps, pddl_plan, cost) with steps annotated with 't_start'/'t_end'
        (relative to arrival_time) plus 'action'/'args'/'description'.

        The one addition is 'wait-for': the classical planner only decides
        *where in the plan* to insert it (whenever it needs a reserved
        item); its actual duration is filled in here from the live
        `reservations` table — whatever's left of the other robot's
        reservation once this robot's own plan-so-far has caught up to it.

        Classical search has no notion of wall-clock time, so it drops
        wait-for wherever is cheapest to search, which is often first —
        that would stall the whole plan behind the wait instead of using
        that time productively. _defer_waits() pushes each wait-for to
        right before the specific pick it unblocks, so everything that
        doesn't actually need the reserved item runs first and eats into
        the wait.
        """
        reservations = reservations or {}
        plan, cost = self.get_cost_and_state_from_task(restaurant, robot, task, reservations)
        if plan is None:
            return None, None, cost
        plan = _defer_waits(plan)

        steps = []
        t = 0.0
        for action in plan:
            name, args = action.name, action.args

            if name == 'move':
                duration = restaurant.known_cost[args[1]][args[2]]
                desc = f"{args[0]} moves {args[1]} -> {args[2]}"
            elif name == 'pick':
                duration = _FIXED_COST['pick']
                desc = f"{args[0]} picks {args[1]} from {args[2]}"
            elif name == 'place':
                duration = _FIXED_COST['place']
                desc = f"{args[0]} places {args[1]} at {args[2]}"
            elif name == 'wash':
                duration = _FIXED_COST['wash']
                desc = f"{args[0]} washes {args[1]}"
            elif name == 'mix':
                duration = _FIXED_COST['mix']
                desc = f"{args[0]} mixes {args[1]} into {args[2]} at {args[3]}"
            elif name == 'serve':
                duration = _FIXED_COST['serve']
                desc = f"{args[0]} serves {args[1]} in {args[2]} at {args[3]}"
            elif name == 'restock':
                duration = _FIXED_COST['restock']
                desc = f"{args[0]} restocks {args[1]}"
            elif name == 'wait-for':
                item = args[1]
                res = reservations.get(item)
                now = arrival_time + t
                until = res['until'] if res else now
                duration = max(0.0, until - now)
                holder = res['robot'] if res else 'someone'
                desc = f"{args[0]} waits for '{item}' — held by {holder} until t={until:.1f}"
            else:
                duration = 0
                desc = f"{name} {' '.join(str(a) for a in args)}"

            steps.append({
                't_start': t,
                't_end': t + duration,
                'action': name,
                'args': args,
                'description': desc,
            })
            t += duration

        return steps, plan, cost
