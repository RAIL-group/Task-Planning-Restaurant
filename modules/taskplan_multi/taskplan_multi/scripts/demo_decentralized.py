"""
Decentralized two-robot restaurant planning: broadcast + reservations,
each robot working through its own queue of tasks.

Protocol (no shared world-clock event loop):

  1. A robot pulls the next task off its own queue as soon as it's free
     (i.e. once its previous task's plan has finished) -- one robot's
     queue never blocks on the other's.
  2. It plans locally with TemporalDecentralizedPlanner -- a single-agent
     :durative-action plan solved by Temporal Fast Downward against a PDDL
     problem where it is the *only* robot object. Nothing here can ever
     produce another robot's action.
  3. It broadcasts the resulting timed plan (real concurrent-schedule
     timing straight from the solver, not hand-rolled afterward).
  4. Every other robot records the items that plan touches as reservations:
     an absolute [from, until) window per item, spanning from the first
     time the plan touches it to the last. The plan's effects (positions,
     dirty/empty flags) are also folded into the shared world state right
     away, so the next robot to plan -- whether that's the same robot's
     next queued task or the other robot -- sees a physically consistent
     picture, not a stale snapshot.
  5. When a robot plans, every item still reserved by someone else is
     written into its problem as `reserved` plus a precomputed
     `release-wait` duration (the TFD build in use has no timed-initial-
     literal support, so the release time has to be a plan-time constant,
     not a fact the world flips on its own). `wait-for` clears `reserved`
     after exactly that long -- but as a durative action with no rob-at
     footprint at all, so nothing marks it mutex with anything else the
     robot does. TFD is free to run it fully in the background, starting
     at t=0, while genuinely concurrent, causally-independent physical
     actions (fetching an unrelated ingredient, say) happen alongside it --
     unlike the classical planner's `wait-for`, which a non-temporal search
     had no reason to place anywhere but the front of the plan, stalling
     everything else behind it.

Two robots become "ready to plan" at the same instant in two cases: both
get their very first task at t=0, or (less predictably) their queues
happen to sync back up later. Either way, whoever is ready earliest always
goes first; an exact tie is broken with a coin flip.

Scenario: each robot works through 3 tasks, each needing a differently
typed container (bowl / pan / mug) so that a container mix'd dirty by one
robot's prep task doesn't block the other robot's later use of that same
type (there are two of each container type; the restaurant has no
cleaner_bot to re-wash one, so tasks are deliberately spread across
distinct types/foods rather than forced onto one specific instance).
Because there's only one milk asset, server_bot's later rounds need it
restocked -- which its own plan will do, since restock is one of its
actions -- once state propagation makes the emptied milk visible to it.

Whether a reservation actually triggers a real wait on a given round
depends on whether both robots' plans happen to want the *same*
bowl/pan/mug instance (there are two of each) -- this is intentionally
left to the planner rather than hand-forced, since forcing both robots
onto one specific instance would make one of the two tasks physically
unsolvable (prep leaves a container dirty; serve requires it clean; there
is no cleaner_bot in this scenario to reset it).

Run via Docker (needs Temporal Fast Downward, Docker-only):
    make decentralized-demo USE_GPU=false
"""

import random

import taskplan_multi.environments.restaurant
import taskplan_multi.pddl.task
import taskplan_multi.planners.temporal_decentralized_planner

SEED = 5
ROBOTS = ['cook_bot', 'server_bot']

# Each robot works through these in order; robot2's queue never waits on
# robot1's -- only on robot1's *specific reservations*.
TASK_QUEUES = {
    'cook_bot': [
        taskplan_multi.pddl.task.prep_item('pasta',  'bowl', 'stove'),
        taskplan_multi.pddl.task.prep_item('oats',   'pan',  'stove'),
        taskplan_multi.pddl.task.prep_item('cereal', 'mug',  'stove'),
    ],
    'server_bot': [
        taskplan_multi.pddl.task.serve_item('milk', 'bowl', 'servingtable1'),
        taskplan_multi.pddl.task.serve_item('milk', 'pan',  'servingtable2'),
        taskplan_multi.pddl.task.serve_item('milk', 'mug',  'servingtable1'),
    ],
}

BOARD_W = 70
RELEASE_ACTIONS = ('pick', 'place', 'mix', 'serve', 'restock')
ITEM_ARG_INDEXES = {
    'pick': (1,), 'place': (1,), 'restock': (1,),
    'mix': (1, 2), 'serve': (1, 2),
}


def reservation_windows(steps, base_time):
    """
    For every item a broadcast plan references, the [from, until) window
    it's unavailable to everyone else: first touch to last touch.

    mix/serve take *two* items (food at args[1], bowl/container at args[2])
    — both have to be captured, or the container's reservation would end
    early (at its place/pick) even though the plan still uses it later.
    """
    span = {}
    for step in steps:
        if step['action'] not in RELEASE_ACTIONS:
            continue
        start = base_time + step['t_start']
        end = base_time + step['t_end']
        for idx in ITEM_ARG_INDEXES[step['action']]:
            item = step['args'][idx]
            if item not in span:
                span[item] = [start, end]
            else:
                span[item][0] = min(span[item][0], start)
                span[item][1] = max(span[item][1], end)
    return span


def final_robot_location(robot, plan, current_loc):
    """Where a robot actually ends up: the destination of its last move."""
    loc = current_loc
    for action in plan:
        if action.name == 'move' and action.args[0] == robot:
            loc = action.args[2]
    return loc


def print_plan(robot, t, steps):
    plan_end = t + max(s['t_end'] for s in steps)
    print(f"  ┌─ {robot} broadcasts  [{len(steps)} steps, finishes t={plan_end:.1f}]")
    for s in steps:
        print(f"  │  t={t + s['t_start']:>6.1f} - {t + s['t_end']:<6.1f}"
              f"  {s['description']}")
    print(f"  └{'─' * BOARD_W}")


def print_reservations(reservations):
    if not reservations:
        print("    (none)")
        return
    for item, info in sorted(reservations.items()):
        print(f"    {item:<12} reserved by {info['robot']:<10} "
              f"t={info['from']:.1f} -> t={info['until']:.1f}")


def main() -> None:
    random.seed(SEED)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(
        seed=SEED, agents=ROBOTS, active='cook_bot')
    planner = taskplan_multi.planners.temporal_decentralized_planner.TemporalDecentralizedPlanner()

    queues = {r: list(tasks) for r, tasks in TASK_QUEUES.items()}
    total_tasks = {r: len(tasks) for r, tasks in queues.items()}
    done_count = {r: 0 for r in ROBOTS}
    next_available = {r: 0.0 for r in ROBOTS}   # when each robot is next free to plan
    reservations = {}   # item -> {'robot': ..., 'from': t, 'until': t}

    print(f"\n{'═' * BOARD_W}")
    print("  DECENTRALIZED TWO-ROBOT PLANNING  (broadcast + reservations, queued tasks)")
    print(f"{'═' * BOARD_W}")

    while any(queues[r] for r in ROBOTS):
        eligible = [r for r in ROBOTS if queues[r]]
        t_min = min(next_available[r] for r in eligible)
        ready = [r for r in eligible if next_available[r] == t_min]
        if len(ready) > 1:
            random.shuffle(ready)
            print(f"\n  t={t_min:>6.1f}  {len(ready)} robots ready simultaneously "
                  f"-- coin-flip tie-break order: {ready}")

        for robot in ready:
            task = queues[robot].pop(0)
            done_count[robot] += 1
            print(f"\n{'─' * BOARD_W}")
            print(f"  t={t_min:>6.1f}  {robot} starts task "
                  f"{done_count[robot]}/{total_tasks[robot]}")

            active_reservations = {item: info for item, info in reservations.items()
                                    if info['until'] > t_min}
            print("  Reservations known at planning time:")
            print_reservations(active_reservations)

            restaurant.active_robot = robot
            steps, pddl_plan, cost = planner.get_timed_plan(
                restaurant, robot, task, active_reservations, arrival_time=t_min)

            if steps is None:
                print(f"  {robot} FAILED to find a plan (cost={cost}) -- "
                      f"skipping this task, moving to its next one")
                next_available[robot] = t_min
                continue

            print_plan(robot, t_min, steps)

            for item, (start, end) in reservation_windows(steps, t_min).items():
                reservations[item] = {'robot': robot, 'from': start, 'until': end}

            # Fold this plan's effects into the shared world state and this
            # robot's real position, so the *next* problem generated for
            # either robot reflects what actually happened here.
            restaurant.update_container_props(
                restaurant.get_final_state_from_plan(pddl_plan))
            restaurant.restaurant[robot]['rob_at'] = final_robot_location(
                robot, pddl_plan, restaurant.restaurant[robot]['rob_at'])

            # max(), not steps[-1]: TFD's plan is sorted by *start* time,
            # and while a single robot's own actions can never overlap
            # (every stationary action holds `rob-at` for its whole
            # duration), taking the max is a cheap, always-correct
            # makespan regardless of that assumption.
            next_available[robot] = t_min + max(s['t_end'] for s in steps)

    print(f"\n{'═' * BOARD_W}")
    print("  ALL ROBOTS FINISHED THEIR QUEUES")
    print(f"{'═' * BOARD_W}\n")


if __name__ == '__main__':
    main()
