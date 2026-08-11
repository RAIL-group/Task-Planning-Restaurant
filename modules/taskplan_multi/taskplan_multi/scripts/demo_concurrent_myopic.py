"""
Decentralized concurrent restaurant simulation using MyopicPlanner.

When a robot receives a task it announces its full timestamped plan to all
others.  When a second robot arrives it is shown which items are currently
locked and when/where they will be free, then plans against the broadcast
state (current world state + committed future effects of every running robot).

Item locks are load-bearing, not just cosmetic: if a robot is about to pick
up an item that another robot currently holds, its remaining schedule is
pushed back until that lock clears (and everything after that pick shifts
with it), so two robots can never physically hold the same item at once.

Scenario:
  cook_bot   (arrives t=0)   → prep pasta in bowl on stove
  server_bot (arrives t=20)  → serve milk in mug at servingtable1

Run via Docker:
    make concurrent-demo USE_GPU=false
"""

import copy
import heapq
import random
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import taskplan_multi.environments.restaurant
import taskplan_multi.pddl.task
import taskplan_multi.planners.myopic_planner

SEED  = 5
ROBOTS = ['cook_bot', 'server_bot', 'cleaner_bot']

TASKS: List[Tuple[float, str, Any]] = [
    (0.0,  'cook_bot',   taskplan_multi.pddl.task.prep_item('pasta', 'bowl', 'stove')),
    (20.0, 'server_bot', taskplan_multi.pddl.task.serve_item('milk', 'bowl', 'servingtable1')),
]

BOARD_W = 70

# A minimal stand-in for a solved-plan action, used to apply the effect of a
# single already-executed step to the shared world state via
# restaurant.get_final_state_from_plan(), which only reads .name / .args.
SimpleAction = namedtuple('SimpleAction', ['name', 'args'])


@dataclass(order=True)
class Event:
    time:  float
    seq:   int
    robot: str = field(compare=False)
    kind:  str = field(compare=False)  # 'TASK_ARRIVAL' | 'ACTION_END'
    data:  Any = field(compare=False)


# ---------------------------------------------------------------------------
# Item-lock extraction
# ---------------------------------------------------------------------------

def _item_locks_from_steps(robot: str, steps: List[dict], base_time: float) -> Dict[str, dict]:
    """
    Walk a timed plan and return {item: {robot, from, until, at}} for each
    item the robot picks up.  A lock opens at pick-start and closes when the
    item is released (place / mix / serve / restock).
    """
    held:  Dict[str, float] = {}   # item -> absolute t_pick
    locks: Dict[str, dict]  = {}

    for step in steps:
        action, args = step['action'], step['args']
        t_end = base_time + step['t_end']

        if action == 'pick':
            held[args[1]] = base_time + step['t_start']

        elif action in ('place', 'mix', 'serve', 'restock'):
            item = args[1]
            if item in held:
                final_loc = args[2] if action == 'place' else f'{action}d'
                locks[item] = {
                    'robot': robot,
                    'from':  held.pop(item),
                    'until': t_end,
                    'at':    final_loc,
                }

    # Items still held at plan end (e.g. wash without subsequent place yet)
    if steps:
        plan_end = base_time + steps[-1]['t_end']
        for item, t_pick in held.items():
            locks[item] = {'robot': robot, 'from': t_pick,
                           'until': plan_end, 'at': '?'}
    return locks


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    restaurant = taskplan_multi.environments.restaurant.RESTAURANT(
        seed=SEED,
        agents=ROBOTS,
        active='cook_bot',
    )
    planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()

    robot_timed_steps: Dict[str, List[dict]]     = {r: [] for r in ROBOTS}
    robot_pddl_plans:  Dict[str, Optional[list]] = {r: None for r in ROBOTS}
    robot_busy:        Dict[str, bool]            = {r: False for r in ROBOTS}
    robot_pending:     Dict[str, Optional[Any]]   = {r: None for r in ROBOTS}
    robot_status:      Dict[str, Optional[Tuple[str, float, float]]] = {r: None for r in ROBOTS}
    item_locks:        Dict[str, dict]            = {}

    # Shared world state — reflects all effects of completed actions.
    # Starts at the initial restaurant state and is updated incrementally,
    # action-by-action, as robots actually finish releasing items (not just
    # when a robot's whole plan is done) so it always reflects real,
    # up-to-the-moment ground truth rather than an end-of-plan snapshot.
    current_state = copy.deepcopy(restaurant.get_current_object_state())

    event_queue: List[Event] = []
    _seq = 0

    def push(t: float, robot: str, kind: str, data: Any) -> None:
        nonlocal _seq
        heapq.heappush(event_queue, Event(t, _seq, robot, kind, data))
        _seq += 1

    def get_broadcast_state() -> list:
        """
        Project current_state forward through every committed robot plan.
        Tells a newly-planning robot what the world will look like once all
        running robots finish — analogous to broadcast_state() in the A* demo.
        """
        restaurant.update_container_props(current_state)
        for r, plan in robot_pddl_plans.items():
            if plan:
                final = restaurant.get_final_state_from_plan(plan)
                restaurant.update_container_props(final)
        broadcast = copy.deepcopy(restaurant.get_current_object_state())
        restaurant.update_container_props(current_state)   # restore
        return broadcast

    def apply_release_effect(step: dict) -> None:
        """
        Fold a single just-completed release action (place/mix/serve/restock)
        into current_state immediately, so the shared world state is always
        accurate at the instant it happens rather than only once an entire
        robot plan finishes.
        """
        nonlocal current_state
        action = SimpleAction(step['action'], step['args'])
        restaurant.update_container_props(current_state)
        current_state = copy.deepcopy(restaurant.get_final_state_from_plan([action]))
        restaurant.update_container_props(current_state)

    def resolve_start_conflict(robot: str, steps: List[dict]) -> None:
        """
        If the next queued step is a 'pick' of an item another robot
        currently holds, push this robot's *entire* remaining schedule back
        until that lock clears (own future lock announcements shift with
        it), instead of letting two robots grab the same item at once.
        """
        if not steps or steps[0]['action'] != 'pick':
            return
        step = steps[0]
        item = step['args'][1]

        for _ in range(5):   # bounded: guards against pathological re-conflicts
            lock = item_locks.get(item)
            if lock is None or lock['robot'] == robot or lock['until'] <= step['abs_start']:
                return

            orig_start = step['abs_start']
            delta = lock['until'] - orig_start
            for s in steps:
                s['abs_start'] += delta
                s['abs_end']   += delta
            for info in item_locks.values():
                if info['robot'] == robot and info['from'] >= orig_start:
                    info['from']  += delta
                    info['until'] += delta

            print(f"  ⏳ {robot} waits for '{item}' — held by {lock['robot']} "
                  f"until t={lock['until']:.1f}  (delayed by {delta:.1f})")
            if lock['at'] not in ('?', step['args'][2]):
                print(f"     ⚠ {robot}'s plan may be stale: expected '{item}' at "
                      f"'{step['args'][2]}', but {lock['robot']} will leave it at '{lock['at']}'")

    def show_resource_info(t: float, planning_robot: str) -> None:
        active = {item: info for item, info in item_locks.items()
                  if info['until'] > t}
        print(f"  {'╌' * BOARD_W}")
        if active:
            print(f"  Locks visible to {planning_robot} at t={t:.0f}:")
            for item, info in sorted(active.items()):
                print(f"    {item:<16} ← {info['robot']:<12} "
                      f"t={info['from']:.0f} → t={info['until']:.0f}"
                      f"  will be at: {info['at']}")
        else:
            print(f"  No active resource locks at t={t:.0f}")
        print(f"  Planning {planning_robot} against broadcast state ...")
        print(f"  {'╌' * BOARD_W}")

    def announce_plan(t: float, robot: str, steps: List[dict]) -> None:
        plan_end = t + steps[-1]['t_end']
        print(f"  ┌─ {robot}  [{len(steps)} steps, finishes t={plan_end:.0f}]")
        for s in steps:
            print(f"  │  t={t + s['t_start']:>6.1f} – {t + s['t_end']:<6.1f}"
                  f"  {s['description']}")
        print(f"  └{'─' * BOARD_W}")

    def render_row(robot: str) -> str:
        st = robot_status[robot]
        if st is None:
            return 'idle'
        desc, start, end = st
        return f"{desc:<44} [t={start:>6.1f} → {end:>6.1f}]"

    def print_board(t: float, headline: str) -> None:
        """
        User-facing 'now' view: one fixed row per robot, showing exactly
        what it's doing and the absolute time span of that action, framed by
        horizontal rules so concurrent activity is easy to read at a glance.
        """
        print(f"\n  {'─' * BOARD_W}")
        print(f"  t={t:>6.1f}  {headline}")
        print(f"  {'─' * BOARD_W}")
        for i, r in enumerate(ROBOTS, start=1):
            print(f"  Row {i} {r:<12}│ {render_row(r)}")
        print(f"  {'─' * BOARD_W}")

    def plan_and_start(t: float, robot: str, task: Any) -> None:
        show_resource_info(t, robot)

        # Plan against the projected broadcast state
        broadcast = get_broadcast_state()
        restaurant.update_container_props(broadcast)
        restaurant.active_robot = robot
        steps, pddl_plan, cost = planner.get_timed_plan(restaurant, task)
        restaurant.update_container_props(current_state)   # restore

        if steps is None:
            print(f"  t={t:>6.1f} | {robot} FAILED — no plan found")
            return

        announce_plan(t, robot, steps)

        # Stamp absolute times onto each step
        for s in steps:
            s['abs_start'] = t + s['t_start']
            s['abs_end']   = t + s['t_end']

        robot_timed_steps[robot] = list(steps)
        robot_pddl_plans[robot]  = pddl_plan
        robot_busy[robot]        = True
        item_locks.update(_item_locks_from_steps(robot, steps, t))

        # Resolve any conflict on the very first action before committing to it
        resolve_start_conflict(robot, robot_timed_steps[robot])

        # Kick off first action
        first = robot_timed_steps[robot][0]
        robot_status[robot] = (first['description'], first['abs_start'], first['abs_end'])
        print_board(t, f"{robot} ▶ {first['description']}")
        push(first['abs_end'], robot, 'ACTION_END', first)

    # Seed task-arrival events
    for arrival_t, robot, task in TASKS:
        push(arrival_t, robot, 'TASK_ARRIVAL', task)

    print(f"\n{'═' * BOARD_W}")
    print(f"  CONCURRENT MYOPIC PLANNING SIMULATION  (seed={SEED})")
    print(f"{'═' * BOARD_W}")

    while event_queue:
        ev = heapq.heappop(event_queue)
        t, robot = ev.time, ev.robot

        if ev.kind == 'TASK_ARRIVAL':
            task = ev.data
            print(f"\n{'─' * BOARD_W}")
            print(f"  t={t:>6.1f}  TASK_ARRIVAL → {robot}")
            if robot_busy[robot]:
                robot_pending[robot] = task
                print(f"           {robot} is busy — task deferred")
            else:
                plan_and_start(t, robot, task)

        elif ev.kind == 'ACTION_END':
            step  = ev.data
            steps = robot_timed_steps[robot]

            # Release item lock and fold this action's effect into the
            # shared world state the instant it actually happens.
            if step['action'] in ('place', 'mix', 'serve', 'restock'):
                item_locks.pop(step['args'][1], None)
                apply_release_effect(step)

            if steps:
                steps.pop(0)

            if steps:
                resolve_start_conflict(robot, steps)
                nxt = steps[0]
                robot_status[robot] = (nxt['description'], nxt['abs_start'], nxt['abs_end'])
                print_board(t, f"{robot} ■ done: {step['description']}")
                push(nxt['abs_end'], robot, 'ACTION_END', nxt)
            else:
                robot_pddl_plans[robot] = None
                robot_busy[robot] = False
                robot_status[robot] = None
                print_board(t, f"{robot} ✓ DONE — idle  (finished: {step['description']})")

                if robot_pending[robot] is not None:
                    pending = robot_pending[robot]
                    robot_pending[robot] = None
                    print(f"           {robot} resuming deferred task")
                    plan_and_start(t, robot, pending)

    print(f"\n{'═' * BOARD_W}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'═' * BOARD_W}\n")


if __name__ == '__main__':
    main()
