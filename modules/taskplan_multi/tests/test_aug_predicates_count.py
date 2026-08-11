"""
Compare augmentation predicate counts between _aug_predicates_for_food_robot
and _aug_predicates_for_any_robot without running the PDDL planner.

Key design difference being tested:
  _aug_predicates_for_food_robot — needs myopic plan context (used_items,
      used_containers) to select candidate locations; generates nothing without it.
  _aug_predicates_for_any_robot  — needs only the robot + current restaurant state;
      enumerates all accessible items × all accessible locations × all state variants.

get_anticipated_cost is mocked so all candidates pass the cost filter,
revealing the raw candidate pool each method explores.
random.sample is intercepted to record the pool size before the 200-cap.

Run via Docker:
    make test PYTEST_FILTER="test_aug_predicates_count" USE_GPU=false TEST_ADDITIONAL_ARGS="-s"
"""

import random
from unittest.mock import patch

import pytest
import taskplan_multi.environments.restaurant
import taskplan_multi.pddl.task
import taskplan_multi.planners.myopic_planner
import taskplan_multi.planners.anticipatory_planner
from taskplan_multi.utilities.restaurant_primitives import (
    KITCHEN_CONTAINERS, SERVING_ROOM_CONTAINERS,
    COOK_BOT_RESTRICT, SERVER_BOT_RESTRICT, CLEANER_BOT_RESTRICT,
)


def _fmt(n):
    for unit, threshold in [('T', 1e12), ('B', 1e9), ('M', 1e6), ('K', 1e3)]:
        if n >= threshold:
            return f"{n / threshold:.2f}{unit}"
    return str(n)


def _count_aug_candidates(robot, restaurant, other_bots=frozenset()):
    """Total candidates _aug_predicates_for_any_robot would evaluate: (1+M)^N - 1."""
    if robot == 'cook_bot':
        restrict = COOK_BOT_RESTRICT
    elif robot == 'server_bot':
        restrict = SERVER_BOT_RESTRICT
    else:
        restrict = CLEANER_BOT_RESTRICT

    accessible_conts = [c for c in KITCHEN_CONTAINERS + SERVING_ROOM_CONTAINERS if c not in restrict]
    if robot == 'cook_bot' and len(other_bots) > 0:
        accessible_conts += ['servingtable1', 'servingtable2']
    if robot in ('server_bot', 'cleaner_bot') and 'cook_bot' in other_bots:
        accessible_conts.append('stove')

    seen = set()
    for cont in accessible_conts:
        for obj in restaurant.get_objects_by_container_name(cont):
            seen.add(obj['assetId'])

    N = len(seen)
    M = len(accessible_conts)
    return (1 + M) ** N - 1

SEED = 5

TASK_SCENARIOS = [
    ('cook_bot',   'cook_pasta_bowl_stove',  taskplan_multi.pddl.task.prep_item('pasta', 'bowl', 'stove')),
    ('server_bot', 'serve_milk_mug_table1',  taskplan_multi.pddl.task.serve_item('milk', 'mug', 'servingtable1')),
]


class _MockAnticipatoryPlanner(taskplan_multi.planners.anticipatory_planner.AntcipatoryPlanner):
    """Skips neural network loading; get_anticipated_cost always returns 0."""
    def __init__(self):
        self.concern = 'joint'
        self.myopic_planner = taskplan_multi.planners.myopic_planner.MyopicPlanner()

    def get_anticipated_cost(self, restaurant):
        return 0.0


def make_restaurant(active_robot):
    random.seed(SEED)
    return taskplan_multi.environments.restaurant.RESTAURANT(
        seed=SEED,
        agents=['cook_bot', 'cleaner_bot', 'server_bot'],
        active=active_robot,
    )


def _call_with_pool_tracking(method, *args, **kwargs):
    """Call an aug method and record the candidate pool size before the 200-sample cap."""
    pool_sizes = []
    original_sample = random.sample

    def tracking_sample(population, k):
        pool_sizes.append(len(population))
        return original_sample(population, k)

    with patch('random.sample', side_effect=tracking_sample):
        predicates = method(*args, **kwargs)

    return predicates, (pool_sizes[0] if pool_sizes else 0)


@pytest.fixture(scope='module')
def myopic_planner():
    return taskplan_multi.planners.myopic_planner.MyopicPlanner()


@pytest.fixture(scope='module')
def mock_ap():
    return _MockAnticipatoryPlanner()


@pytest.mark.parametrize('robot,scenario_name,task', TASK_SCENARIOS, ids=[s[1] for s in TASK_SCENARIOS])
def test_aug_predicates_count(myopic_planner, mock_ap, robot, scenario_name, task):
    print(f"\n\n{'='*68}")
    print(f"  Scenario : {scenario_name}  |  Robot: {robot}")
    print(f"{'='*68}")

    # --- food_robot method: needs myopic plan context to select candidate locations ---
    restaurant = make_restaurant(robot)
    plan, myopic_cost = myopic_planner.get_cost_and_state_from_task(restaurant, task)
    assert plan is not None, f"Myopic planner failed for '{scenario_name}'"

    used_items, used_containers, other_bots = mock_ap._extract_plan_context(plan)
    print(f"\n  Myopic cost      : {myopic_cost}")
    print(f"  used_items       : {sorted(used_items)}")
    print(f"  used_containers  : {sorted(used_containers)}")
    # print(f"  other_bots (from plan) : {sorted(other_bots)}")

    restaurant = make_restaurant(robot)
    food_preds, food_pool = _call_with_pool_tracking(
        mock_ap._aug_predicates_for_food_robot,
        robot, restaurant, used_items, used_containers, float('inf'), other_bots=other_bots,
    )

    # --- any_robot method: needs only the robot + current state, no myopic plan ---
    # other_bots derived directly from the restaurant's agent list
    restaurant = make_restaurant(robot)
    task_other_bots = set(restaurant.agent_list) - {robot}
    # print(f"  other_bots (from restaurant) : {sorted(task_other_bots)}")

    any_total = _count_aug_candidates(robot, restaurant, other_bots=task_other_bots)

    restaurant = make_restaurant(robot)
    any_preds = mock_ap._aug_predicates_for_any_robot(
        robot, restaurant, float('inf'), other_bots=task_other_bots,
    )

    W = 72
    print(f"\n  {'Metric':<38} {'Focused':>15} {'Exhaustive':>15}")
    print(f"  {'─' * W}")
    print(f"  {'Number of predicates generated':<38} {_fmt(len(food_preds)):>15} {_fmt(any_total):>15}")
    # print(f"  {'Evaluated by oracle':<38} {_fmt(min(food_pool, 200)):>15} {_fmt(any_total):>15}")
    # print(f"  {'Returned (passed cost filter)':<38} {_fmt(len(food_preds)):>15} {_fmt(len(any_preds)):>15}")
    print(f"  {'─' * W}")
    print()

    # assert len(food_preds) >= 0
    # assert len(any_preds) >= 0
    # assert any_pool >= food_pool, (
    #     f"_aug_predicates_for_any_robot should explore at least as large a pool "
    #     f"as the food-specific method (any={any_pool}, food={food_pool})"
    # )
