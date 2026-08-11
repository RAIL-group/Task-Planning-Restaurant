"""
Compare LLMPlanner vs MyopicPlanner on a representative set of tasks.

Run via Docker:
    make test PYTEST_FILTER="test_llm_vs_myopic" USE_GPU=false TEST_ADDITIONAL_ARGS="-s"
"""

import random
import time

import pytest
import taskplan_multi.environments.restaurant
import taskplan_multi.pddl.task
import taskplan_multi.planners.myopic_planner
import taskplan_multi.planners.llm_planner

SEED = 5
MODEL = 'claude-opus-4-8'
FAILED_COST = 2000

TASK_SCENARIOS = [
    ('cook_bot',     'cook_pasta_bowl_stove',      taskplan_multi.pddl.task.prep_item('pasta', 'bowl', 'stove')),
    # ('cook_bot',     'cook_oats_pan_countertop',   taskplan_multi.pddl.task.prep_item('oats', 'pan', 'countertop')),
    # ('server_bot',   'serve_milk_mug_table1',      taskplan_multi.pddl.task.serve_item('milk', 'mug', 'servingtable1')),
    # ('server_bot',   'restock_cereal',             taskplan_multi.pddl.task.restock_something('cereal')),
    # ('cleaner_bot',  'clean_pan1',                 taskplan_multi.pddl.task.clean_something('pan1')),
    # ('cleaner_bot',  'clean_both_bowls',           taskplan_multi.pddl.task.clean_both_item('bowl')),
]


def make_restaurant(active_robot):
    random.seed(SEED)
    return taskplan_multi.environments.restaurant.RESTAURANT(
        seed=SEED,
        agents=['cook_bot', 'cleaner_bot', 'server_bot'],
        active=active_robot,
    )


def print_item_states(restaurant, item_names):
    """Print location and dirty/empty status for named items."""
    targets = set(item_names)
    found = {}
    for container in restaurant.containers:
        children = container.get('children') or []
        for child in children:
            if child['assetId'] in targets:
                state = []
                if child.get('dirty') == 1:
                    state.append('dirty')
                if child.get('empty') == 1:
                    state.append('empty')
                found[child['assetId']] = (container['assetId'], state or ['clean/full'])
    print(f"\n  --- Initial item states ---")
    for name in item_names:
        if name in found:
            loc, flags = found[name]
            print(f"  {name:<16} at {loc:<16} [{', '.join(flags)}]")
        else:
            print(f"  {name:<16} not found in containers")


def format_plan(plan):
    if plan is None:
        return '  <no plan>'
    lines = [f'  {i+1}. ({a.name} {" ".join(a.args)})' for i, a in enumerate(plan)]
    return '\n'.join(lines)


@pytest.fixture(scope='module')
def myopic_planner():
    return taskplan_multi.planners.myopic_planner.MyopicPlanner()


@pytest.fixture(scope='module')
def llm_planner():
    return taskplan_multi.planners.llm_planner.LLMPlanner(model=MODEL)


@pytest.mark.parametrize('robot,scenario_name,task', TASK_SCENARIOS, ids=[s[1] for s in TASK_SCENARIOS])
def test_llm_vs_myopic(myopic_planner, llm_planner, robot, scenario_name, task):
    print(f"\n\n{'='*68}")
    print(f"  Scenario : {scenario_name}  |  Robot: {robot}")
    print(f"{'='*68}")

    # Print initial state of key items before planning
    restaurant = make_restaurant(robot)
    print_item_states(restaurant, ['pasta', 'bowl1', 'bowl2'])

    # Run myopic
    restaurant = make_restaurant(robot)
    t0 = time.time()
    m_plan, m_cost = myopic_planner.get_cost_and_state_from_task(restaurant, task)
    m_elapsed = time.time() - t0

    # Run LLM on a fresh identical state
    restaurant = make_restaurant(robot)
    t0 = time.time()
    l_plan, l_cost = llm_planner.get_cost_and_state_from_task(restaurant, task)
    l_elapsed = time.time() - t0

    m_status = 'OK' if m_plan is not None else 'FAILED'
    l_status = 'OK' if l_plan is not None else 'FAILED'
    cost_delta = l_cost - m_cost

    print(f"\n  {'Metric':<18} {'Myopic':>18} {'LLM':>18}")
    print(f"  {'-'*56}")
    print(f"  {'Status':<18} {m_status:>18} {l_status:>18}")
    print(f"  {'Cost':<18} {m_cost:>18.1f} {l_cost:>18.1f}  (delta: {cost_delta:+.1f})")
    print(f"  {'Steps':<18} {len(m_plan) if m_plan else 0:>18} {len(l_plan) if l_plan else 0:>18}")
    print(f"  {'Time (s)':<18} {m_elapsed:>18.2f} {l_elapsed:>18.2f}")
    print(f"\n  --- Myopic plan ---\n{format_plan(m_plan)}")
    print(f"\n  --- LLM plan ---\n{format_plan(l_plan)}")

    assert m_plan is not None, f"Myopic planner unexpectedly failed on '{scenario_name}'"

    assert l_plan is not None, (
        f"LLM planner returned no plan for '{scenario_name}'. "
        f"Myopic cost was {m_cost:.1f}."
    )

    assert len(l_plan) > 0, "LLM plan is empty"

    known_actions = {'move', 'pick', 'place', 'wash', 'mix', 'serve', 'restock', 'ask-help'}
    unknown = {a.name for a in l_plan} - known_actions
    assert not unknown, f"LLM plan contains unknown actions: {unknown}"

    # Collect all object names declared in the PDDL problem
    import taskplan_multi.pddl.problem
    pddl_problem = taskplan_multi.pddl.problem.get_problem(make_restaurant(robot), task)
    # Extract declared objects from the :objects block
    import re
    objects_block = re.search(r'\(:objects(.*?)\)', pddl_problem, re.DOTALL)
    declared_objects = set(objects_block.group(1).split()) if objects_block else set()
    # Strip type annotations (words after ' - ')
    declared_objects = {w for w in declared_objects if w != '-' and not w.startswith('(')}

    hallucinated = set()
    for action in l_plan:
        for arg in action.args:
            if arg not in declared_objects:
                hallucinated.add(arg)
    assert not hallucinated, (
        f"LLM plan references objects not in the PDDL problem: {hallucinated}"
    )

    # Every robot that acts must have been activated via ask-help exactly once
    activated = set()
    for action in l_plan:
        if action.name == 'ask-help':
            robot = action.args[0]
            assert robot not in activated, (
                f"Robot '{robot}' has ask-help called more than once (it stays active after the first)"
            )
            activated.add(robot)
        elif action.name in ('move', 'pick', 'place', 'wash', 'mix', 'serve', 'restock'):
            acting_robot = action.args[0]
            assert acting_robot in activated, (
                f"Robot '{acting_robot}' performs ({action.name}) before being activated "
                f"via ask-help"
            )
