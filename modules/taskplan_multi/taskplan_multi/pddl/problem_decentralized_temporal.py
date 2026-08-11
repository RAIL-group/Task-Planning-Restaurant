from taskplan.pddl.helper import generate_pddl_problem


def get_problem(restaurant, robot, task, reservations=None, arrival_time=0.0):
    """
    Temporal counterpart of problem_decentralized.get_problem: same
    single-agent-only object declarations. Reservations are still cleared
    by an explicit `wait-for` (the TFD build in use has no
    :timed-initial-literals support), but its duration is written in here
    as a precomputed constant via `release-wait` rather than being
    computed after the fact the way the classical planner did it.

    `reservations[item]['until']` is in the caller's *absolute* simulation
    clock; this problem's own internal clock starts at 0 at `arrival_time`,
    so `release-wait` is `until - arrival_time`.
    """
    reservations = reservations or {}
    containers = restaurant.containers
    objects = {}
    init_states = [
        '(= (total-cost) 0)',
    ]

    base = 'base_' + robot
    objects['robot'] = [robot]
    objects['base'] = [base]
    declared_locations = {base}

    init_states.append(f"(rob-at {robot} {restaurant.restaurant[robot]['rob_at']})")
    init_states.append(f"(hand-is-free {robot})")
    init_states.append(f"(type {robot} {robot})")

    for container in containers:
        cnt_name = container['assetId']
        gen_name = ''.join([i for i in cnt_name if not i.isdigit()])
        if gen_name not in objects:
            objects[gen_name] = [cnt_name]
        else:
            objects[gen_name].append(cnt_name)
        declared_locations.add(cnt_name)

        children = container.get('children')
        if children is not None:
            for child in children:
                chld_name = child['assetId']
                gen_name_child = ''.join([i for i in chld_name if not i.isdigit()])
                if gen_name_child not in objects:
                    objects[gen_name_child] = [chld_name]
                else:
                    objects[gen_name_child].append(chld_name)
                init_states.append(f"(is-at {chld_name} {cnt_name})")
                init_states.append(f"(type {chld_name} {gen_name_child})")
                if child.get('dirty') == 1:
                    init_states.append(f"(is-dirty {chld_name})")
                if child.get('empty') == 1:
                    init_states.append(f"(is-empty {chld_name})")
                if chld_name in reservations:
                    release_wait = max(0.0, reservations[chld_name]['until'] - arrival_time)
                    init_states.append(f"(reserved {chld_name})")
                    init_states.append(f"(= (release-wait {chld_name}) {release_wait})")

    for c1, dests in restaurant.known_cost.items():
        if c1 not in declared_locations:
            continue
        for c2, val in dests.items():
            if c1 == c2 or c2 not in declared_locations:
                continue
            init_states.append(f"(= (known-cost {c1} {c2}) {val})")

    goal = [f'(and (hand-is-free {robot}) {task})']

    PROBLEM_PDDL = generate_pddl_problem(
        domain_name='restaurant_decentralized_temporal',
        problem_name='restaurant-decentralized-temporal-problem',
        objects=objects,
        init_states=init_states,
        goal_states=goal,
    )
    return PROBLEM_PDDL
