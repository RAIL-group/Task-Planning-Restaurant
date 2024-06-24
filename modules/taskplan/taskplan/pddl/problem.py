import taskplan
from taskplan.pddl.helper import generate_pddl_problem


def get_problem(restaurant, task):
    containers = restaurant.containers
    objects = {
         'init_r': ['initial_robot_pose']
    }
    init_states = [
        '(= (total-cost) 0)',
        '(restrict-move-to initial_robot_pose)',
        '(hand-is-free)',
        '(rob-at initial_robot_pose)',
        # '(= (find-cost mug1) 0)',
        # '(= (find-cost mug2) 0)',
        # '(= (find-cost mug3) 0)',
        '(is-fillable coffeemachine)'
    ]
    for container in containers:
        cnt_name = container['assetId']
        if cnt_name == 'coffeemachine':
            init_states.append("(is-fillable coffeemachine)")
            if 'filled' in container and container['filled'] == 1:
                init_states.append(f"(is-at water {cnt_name})")
        gen_name = ''.join([i for i in cnt_name if not i.isdigit()])
        if gen_name not in objects:
            objects[gen_name] = [cnt_name]
        else:
            objects[gen_name].append(cnt_name)
        children = container.get('children')
        if children is not None:
            for child in children:
                chld_name = child['assetId']
                gen_name_child = ''.join([i for i in chld_name if not i.isdigit()])
                if 'spread' in gen_name_child:
                    gen_name_child = 'spread'
                if gen_name_child not in objects:
                    objects[gen_name_child] = [chld_name]
                else:
                    objects[gen_name_child].append(chld_name)
                # objects[chld_name] = [chld_name]
                init_states.append(f"(is-at {chld_name} {cnt_name})")
                if 'missing' in child and child['missing'] == 1:
                    init_states.append(f"(not (is-located {chld_name}))")
                else:
                    init_states.append(f"(is-located {chld_name})")
                if 'isLiquid' in child and child['isLiquid'] == 1:
                    init_states.append(f"(is-liquid {chld_name})")
                if 'pickable' in child and child['pickable'] == 1:
                    init_states.append(f"(is-pickable {chld_name})")
                if 'spreadable' in child and child['spreadable'] == 1:
                    init_states.append(f"(is-spreadable {chld_name})")
                if 'washable' in child and child['washable'] == 1:
                    init_states.append(f"(is-washable {chld_name})")
                if 'dirty' in child and child['dirty'] == 1:
                    init_states.append(f"(is-dirty {chld_name})")
                if 'spread' in child and child['spread'] == 1:
                    init_states.append(f"(is-spread {chld_name})")
                if 'fillable' in child and child['fillable'] == 1:
                    init_states.append(f"(is-fillable {chld_name})")
    for c1 in restaurant.known_cost:
        for c2 in restaurant.known_cost[c1]:
            if c1 == c2:
                continue
            val = restaurant.known_cost[c1][c2]
            init_states.append(
                f"(= (known-cost {c1} {c2}) {val})"
            )
    # print(objects)
    # raise NotImplementedError
    # task = taskplan.pddl.task.serve_water('servingtable1')
    # task = taskplan.pddl.task.fill_coffeemachine_with_water()
    # task = taskplan.pddl.task.serve_coffee('servingtable1')
    # task = taskplan.pddl.task.clean_something('plate2')
    # task = taskplan.pddl.task.make_sandwich()
    # task = taskplan.pddl.task.make_sandwich('peanutbutterspread')
    # task = taskplan.pddl.task.serve_sandwich('servingtable2', 'orangespread')
    # task = taskplan.pddl.task.serve_sandwich('servingtable3')
    # task = taskplan.pddl.task.hold_something()
    # task = taskplan.pddl.task.clear_surface('shelf5')
    # task = taskplan.pddl.task.clean_everything()
    goal = [f'(and (hand-is-free) (rob-at initial_robot_pose) {task})']
    # goal = [task]
    PROBLEM_PDDL = generate_pddl_problem(
        domain_name='restaurant',
        problem_name='restaurant-problem',
        objects=objects,
        init_states=init_states,
        goal_states=goal
    )
    return PROBLEM_PDDL
