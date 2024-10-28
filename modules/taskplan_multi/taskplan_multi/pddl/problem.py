import taskplan
from taskplan.pddl.helper import generate_pddl_problem


def get_problem(restaurant, task):
    containers = restaurant.containers
    objects = {
         'init_tall': ['init_tall'],
         'init_tiny': ['init_tiny'],
         'agent_tall': ['agent_tall'],
         'agent_tiny': ['agent_tiny'],
    }
    init_states = [
        '(= (total-cost) 0)',
        '(hand-is-free agent_tall)',
        '(hand-is-free agent_tiny)',
        '(can-reach agent_tall countertop)',
        '(can-reach agent_tall cabinet)',
        '(can-reach agent_tall servingtable1)',
        '(can-reach agent_tiny countertop)',
        '(can-reach agent_tiny dishwasher)',
    ]
    init_states.append(f"(rob-at agent_tall {restaurant.tall_agent_at})")
    init_states.append(f"(rob-at agent_tiny {restaurant.tiny_agent_at})")
    for container in containers:
        cnt_name = container['assetId']
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
                if gen_name_child not in objects:
                    objects[gen_name_child] = [chld_name]
                else:
                    objects[gen_name_child].append(chld_name)
                init_states.append(f"(is-at {chld_name} {cnt_name})")
                # if 'isLiquid' in child and child['isLiquid'] == 1:
                #     init_states.append(f"(is-liquid {chld_name})")
                # if 'pickable' in child and child['pickable'] == 1:
                #     init_states.append(f"(is-pickable {chld_name})")
                # if 'spreadable' in child and child['spreadable'] == 1:
                #     init_states.append(f"(is-spreadable {chld_name})")
                # if 'washable' in child and child['washable'] == 1:
                #     init_states.append(f"(is-washable {chld_name})")
                # if 'dirty' in child and child['dirty'] == 1:
                #     init_states.append(f"(is-dirty {chld_name})")
                # if 'spread' in child and child['spread'] == 1:
                #     init_states.append(f"(is-spread {chld_name})")
                # if 'fillable' in child and child['fillable'] == 1:
                #     init_states.append(f"(is-fillable {chld_name})")
                # if 'filled' in child and child['filled'] == 1:
                #     init_states.append(f"(filled-with water {chld_name})")
                # if 'jar' in child and child['jar'] == 1:
                #     init_states.append(f"(is-jar {chld_name})")
                # if 'slicable' in child and child['slicable'] == 1:
                #     init_states.append(f"(is-slicable {chld_name})")
                # if 'container' in child and child['container'] == 1:
                #     init_states.append(f"(is-container {chld_name})")
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
    # goal = [f'(and (hand-is-free) {task})']
    goal = [task]
    PROBLEM_PDDL = generate_pddl_problem(
        domain_name='restaurant',
        problem_name='restaurant-problem',
        objects=objects,
        init_states=init_states,
        goal_states=goal
    )
    return PROBLEM_PDDL
