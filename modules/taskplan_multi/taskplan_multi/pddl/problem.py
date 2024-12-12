import taskplan
from taskplan.pddl.helper import generate_pddl_problem

COOK_BOT_REACHABLES = ['stove', 'fridge', 'countertop']
SERVER_BOT_REACHABLES = ['servingtable1', 'servingtable2', 'cabinet', 'countertop', 'bussingcart']
CLEANER_BOT_REACHABLES = ['dishwasher', 'bussingcart', 'countertop']

def get_problem(restaurant, task):
    containers = restaurant.containers
    objects = {}
    init_states = [
        '(= (total-cost) 0)',
    ]
    for agent in restaurant.agent_list:
        objects[agent] = [agent]
        base = 'base_' + agent
        if 'base' not in objects:
            objects['base'] = [base]
        else:
            objects['base'].append(base)
        init_states.append(f"(rob-at {agent} {restaurant.restaurant[agent]['rob_at']})")
        init_states.append(f"(hand-is-free {agent})")
        init_states.append(f"(restrict-place-to {base})")
        init_states.append(f"(can-reach {agent} {base})")
        if agent == 'cook_bot':
            for item in COOK_BOT_REACHABLES:
                init_states.append(f"(can-reach {agent} {item})")
        if agent == 'server_bot':
            for item in SERVER_BOT_REACHABLES:
                init_states.append(f"(can-reach {agent} {item})")
        if agent == 'cleaner_bot':
            for item in CLEANER_BOT_REACHABLES:
                init_states.append(f"(can-reach {agent} {item})")
    init_states.append(f"(robot-active {restaurant.active_robot})")
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
                init_states.append(f"(type {chld_name} {gen_name_child})")
                # if 'isLiquid' in child and child['isLiquid'] == 1:
                #     init_states.append(f"(is-liquid {chld_name})")
                # if 'pickable' in child and child['pickable'] == 1:
                #     init_states.append(f"(is-pickable {chld_name})")
                # if 'cookable' in child and child['cookable'] == 1:
                #     init_states.append(f"(is-cookable {chld_name})")
                # if 'washable' in child and child['washable'] == 1:
                #     init_states.append(f"(is-washable {chld_name})")
                if 'dirty' in child and child['dirty'] == 1:
                    init_states.append(f"(is-dirty {chld_name})")
                if 'cooked' in child and child['cooked'] == 1:
                    init_states.append(f"(is-cooked {chld_name})")
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
    # for state in init_states:
    #     print(state)
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
    base_loc = restaurant.restaurant[restaurant.active_robot]['rob_at']
    goal = [f'(and (hand-is-free {restaurant.active_robot}) (rob-at {restaurant.active_robot} {base_loc}) {task})']
    PROBLEM_PDDL = generate_pddl_problem(
        domain_name='restaurant',
        problem_name='restaurant-problem',
        objects=objects,
        init_states=init_states,
        goal_states=goal
    )
    return PROBLEM_PDDL
