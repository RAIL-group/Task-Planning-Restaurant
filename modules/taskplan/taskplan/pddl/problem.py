def generate_pddl_problem(domain_name, problem_name, objects, init_states,
                          goal_states):
    """
    Generates a PDDL problem file content.

    :param domain_name: Name of the PDDL domain.
    :param problem_name: Name of the PDDL problem.
    :param objects: Dictionary of objects, where keys are types and values are lists of object names.
    :param init_states: List of strings representing the initial states.
    :param goal_states: List of strings representing the goal states.
    :return: A string representing the content of a PDDL problem file.
    """
    # Start the problem definition
    problem_str = f"(define (problem {problem_name})\n"
    problem_str += f"    (:domain {domain_name})\n"

    # Define objects
    problem_str += "    (:objects\n"
    for obj_type, obj_names in objects.items():
        problem_str += "        " + " ".join(obj_names) + " - " + obj_type + "\n"
    problem_str += "    )\n"

    # Define initial state
    problem_str += "    (:init\n"
    for state in init_states:
        problem_str += "        " + state + "\n"
    problem_str += "    )\n"

    # Define goal state
    problem_str += "    (:goal\n"
    problem_str += "        (and\n"
    for state in goal_states:
        problem_str += "            " + state + "\n"
    problem_str += "        )\n"
    problem_str += "    )\n"

    problem_str += "    (:metric minimize (total-cost))\n"

    # Close the problem definition
    problem_str += ")\n"

    return problem_str


def get_problem(restaurant):
    containers = restaurant.containers
    objects = {
         'init_r': ['initial_robot_pose']
    }
    init_states = [
        '(= (total-cost) 0)',
        '(restrict-move-to initial_robot_pose)',
        '(hand-is-free)',
        '(rob-at initial_robot_pose)',
        '(= (find-cost mug1) 0)',
        '(= (find-cost mug2) 0)',
        '(= (find-cost mug3) 0)',
        '(is-fillable coffeemachine)'
    ]
    for container in containers:
        cnt_name = container['assetId']
        objects[cnt_name] = [cnt_name]
        children = container.get('children')
        if children is not None:
            for child in children:
                chld_name = child['assetId']
                objects[chld_name] = [chld_name]
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

    print(objects)
    # print(init_states)

    for c1 in restaurant.known_cost:
        for c2 in restaurant.known_cost[c1]:
            val = restaurant.known_cost[c1][c2]
            init_states.append(
                f"(= (known-cost {c1} {c2}) {val})"
            )
            # init_states.append(
            #     f"(= (known-cost {c2} {c1}) {val})"
            # )
            print(f"(= (known-cost {c1} {c2}) {val})")

    # raise NotImplementedError
    # manually initialized required objects
    # objects = {
    #     'init_r': ['initial_robot_pose'],
    #     'water': ['water'],
    #     'wine': ['wine'],
    #     'beer': ['beer'],
    #     'whiskey': ['whiskey'],
    #     'fountain': ['fountain'],
    #     'bar': ['bar'],
    #     'countertop': ['countertop'],
    #     'coffeegrinds': ['coffeegrinds'],
    #     'coffeemachine': ['coffeemachine'],
    #     'servingtable0': ['servingtable0'],
    #     'servingtable1': ['servingtable1'],
    #     'shelf0': ['shelf0'],
    #     'shelf1': ['shelf1'],
    #     'shelf2': ['shelf2'],
    #     'cup': ['cup'],
    #     'mug': ['mug'],
    #     'bread': ['bread'],
    #     'knife': ['knife'],
    #     'strawberry': ['strawberry'],
    #     'orange': ['orange'],
    #     'peanutbutter': ['peanutbutter'],

    # }
    # # add objects from the partial map
    # for idx in partial_map.cnt_node_idx+partial_map.obj_node_idx:
    #     obj = partial_map.org_node_names[idx]
    #     objects[obj] = [obj]

    # init_states = [
    #     "(= (total-cost) 0)",
    #     # set the locations of known space objects
    #     "(is-located water)", "(is-at water fountain)",
    #     "(is-located coffeegrinds)", "(is-at coffeegrinds shelf2)",
    #     "(is-located wine)", "(is-at wine bar)",
    #     "(is-located beer)", "(is-at beer bar)",
    #     "(is-located whiskey)", "(is-at whiskey bar)",
    #     "(is-located bread)", "(is-at bread shelf2)",
    #     "(is-located knife)", "(is-at knife shelf1)",
    #     "(is-located orange)", "(is-at orange shelf0)",
    #     "(is-located strawberry)", "(is-at strawberry shelf0)",
    #     "(is-located peanutbutter)", "(is-at peanutbutter shelf0)",
    #     "(= (find-cost mug) 0)",
    #     "(= (find-cost cup) 0)",
    #     '(hand-is-free)', '(rob-at initial_robot_pose)',
    #     '(is-liquid water)', '(is-liquid coffee)',
    #     '(is-liquid wine)', '(is-liquid beer)', '(is-liquid whiskey)',
    #     '(is-spreadable bread)',
    #     '(is-spread orange)', '(is-spread strawberry)', '(is-spread peanutbutter)',
    #     '(is-pickable coffeegrinds)', '(is-pickable cup)', '(is-pickable mug)',
    #     '(is-pickable bread)', '(is-pickable knife)',
    #     '(is-pickable orange)', '(is-pickable strawberry)', '(is-pickable peanutbutter)',
    #     '(is-fillable cup)', '(is-fillable mug)', '(is-fillable coffeemachine)',
    #     '(restrict-move-to initial_robot_pose)'
    # ]
    # # get distance cost from robot to all containers
    # r_to_c_dist = get_robot_to_contianer_distances(partial_map, grid, robot_pose)
    # # get distance cost from every container to other containers
    # c_to_c_dist = get_contianer_to_contianer_distances(partial_map, grid)
    # init_states = init_states + r_to_c_dist + c_to_c_dist

    # if args.task == ['water']
    # if args.task == ['coffee']
    # if args.task == ['alcohol']
    # if args.task == ['sandwich']
    # if args.task == ['alcohol' **args='wine/beer/whiskey']
    # if args.task == ['sandwich' **args='orange/strawberry/peanutbutter']
    goal = [
        # '(is-at cup1 stable1)'
        # '(and (filled-with water cup1) (is-at cup1 stable1))'
        '(exists',
        '    (?c - item)',
        '        (and (filled-with coffee ?c))'
        ')'
        # '(exists',
        # '    (?c - item)',
        # '        (and (is-at ?c stable1) (filled-with coffee ?c))'
        # ')'
        # '(exists',
        # '    (?c - item)',
        # '        (and',
        # '            (is-at ?c servingtable1)',
        # '            (exists',
        # '                (?alc - alcohol)',
        # '                (and',
        # '                    (filled-with ?alc ?c))',
        # '            )',
        # '        )',
        # ')'
    ]

    PROBLEM_PDDL = generate_pddl_problem(
        domain_name='restaurant',
        problem_name='water-problem',
        objects=objects,
        init_states=init_states,
        goal_states=goal
    )
    return PROBLEM_PDDL
