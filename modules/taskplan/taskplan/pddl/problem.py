from taskplan.pddl.helper import generate_pddl_problem


def get_problem():
    # manually initialized required objects
    objects = {
        'init_r': ['initial_robot_pose'],
        'water': ['water'],
        'wine': ['wine'],
        'beer': ['beer'],
        'whiskey': ['whiskey'],
        'fountain': ['fountain'],
        'bar': ['bar'],
        'countertop': ['countertop'],
        'coffeegrinds': ['coffeegrinds'],
        'coffeemachine': ['coffeemachine'],
        'servingtable0': ['servingtable0'],
        'servingtable1': ['servingtable1'],
        'shelf0': ['shelf0'],
        'shelf1': ['shelf1'],
        'shelf2': ['shelf2'],
        'cup': ['cup'],
        'mug': ['mug'],
        'bread': ['bread'],
        'knife': ['knife'],
        'strawberry': ['strawberry'],
        'orange': ['orange'],
        'peanutbutter': ['peanutbutter'],

    }
    # # add objects from the partial map
    # for idx in partial_map.cnt_node_idx+partial_map.obj_node_idx:
    #     obj = partial_map.org_node_names[idx]
    #     objects[obj] = [obj]

    init_states = [
        "(= (total-cost) 0)",
        # set the locations of known space objects
        "(is-located water)", "(is-at water fountain)",
        "(is-located coffeegrinds)", "(is-at coffeegrinds shelf2)",
        "(is-located wine)", "(is-at wine bar)",
        "(is-located beer)", "(is-at beer bar)",
        "(is-located whiskey)", "(is-at whiskey bar)",
        "(is-located bread)", "(is-at bread shelf2)",
        "(is-located knife)", "(is-at knife shelf1)",
        "(is-located orange)", "(is-at orange shelf0)",
        "(is-located strawberry)", "(is-at strawberry shelf0)",
        "(is-located peanutbutter)", "(is-at peanutbutter shelf0)",
        "(= (find-cost mug) 0)",
        "(= (find-cost cup) 0)",
        '(hand-is-free)', '(rob-at initial_robot_pose)',
        '(is-liquid water)', '(is-liquid coffee)',
        '(is-liquid wine)', '(is-liquid beer)', '(is-liquid whiskey)',
        '(is-spreadable bread)',
        '(is-spread orange)', '(is-spread strawberry)', '(is-spread peanutbutter)',
        '(is-pickable coffeegrinds)', '(is-pickable cup)', '(is-pickable mug)',
        '(is-pickable bread)', '(is-pickable knife)',
        '(is-pickable orange)', '(is-pickable strawberry)', '(is-pickable peanutbutter)',
        '(is-fillable cup)', '(is-fillable mug)', '(is-fillable coffeemachine)',
        '(restrict-move-to initial_robot_pose)'
    ]
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
        # '(and (is-at bread servingtable1) (spread-applied bread strawberry))'
        '(exists',
        '    (?c - item)',
        '        (and (is-at ?c servingtable0) (filled-with coffee ?c))'
        ')'
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
