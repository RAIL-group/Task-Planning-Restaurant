def test_antplan_viz_task_plan():
    current_seed = 5
    save_file = '/data/figures/plan-on-grid-ap-task-plan' \
        + str(current_seed) + '.png'
    random.seed(current_seed)
    proc_data = antplan.utilities.procthor_helper.ProcTHOR_HOUSE(
        seed=current_seed, preprocess=True)
    non_prepared_state = copy.deepcopy(proc_data.randomize_objects())
    proc_data.update_container_props(non_prepared_state)
    environment = antplan.pddl_problem.procthor_world.Environment(verbose=True)
    # task_list = antplan.utilities.utils.get_tasks(proc_data, current_seed)
    # assert len(task_list) > 0
    grid = np.transpose(proc_data.occupancy_grid)
    img = make_plotting_grid(grid)
    # img = np.transpose(grid)
    plt.imshow(img, cmap='gray_r', alpha=0.5)
    # print(task_list)
    # raise NotImplementedError
    task = [('egg', 'fridge')]
    plan, final_state, cost = MyopicPlanner(
        environment).get_cost_and_state_from_task(
            proc_data, task)
    proc_data.update_container_props(final_state)
    move_plans = [p for p in plan if p.name == "move"]
    pick_plans = [p for p in plan if p.name == "pick"]
    place_plans = [p for p in plan if p.name == "place"]
    move_poses = list()
    offset = 0.2
    # locs = proc_data.get_container_pos_list()
    robot = proc_data.agent['position']
    # plt.plot(robot[0], robot[1], marker='x')
    plt.text(robot[0]+offset, robot[1]+offset, 'robot', fontsize=6, rotation=45)
    # for (key, value) in locs:
    #     # plt.plot(value[0], value[1], marker='.')
    #     offset = 0.2
    #     plt.text(value[0]+offset, value[1]+offset, key, fontsize=6,
    #              color='brown', rotation=45)
    #     chldrn = proc_data.get_container_children(key)
    #     for chld in chldrn:
    #         nm = chld['uoid']
    #         plt.text(value[0]+offset, value[1]+offset, nm, fontsize=4,
    #                  color='pink', rotation=45)
    #         offset += 0.2
    for move in move_plans:
        if move.args[1] == 'base':
            pos1 = proc_data.agent['position']
        else:
            pos1 = proc_data.get_container_pos_by_name(move.args[1])
        if move.args[2] == 'base':
            pos2 = proc_data.agent['position']
        else:
            pos2 = proc_data.get_container_pos_by_name(move.args[2])
        # plt.text(pos1[0]-1, pos1[1]-1, f'{move.args[1]}', color='brown',
        #          fontsize=6, rotation=45)
        # plt.text(pos2[0]-1, pos2[1]-1, f'{move.args[2]}', color='brown',
        #          fontsize=6, rotation=45)
        move_poses.append((pos1, pos2))
    paths = list()
    for pos in move_poses:
        src, targate = (pos)
        cost, path = proc_data.get_cost_from_occupancy_grid(
            src[0], src[1], targate[0], targate[1], return_path=True)
        path_points = [(path[0][idx], path[1][idx])
                       for idx in range(len(path[0]))]
        path = geometry.LineString(path_points)
        x, y = path.xy
        plt.plot(x, y, color='blue')
        path = path.buffer(15)
        paths.append(path)

    merged_polygon = None
    for poly in paths:
        if merged_polygon is None:
            merged_polygon = poly
        else:
            merged_polygon = unary_union([merged_polygon, poly])

    x, y = merged_polygon.exterior.xy
    # plt.plot(x, y, color='blue')
    plt.fill(x, y, color='lightblue', alpha=0.5)
    # plt.plot(path[0], path[1], linestyle='--', color='orange', linewidth=1)

    # for pick in pick_plans:
    #     name = pick.args[1]
    #     pos = proc_data.get_container_pos_by_name(pick.args[2])
    #     plt.text(pos[0]+1, pos[1]+1, f'Pick: {name}', color='indigo',
    #              fontsize=4, rotation=45)

    # for place in place_plans:
    #     name = place.args[1]
    #     pos = proc_data.get_container_pos_by_name(place.args[2])
    #     plt.text(pos[0]+2, pos[1]+2, f'Place: {name}', color='fuchsia',
    #              fontsize=4, rotation=45)
    # plt.title(f'Planning Cost: {cost}')
    plt.axis('off')  # Turn off axis
    plt.savefig(save_file, dpi=1200)
    assert True