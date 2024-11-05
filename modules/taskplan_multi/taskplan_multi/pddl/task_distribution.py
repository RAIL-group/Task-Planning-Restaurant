import taskplan_multi


def tall_robots_tasks(items):
    tasks = list()
    for item in items:
        tasks.append(
            {
                f'clean_{item}_to_cabinet': taskplan_multi.pddl.task.place_something(item, 'cabinet'),
            }
        )
        tasks.append(
            {
                f'clean_{item}_to_servingtable1': taskplan_multi.pddl.task.place_something(item, 'servingtable1'),
            }
        )
    return tasks


def tiny_robots_tasks(items):
    tasks = list()
    for item in items:
        tasks.append(
            {
                f'dirty_{item}_to_dishwasher': taskplan_multi.pddl.task.place_something(item, 'dishwasher'),
            }
        )
    return tasks