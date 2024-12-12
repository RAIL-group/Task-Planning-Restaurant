import taskplan_multi
import random

def tasks_for_cook(items):
    tasks = list()
    for item in items:
        tasks.append(
            {
                f'cook_{item}': taskplan_multi.pddl.task.cook_something(item),
            }
        )
        tasks.append(
            {
                f'preserve_{item}': taskplan_multi.pddl.task.place_something(item, 'fridge'),
            }
        )
    tasks.append(
        {
            f'make_pasta': taskplan_multi.pddl.task.make_pasta(),
        }
    )
    tasks.append(
        {
            f'make_omelette': taskplan_multi.pddl.task.make_omelette(),
        }
    )
    tasks.append(
        {
            f'make_oats': taskplan_multi.pddl.task.make_oats(),
        }
    )
    return tasks


def tasks_for_server(items):
    tasks = list()
    for item in items:
        tasks.append(
            {
                f'organize_{item}': taskplan_multi.pddl.task.organize_something(item),
            }
        )
    tasks.append(
        {
            f'serve_milk': taskplan_multi.pddl.task.serve_milk('servingtable1'),
        }
    )
    tasks.append(
        {
            f'serve_oats': taskplan_multi.pddl.task.serve_oats('servingtable1'),
        }
    )
    tasks.append(
        {
            f'serve_pasta': taskplan_multi.pddl.task.serve_pasta('servingtable1'),
        }
    )
    tasks.append(
        {
            f'serve_omelette': taskplan_multi.pddl.task.serve_omelette('servingtable1'),
        }
    )
    tasks.append(
        {
            f'serve_milk': taskplan_multi.pddl.task.serve_milk('servingtable2'),
        }
    )
    tasks.append(
        {
            f'serve_oats': taskplan_multi.pddl.task.serve_oats('servingtable2'),
        }
    )
    tasks.append(
        {
            f'serve_pasta': taskplan_multi.pddl.task.serve_pasta('servingtable2'),
        }
    )
    tasks.append(
        {
            f'serve_omelette': taskplan_multi.pddl.task.serve_omelette('servingtable2'),
        }
    )
    # tasks.append(
    #     {
    #         f'clear_table': taskplan_multi.pddl.task.clear_surface('servingtable1'),
    #     }
    # )
    # tasks.append(
    #     {
    #         f'clear_table': taskplan_multi.pddl.task.clear_surface('servingtable2'),
    #     }
    # )
    return tasks

def tasks_for_cleaner(items, food_items):
    tasks = list()
    for item in items:
        tasks.append(
            {
                f'clean_{item}': taskplan_multi.pddl.task.clean_something(item),
            }
        )
        tasks.append(
            {
                f'clear_bussingcart': taskplan_multi.pddl.task.clear_surface(item, 'bussingcart'),
            }
        )
    for item in food_items:
        tasks.append(
            {
                f'clear_bussingcart': taskplan_multi.pddl.task.clear_surface(item, 'bussingcart'),
            }
        )
    return tasks


def available_tasks_for_cleaner(dirty, on_items):
    tasks = list()
    for item in dirty:
        tasks.append(
            {
                f'clean_{item}': taskplan_multi.pddl.task.clean_something(item),
            }
        )
    for item in on_items:
        tasks.append(
            {
                f'clear_bussingcart': taskplan_multi.pddl.task.clear_surface(item, 'bussingcart'),
            }
        )
    return tasks
