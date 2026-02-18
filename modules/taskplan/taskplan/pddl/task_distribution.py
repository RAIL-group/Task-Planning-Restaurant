import taskplan


def service_tasks():
    return [
        {
            'serve_water': taskplan.pddl.task.serve_water('servingtable1', 'cup1'),
        },
        {
            'serve_water': taskplan.pddl.task.serve_water('servingtable2', 'cup1'),
        },
        {
            'serve_water': taskplan.pddl.task.serve_water('servingtable3', 'cup1'),
        },
        {
            'serve_water': taskplan.pddl.task.serve_water('servingtable1', 'cup2'),
        },
        {
            'serve_water': taskplan.pddl.task.serve_water('servingtable2', 'cup2'),
        },
        {
            'serve_water': taskplan.pddl.task.serve_water('servingtable3', 'cup2'),
        },
        {
            'serve_coffee': taskplan.pddl.task.serve_coffee('servingtable1', 'mug1'),
        },
        {
            'serve_coffee': taskplan.pddl.task.serve_coffee('servingtable2', 'mug1'),
        },
        {
            'serve_coffee': taskplan.pddl.task.serve_coffee('servingtable3', 'mug1'),
        },
        {
            'serve_coffee': taskplan.pddl.task.serve_coffee('servingtable1', 'mug2'),
        },
        {
            'serve_coffee': taskplan.pddl.task.serve_coffee('servingtable2', 'mug2'),
        },
        {
            'serve_coffee': taskplan.pddl.task.serve_coffee('servingtable3', 'mug2'),
        },
    ]


def serve_solid():
    return [
        {
            'serve_sandwich': taskplan.pddl.task.serve_sandwich('servingtable1'),
        },
        {
            'serve_sandwich': taskplan.pddl.task.serve_sandwich('servingtable2'),
        },
        {
            'serve_sandwich': taskplan.pddl.task.serve_sandwich('servingtable3'),
        },
        {
            'serve_fruit_bowl': taskplan.pddl.task.serve_fruit('bowl1', 'servingtable1'),
        },
        {
            'serve_fruit_bowl': taskplan.pddl.task.serve_fruit('bowl1', 'servingtable2'),
        },
        {
            'serve_fruit_bowl': taskplan.pddl.task.serve_fruit('bowl1', 'servingtable3'),
        },
    ]


def cleaning_task():
    return [
        {
            'clean_mug': taskplan.pddl.task.clean_something('mug1'),
        },
        {
            'clean_mug': taskplan.pddl.task.clean_something('mug2'),
        },
        {
            'clean_bowl': taskplan.pddl.task.clean_something('bowl1'),
        },
        {
            'clean_cup': taskplan.pddl.task.clean_something('cup1'),
        },
        {
            'clean_cup': taskplan.pddl.task.clean_something('cup2'),
        },
        {
            'clean_knife': taskplan.pddl.task.clean_something('knife1'),
        },
    ]


def clear_task():
    return [
        {
            'clear_the_table': taskplan.pddl.task.clear_surface('servingtable1'),
        },
        {
            'clear_the_table': taskplan.pddl.task.clear_surface('servingtable2'),
        },
        {
            'clear_the_table': taskplan.pddl.task.clear_surface('servingtable3'),
        },
        {
            'clear_the_countertop': taskplan.pddl.task.clear_surface('countertop'),
        },
        {
            'clear_the_coffeemachine': taskplan.pddl.task.clear_surface('coffeemachine'),
        },
        {
            'clear_the_dishwasher': taskplan.pddl.task.clear_surface(
                'dishwasher'),
        },
    ]


def organizing_task():
    return [
        {
            'place_spread_on_spreadshelf': taskplan.pddl.task.place_something(
                'orangespread', 'spreadshelf'),
        },
        {
            'place_knife_on_cutleryshelf':
            taskplan.pddl.task.place_something('knife1', 'cutleryshelf'),
        },
        {
            'place_cup_on_cupshelf':
            taskplan.pddl.task.place_something('cup1', 'cupshelf'),
        },
        {
            'place_cup_on_cupshelf':
            taskplan.pddl.task.place_something('cup2', 'cupshelf'),
        },
        {
            'place_mug_on_cupshelf':
            taskplan.pddl.task.place_something('mug1', 'cupshelf'),
        },
        {
            'place_mug_on_cupshelf':
            taskplan.pddl.task.place_something('mug2', 'cupshelf'),
        },
        {
            'place_bowl_on_cutleryshelf':
            taskplan.pddl.task.place_something('bowl1', 'cutleryshelf'),
        },
        {
            'place_bowl_on_dishshelf':
            taskplan.pddl.task.place_something('bowl1', 'dishshelf'),
        },
        {
            'place_jar_on_cupshelf':
            taskplan.pddl.task.place_something('jar1', 'cupshelf'),
        },
        {
            'place_jar_on_cupshelf':
            taskplan.pddl.task.place_something('jar2', 'cupshelf'),
        },
        {
            'place_bread_on_breadshelf':
            taskplan.pddl.task.place_something('bread', 'breadshelf'),
        },
    ]
