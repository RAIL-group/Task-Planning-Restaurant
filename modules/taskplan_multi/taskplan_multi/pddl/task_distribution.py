import taskplan_multi
import random

def tasks_for_cook():
    tasks = list()
    
    # Bring Items at Stove to Cook
    # tasks.append(
    #     ('bring_clean_pan_at_stove', taskplan_multi.pddl.task.bring_clean_item('pan', 'stove'), 0.5)
    # )
    # tasks.append(
    #     ('bring_clean_bowl_at_stove', taskplan_multi.pddl.task.bring_clean_item('bowl', 'stove'), 0.5)
    # )
    # tasks.append(
    #     ('bring_clean_mug_at_stove', taskplan_multi.pddl.task.bring_clean_item('mug', 'stove'), 0.5)
    # )
    # tasks.append(
    #     ('bring_sauce_stove', taskplan_multi.pddl.task.place_something('sauce', 'stove'), 0.5)
    # )
    # tasks.append(
    #     ('bring_saltshaker_stove', taskplan_multi.pddl.task.place_something('saltshaker', 'stove'), 0.5)
    # )
    # tasks.append(
    #     ('bring_bowl_and_sauce_stove', taskplan_multi.pddl.task.bring_two_items('sauce', 'bowl', 'stove'), 0.5)
    # )
    # tasks.append(
    #     ('bring_bowl_and_saltshaker_stove', taskplan_multi.pddl.task.bring_two_items('saltshaker', 'bowl', 'stove'), 0.5)
    # )
    # tasks.append(
    #     ('bring_pan_and_sauce_stove', taskplan_multi.pddl.task.bring_two_items('sauce', 'pan', 'stove'), 0.5)
    # )
    # tasks.append(
    #     ('bring_pan_and_saltshaker_stove', taskplan_multi.pddl.task.bring_two_items('saltshaker', 'pan', 'stove'), 0.5)
    # )
    
    # Mix Pasta at Stove and Countertop
    tasks.append(
        ('pasta_in_pan_at_stove', taskplan_multi.pddl.task.prep_item('pasta', 'pan', 'stove'), 0.5)
    )
    tasks.append(
        ('pasta_in_bowl_at_stove', taskplan_multi.pddl.task.prep_item('pasta', 'bowl', 'stove'), 0.5)
    )
    tasks.append(
        ('pasta_in_pan_at_countertop', taskplan_multi.pddl.task.prep_item('pasta', 'pan', 'countertop'), 0.5)
    )
    tasks.append(
        ('pasta_in_bowl_at_countertop', taskplan_multi.pddl.task.prep_item('pasta', 'bowl', 'countertop'), 0.5)
    )

    # Mix Oats at Stove and Countertop
    tasks.append(
        ('oats_in_pan_at_stove', taskplan_multi.pddl.task.prep_item('oats', 'pan', 'stove'), 0.5)
    )
    tasks.append(
        ('oats_in_bowl_at_stove', taskplan_multi.pddl.task.prep_item('oats', 'bowl', 'stove'), 0.5)
    )
    tasks.append(
        ('oats_in_pan_at_countertop', taskplan_multi.pddl.task.prep_item('oats', 'pan', 'countertop'), 0.5)
    )
    tasks.append(
        ('oats_in_bowl_at_countertop', taskplan_multi.pddl.task.prep_item('oats', 'bowl', 'countertop'), 0.5)
    )

    #Cereal
    tasks.append(
        ('cereal_in_pan_at_stove', taskplan_multi.pddl.task.prep_item('cereal', 'pan', 'stove'), 0.5)
    )
    tasks.append(
        ('cereal_in_bowl_at_stove', taskplan_multi.pddl.task.prep_item('cereal', 'bowl', 'stove'), 0.5)
    )
    tasks.append(
        ('cereal_in_mug_at_stove', taskplan_multi.pddl.task.prep_item('cereal', 'mug', 'stove'), 0.5)
    )
    tasks.append(
        ('cereal_in_pan_at_countertop', taskplan_multi.pddl.task.prep_item('cereal', 'pan', 'countertop'), 0.5)
    )
    tasks.append(
        ('cereal_in_bowl_at_countertop', taskplan_multi.pddl.task.prep_item('cereal', 'bowl', 'countertop'), 0.5)
    )
    tasks.append(
        ('cereal_in_mug_at_countertop', taskplan_multi.pddl.task.prep_item('cereal', 'mug', 'countertop'), 0.5)
    )

    # Milk
    tasks.append(
        ('milk_in_pan_at_stove', taskplan_multi.pddl.task.prep_item('milk', 'pan', 'stove'), 0.5)
    )
    tasks.append(
        ('milk_in_bowl_at_stove', taskplan_multi.pddl.task.prep_item('milk', 'bowl', 'stove'), 0.5)
    )
    tasks.append(
        ('milk_in_mug_at_stove', taskplan_multi.pddl.task.prep_item('milk', 'mug', 'stove'), 0.5)
    )
    tasks.append(
        ('milk_in_bowl_at_countertop', taskplan_multi.pddl.task.prep_item('milk', 'bowl', 'countertop'), 0.5)
    )
    tasks.append(
        ('milk_in_mug_at_countertop', taskplan_multi.pddl.task.prep_item('milk', 'mug', 'countertop'), 0.5)
    )

    return tasks


def tasks_for_server():
    tasks = list()
    
    # Bring Items
    # tasks.append(
    #     ('bring_clean_bowl_servingtable1', taskplan_multi.pddl.task.bring_clean_item('bowl', 'servingtable1'), 0.5)
    # )
    # tasks.append(
    #     ('bring_clean_mug_servingtable1', taskplan_multi.pddl.task.bring_clean_item('mug', 'servingtable1'), 0.5)
    # )
    # tasks.append(
    #     ('bring_sauce_servingtable1', taskplan_multi.pddl.task.place_something('sauce', 'servingtable1'), 0.5)
    # )
    # tasks.append(
    #     ('bring_saltshaker_servingtable1', taskplan_multi.pddl.task.place_something('saltshaker', 'servingtable1'), 0.5)
    # )
    # tasks.append(
    #     ('bring_clean_bowl_servingtable2', taskplan_multi.pddl.task.bring_clean_item('bowl', 'servingtable2'), 0.5)
    # )
    # tasks.append(
    #     ('bring_clean_mug_servingtable2', taskplan_multi.pddl.task.bring_clean_item('mug', 'servingtable2'), 0.5)
    # )
    # tasks.append(
    #     ('bring_sauce_servingtable2', taskplan_multi.pddl.task.place_something('sauce', 'servingtable2'), 0.5)
    # )
    # tasks.append(
    #     ('bring_saltshaker_servingtable2', taskplan_multi.pddl.task.place_something('saltshaker', 'servingtable2'), 0.5)
    # )
    # tasks.append(
    #     ('bring_bowl_and_sauce_servingtable1', taskplan_multi.pddl.task.bring_two_items('sauce', 'bowl', 'servingtable1'), 0.5)
    # )
    # tasks.append(
    #     ('bring_bowl_and_saltshaker_servingtable1', taskplan_multi.pddl.task.bring_two_items('saltshaker', 'bowl', 'servingtable1'), 0.5)
    # )
    # tasks.append(
    #     ('bring_bowl_and_sauce_servingtable2', taskplan_multi.pddl.task.bring_two_items('sauce', 'bowl', 'servingtable2'), 0.5)
    # )
    # tasks.append(
    #     ('bring_bowl_and_saltshaker_servingtable2', taskplan_multi.pddl.task.bring_two_items('saltshaker', 'bowl', 'servingtable2'), 0.5)
    # )

    # Serve Pasta
    tasks.append(
        ('serve_pasta_in_bowl_at_servingtable1', taskplan_multi.pddl.task.serve_item('pasta', 'bowl', 'servingtable1'), 0.5)
    )
    tasks.append(
        ('serve_pasta_in_bowl_at_servingtable2', taskplan_multi.pddl.task.serve_item('pasta', 'bowl', 'servingtable2'), 0.5)
    )

    # Serve Oats
    tasks.append(
        ('serve_oats_in_bowl_at_servingtable1', taskplan_multi.pddl.task.serve_item('oats', 'bowl', 'servingtable1'), 0.5)
    )
    tasks.append(
        ('serve_oats_in_bowl_at_servingtable2', taskplan_multi.pddl.task.serve_item('oats', 'bowl', 'servingtable2'), 0.5)
    )

    # Serve Oats
    tasks.append(
        ('serve_cereal_in_bowl_at_servingtable1', taskplan_multi.pddl.task.serve_item('cereal', 'bowl', 'servingtable1'), 0.5)
    )
    tasks.append(
        ('serve_cereal_in_bowl_at_servingtable2', taskplan_multi.pddl.task.serve_item('cereal', 'bowl', 'servingtable2'), 0.5)
    )
    tasks.append(
        ('serve_cereal_in_mug_at_servingtable1', taskplan_multi.pddl.task.serve_item('cereal', 'mug', 'servingtable1'), 0.5)
    )
    tasks.append(
        ('serve_cereal_in_mug_at_servingtable2', taskplan_multi.pddl.task.serve_item('cereal', 'mug', 'servingtable2'), 0.5)
    )

    # Serve Milk
    tasks.append(
        ('serve_milk_in_bowl_at_servingtable1', taskplan_multi.pddl.task.serve_item('milk', 'bowl', 'servingtable1'), 0.5)
    )
    tasks.append(
        ('serve_milk_in_bowl_at_servingtable2', taskplan_multi.pddl.task.serve_item('milk', 'bowl', 'servingtable2'), 0.5)
    )
    tasks.append(
        ('serve_milk_in_mug_at_servingtable1', taskplan_multi.pddl.task.serve_item('milk', 'mug', 'servingtable1'), 0.5)
    )
    tasks.append(
        ('serve_milk_in_mug_at_servingtable2', taskplan_multi.pddl.task.serve_item('milk', 'mug', 'servingtable2'), 0.5)
    )

    # Restock items
    tasks.append(
        ('restock_pasta', taskplan_multi.pddl.task.restock_something('pasta'), 0.5)
    )
    tasks.append(
        ('restock_oats', taskplan_multi.pddl.task.restock_something('oats'), 0.5)
    )
    tasks.append(
        ('restock_cereal', taskplan_multi.pddl.task.restock_something('cereal'), 0.5)
    )
    tasks.append(
        ('restock_milk', taskplan_multi.pddl.task.restock_something('milk'), 0.5)
    )

    # Rearrangeent task
    # tasks.append(
    #     ('organize_sauce_in_cabinet', taskplan_multi.pddl.task.place_something('sauce', 'cabinet'), 0.5)
    # )
    # tasks.append(
    #     ('organize_saltshaker_in_cabinet', taskplan_multi.pddl.task.place_something('saltshaker', 'cabinet'), 0.5)
    # )
    # tasks.append(
    #     ('organize_pasta_in_cabinet', taskplan_multi.pddl.task.place_something('pasta', 'cabinet'), 0.5)
    # )
    # tasks.append(
    #     ('organize_oats_in_fridge', taskplan_multi.pddl.task.place_something('oats', 'fridge'), 0.5)
    # )
    # tasks.append(
    #     ('organize_cereal_in_fridge', taskplan_multi.pddl.task.place_something('cereal', 'fridge'), 0.5)
    # )
    # tasks.append(
    #     ('organize_milk_in_fridge', taskplan_multi.pddl.task.place_something('milk', 'fridge'), 0.5)
    # )

    return tasks

def tasks_for_cleaner():
    tasks = list()
    tasks.append(
        ('clean_mug', taskplan_multi.pddl.task.clean_something('mug1'), 0.5)
    )
    tasks.append(
        ('clean_mug', taskplan_multi.pddl.task.clean_something('mug2'), 0.5)
    )
    tasks.append(
        ('clean_bowl', taskplan_multi.pddl.task.clean_something('bowl1'), 0.5)
    )
    tasks.append(
        ('clean_bowl', taskplan_multi.pddl.task.clean_something('bowl2'), 0.5)
    )
    tasks.append(
        ('clean_pan', taskplan_multi.pddl.task.clean_something('pan1'), 0.5)
    )
    tasks.append(
        ('clean_pan', taskplan_multi.pddl.task.clean_something('pan2'), 0.5)
    )

    # Both 
    tasks.append(
        ('clean_both_mug', taskplan_multi.pddl.task.clean_both_item('mug'), 0.5)
    )
    tasks.append(
        ('clean_both_bowl', taskplan_multi.pddl.task.clean_both_item('bowl'), 0.5)
    )
    tasks.append(
        ('clean_both_pan', taskplan_multi.pddl.task.clean_both_item('pan'), 0.5)
    )

    # Clean 1 item each 
    tasks.append(
        ('clean_all_items', taskplan_multi.pddl.task.clean_items('mug1', 'bowl1', 'pan1'), 0.5)
    )
    tasks.append(
        ('clean_all_items', taskplan_multi.pddl.task.clean_items('mug1', 'bowl2', 'pan1'), 0.5)
    )
    tasks.append(
        ('clean_all_items', taskplan_multi.pddl.task.clean_items('mug1', 'bowl1', 'pan2'), 0.5)
    )
    tasks.append(
        ('clean_all_items', taskplan_multi.pddl.task.clean_items('mug1', 'bowl2', 'pan2'), 0.5)
    )
    tasks.append(
        ('clean_all_items', taskplan_multi.pddl.task.clean_items('mug2', 'bowl1', 'pan1'), 0.5)
    )
    tasks.append(
        ('clean_all_items', taskplan_multi.pddl.task.clean_items('mug2', 'bowl2', 'pan1'), 0.5)
    )
    tasks.append(
        ('clean_all_items', taskplan_multi.pddl.task.clean_items('mug2', 'bowl1', 'pan2'), 0.5)
    )
    tasks.append(
        ('clean_all_items', taskplan_multi.pddl.task.clean_items('mug2', 'bowl2', 'pan2'), 0.5)
    )

    # Sorting items
    tasks.append(
        ('sort_mug', taskplan_multi.pddl.task.sort_item('mug1'), 0.5)
    )
    tasks.append(
        ('sort_mug', taskplan_multi.pddl.task.sort_item('mug2'), 0.5)
    )
    tasks.append(
        ('sort_bowl', taskplan_multi.pddl.task.sort_item('bowl1'), 0.5)
    )
    tasks.append(
        ('sort_bowl', taskplan_multi.pddl.task.sort_item('bowl2'), 0.5)
    )
    tasks.append(
        ('sort_pan', taskplan_multi.pddl.task.sort_item('pan1'), 0.5)
    )
    tasks.append(
        ('sort_pan', taskplan_multi.pddl.task.sort_item('pan2'), 0.5)
    )
    return tasks


# def available_tasks_for_cleaner(dirty, on_items):
#     tasks = list()
#     for item in dirty:
#         tasks.append(
#             {
#                 f'clean_{item}': taskplan_multi.pddl.task.clean_something(item),
#             }
#         )
#     for item in on_items:
#         tasks.append(
#             {
#                 f'clear_bussingcart': taskplan_multi.pddl.task.clear_surface(item, 'bussingcart'),
#             }
#         )
#     return tasks
