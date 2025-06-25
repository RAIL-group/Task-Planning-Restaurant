def move_robot(agent, container):
    return f'''
            (rob-at {agent} {container})
            '''

def place_something(item, container):
    return f'''
            (is-at {item} {container})
            '''

def bring_clean_item(pot, container):
    str = f'''(exists (?i - item)
        (and 
            (type ?i {pot})
            (not (is-dirty ?i))
            (is-at ?i {container})
        )
    )'''
    return str

def bring_two_items(garnish, pot, container):
    str = f'''(exists (?i - item)
        (and 
            (type ?i {pot})
            (not (is-dirty ?i))
            (is-at ?i {container})
            (is-at {garnish} {container})
        )
    )'''
    return str

def organize_something(item):
    return f'''
            (and
                (is-at {item} cabinet)
                (not (is-dirty {item}))
            )
            '''

def clean_something(item):
    str = f'(not (is-dirty {item}))'
    return str

def clean_both_item(item):
    item1 = item + '1'
    item2 = item + '2'
    str = f'(and (not (is-dirty {item1})) (not (is-dirty {item2})))'
    return str

def clean_all(item):
    str = f'''(forall (?m - item) (and (type ?m {item}) (not (is-dirty ?m))))'''
    return str

# def cook_something(item):
#     str = f'(is-cooked {item})'
#     return str

def prep_item(food, pot, container, garnish=None):
    if garnish: 
        str = f'''(and
                (exists (?i - item ?b - item)
                    (and 
                        (type ?i {food})
                        (type ?b {pot})
                        (item-in-item ?i ?b {container})
                    )
                )
                (is-at {garnish} {container})
            )'''
    else:
        str = f'''(exists (?i - item ?b - item)
                    (and 
                        (type ?i {food})
                        (type ?b {pot})
                        (item-in-item ?i ?b {container})
                    )
                )'''
    return str

def serve_item(food, pot, container, garnish=None):
    if garnish: 
        str = f'''(and
                (exists (?i - item ?b - item)
                    (and 
                        (type ?i {food})
                        (type ?b {pot})
                        (meal-served ?i ?b {container})
                    )
                )
                (is-at {garnish} {container})
            )'''
    else:
        str = f'''(exists (?i - item ?b - item)
                    (and 
                        (type ?i {food})
                        (type ?b {pot})
                        (meal-served ?i ?b {container})
                    )
                )'''
    return str


def clean_items(item1, item2, item3):
    str = f'(and (not (is-dirty {item1})) (not (is-dirty {item2})) (not (is-dirty {item3})))'
    return str

def sort_item(item):
    str = f'''
    (or
        (and (not (is-dirty {item})) (is-at {item} shelf))
        (and (is-dirty {item}) (is-at {item} bussingcart))
    )
    '''
    return str

def restock_something(item):
    str = f'(not (is-empty {item}))'
    return str

# def make_oats(container):
#     str = f'''(and
#                 (exists (?i - item) (and (is-cooked ?i) (is-at ?i {container}) (type ?i oats)))
#                 (exists (?m - item) (and (is-cooked ?m) (is-at ?m {container}) (type ?m milk)))
#             )'''
#     return str

# def make_milk():
#     str = f'''(and
#                 (exists (?m - item) (and (is-cooked ?m) (is-at ?m stove) (type ?m milk)))
#             )'''
#     return str

# def make_egg_pasta():
#     str = f'''(and
#                 (exists (?p - item) (and (is-cooked ?p) (is-at ?p stove) (type ?p pasta)))
#                 (exists (?e - item) (and (is-cooked ?e) (is-at ?e stove) (type ?e egg)))
#                 (exists (?s - item) (and (is-at ?s stove) (type ?s sauce)))
#                 (exists (?k - item) (and (is-at ?k stove) (type ?k saltshaker)))
#             )'''
#     return str

# def make_pasta(container):
#     str = f'''(and
#                 (exists (?i - item) (and (is-cooked ?i) (is-at ?i {container}) (type ?i pasta)))
#                 (exists (?m - item) (and (is-at ?m {container}) (type ?m sauce)))
#             )'''
#     return str

# def make_cereal(container):
#     str = f'''(and
#                 (exists (?i - item) (and (is-cooked ?i) (is-at ?i {container}) (type ?i cereal)))
#                 (exists (?m - item) (and (is-cooked ?m) (is-at ?m {container}) (type ?m milk)))
#             )'''
#     return str

# def make_omelette():
#     str = f'''(and
#                 (exists (?i - item) (and (is-cooked ?i) (is-at ?i stove) (type ?i egg)))
#                 (exists (?m - item) (and (is-at ?m stove) (type ?m saltshaker)))
#             )'''
#     return str

# def serve_milk(container):
#     str = f'''(exists (?i - item ?b - item) 
#                 (and 
#                     (type ?i milk)
#                     (type ?b mug)
#                     (meal-served ?i ?b {container})
#                 )
#             )'''
#     return str

# def serve_oats(container):
#     str = f'''(and
#                 (exists (?i - item ?b - item)
#                     (and 
#                         (type ?i oats)
#                         (type ?b bowl)
#                         (meal-served ?i ?b {container})
#                     )
#                 )
#                 (exists (?m - item) (and (is-cooked ?m) (is-at ?i {container}) (type ?m milk)))
#             )'''
#     return str

# def serve_cereal(container):
#     str = f'''(and
#                 (exists (?i - item ?b - item)
#                     (and 
#                         (type ?i oats)
#                         (type ?b bowl)
#                         (meal-served ?i ?b {container})
#                     )
#                 )
#                 (exists (?m - item) (and (is-cooked ?m) (is-at ?i {container}) (type ?m milk)))
#             )'''
#     return str

# def serve_pasta(container):
#     str = f'''(and
#              (exists (?i - item ?b - item)
#                 (and 
#                     (type ?i pasta)
#                     (type ?b bowl)
#                     (meal-served ?i ?b {container})
#                 )
#              )
#              (exists (?m - item) (and (is-at ?m {container}) (type ?m sauce)))
#             )'''
#     return str

# def serve_omelette(container):
#     str = f'''(and
#              (exists (?i - item ?b - item) 
#                 (and 
#                     (type ?i egg)
#                     (type ?b bowl)
#                     (meal-served ?i ?b {container})
#                 )
#              )
#              (exists (?m - item) (and (is-at ?m {container}) (type ?m saltshaker)))
#             )'''
#     return str

# def serve_egg_pasta(container):
#     str = f'''(and
#                 (exists (?i - item ?b - item)
#                     (and 
#                         (type ?i pasta)
#                         (type ?b bowl)
#                         (meal-served ?i ?b {container})
#                     )
#                 )
#                 (exists (?a - item) (and (is-at ?a {container}) (type ?a saltshaker)))
#                 (exists (?c - item) (and (is-at ?c {container}) (type ?c sauce)))
#                 (exists (?e - item) (and (is-at ?e {container}) (is-cooked ?e) (type ?e egg)))
#             )'''
#     return str

# def clear_surface(item, container):
#     str = f'''(cleared {item} {container})'''
#     return str

def clean_and_place_something(item, cont):
    str = f'(and (not (is-dirty {item})) (is-at {item} {cont}))'
    return str

def stock_and_place_something(item, cont):
    str = f'(and (not (is-empty {item})) (is-at {item} {cont}))'
    return str