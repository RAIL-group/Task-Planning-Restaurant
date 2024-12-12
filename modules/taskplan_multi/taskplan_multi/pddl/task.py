def move_robot(agent, container):
    return f'''
            (rob-at {agent} {container})
            '''

def place_something(item, container):
    return f'''
            (is-at {item} {container})
            '''

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

def clean_all(item):
    str = f'''(and
                (forall (?m - item) (and (type ?m {item}) (not (is-dirty ?m))))
            )'''
    return str

def cook_something(item):
    str = f'(is-cooked {item})'
    return str

def make_oats():
    str = f'''(and
                (exists (?i - item) (and (is-cooked ?i) (is-at ?i countertop) (type ?i oats)))
                (exists (?m - item) (and (is-at ?m countertop) (type ?m milk)))
            )'''
    return str

def make_pasta():
    str = f'''(and
                (exists (?i - item) (and (is-cooked ?i) (is-at ?i countertop) (type ?i pasta)))
                (exists (?m - item) (and (is-at ?m countertop) (type ?m sauce)))
            )'''
    return str

def make_omelette():
    str = f'''(and
                (exists (?i - item) (and (is-cooked ?i) (is-at ?i stove) (type ?i egg)))
                (exists (?m - item) (and (is-at ?m stove) (type ?m saltshaker)))
            )'''
    return str

def serve_milk(container):
    str = f'''(and
                (exists (?i - item) (and (meal-served ?i {container}) (type ?i milk)))
                (exists (?m - item) (and (is-at ?m {container}) (not (is-dirty ?m)) (type ?m mug)))
            )'''
    return str

def serve_oats(container):
    str = f'''(and
                (exists (?i - item) (and (meal-served ?i {container}) (type ?i oats)))
                (exists (?m - item) (and (is-at ?m {container}) (type ?m milk)))
                (exists (?b - item) (and (is-at ?b {container}) (not (is-dirty ?b)) (type ?b bowl)))
            )'''
    return str

def serve_pasta(container):
    str = f'''(and
                (exists (?i - item) (and (meal-served ?i {container}) (type ?i pasta)))
                (exists (?m - item) (and (is-at ?m {container}) (type ?m sauce)))
                (exists (?b - item) (and (is-at ?b {container}) (not (is-dirty ?b)) (type ?b bowl)))
            )'''
    return str

def serve_omelette(container):
    str = f'''(and
                (exists (?i - item) (and (meal-served ?i {container}) (type ?i egg)))
                (exists (?m - item) (and (is-at ?m {container}) (type ?m saltshaker)))
                (exists (?b - item) (and (is-at ?b {container}) (not (is-dirty ?b)) (type ?b bowl)))
            )'''
    return str

def clear_surface(item, container):
    str = f'''(exists (?loc - location)
            (and (not (= ?loc {container})) (is-at {item} ?loc)))'''
    return str
