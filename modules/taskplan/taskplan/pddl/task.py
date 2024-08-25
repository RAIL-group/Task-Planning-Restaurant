def pour_water(container=None):
    if container:
        str = f'(filled-with water {container})'
    else:
        str = '''
        (exists
            (?c - item)
                (and (filled-with water ?c))
        )
        '''
    return str


def empty_container(container):
    str = f'(not (filled-with water {container}))'
    return str


def serve_water(location, container=None):
    if container:
        str = f'(and (filled-with water {container}) (is-at {container} {location}))'
    else:
        str = f'''
        (exists
            (?c - item)
                (and (filled-with water ?c) (is-at ?c {location}))
        )
        '''
    return str


def fill_coffeemachine_with_water():
    return '''
            (is-at water coffeemachine)
        '''


def make_coffee(container=None):
    if container:
        str = f'(filled-with coffee {container})'
    else:
        str = '''
        (exists
            (?c - item)
                (and (filled-with coffee ?c))
        )
        '''
    return str


def serve_coffee(location, container=None):
    if container:
        str = f'(and (filled-with coffee {container}) (is-at {container} {location}))'
    else:
        str = f'''
        (exists
            (?c - item)
                (and (filled-with coffee ?c) (is-at ?c {location}))
        )
        '''
    return str


def make_sandwich(spread=None):
    if spread:
        str = f'(spread-applied bread {spread})'
    else:
        str = '''
        (exists
            (?sprd - spread)
                (spread-applied bread ?sprd)
        )
        '''
    return str


def serve_sandwich(location, spread=None):
    if spread:
        str = f'(and (spread-applied bread {spread}) (is-at bread {location}))'
    else:
        str = f'''
        (exists
            (?sprd - spread)
                (and (spread-applied bread ?sprd) (is-at bread {location}))
        )
        '''
    return str


def hold_something():
    return '''
        (exists
            (?c - mug)
                (and (is-holding ?c))
        )
    '''


def clear_surface(location):
    return f'''
        (forall (?i - item)
            (not (is-at ?i {location}))
        )
    '''


def clean_everything():
    return '''
        (forall (?i - item)
            (not (is-dirty ?i))
        )
    '''


def clean_something(object):
    str = f'(not (is-dirty {object}))'
    return str


def place_something(item, container):
    return f'''
            (is-at {item} {container})
            '''


def clean_and_place(item, container):
    return f'''
            (and (is-at {item} {container}) (not (is-dirty {item})))
            '''


def move_robot(container):
    return f'''
            (rob-at {container})
            '''


def serve_fruit(bowl, table):
    return f'''
            (and (is-in apple {bowl}) (is-at {bowl} {table}))
            '''
