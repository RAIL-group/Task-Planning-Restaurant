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


def clean_something(object):
    str = f'(not (is-dirty {object}))'
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
