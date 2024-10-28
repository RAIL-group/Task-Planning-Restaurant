def move_robot(agent, container):
    return f'''
            (rob-at {agent} {container})
            '''


def place_something(item, container):
    return f'''
            (is-at {item} {container})
            '''