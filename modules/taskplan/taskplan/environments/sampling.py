import random
from shapely.geometry import box, Polygon, LineString
import json
import os


def load_data():
    """
    Laods the 1940 layouts for the restaurant
    """
    random_seed = random.randint(0, 1939)
    root = "/modules/taskplan/taskplan/environments/layouts"
    json_file = ''
    for path, _, files in os.walk(root):
        for name in files:
            if 'restaurant_samples.json' == name:
                json_file = os.path.join(path, name)
                datum = json.load(open(json_file))
                return datum[random_seed]


def load_assets():
    """
    Laods the assets for the restaurant
    """
    root = "/modules/taskplan/taskplan/environments/layouts"
    json_file = ''
    for path, _, files in os.walk(root):
        for name in files:
            if 'assets.json' == name:
                json_file = os.path.join(path, name)
                datum = json.load(open(json_file))
                return datum


def create_corner_rectangles(room_points, min_width, max_width, min_height,
                             max_height, door_coords, door_buffer,
                             corner_conts=[]):
    """
    Creates containers in the corner of the room if possible.
    Max width and height is the parameters for container.
    Door buffer is to place the containers with respect to certain buffer.
    """
    min_x = min(point['x'] for point in room_points)
    max_x = max(point['x'] for point in room_points)
    min_z = min(point['z'] for point in room_points)
    max_z = max(point['z'] for point in room_points)

    room_polygon = Polygon([(point['x'], point['z']) for point in room_points])
    rectangles = []

    # Create a buffer around the door
    door_line = LineString(door_coords)
    door_buffer_polygon = door_line.buffer(door_buffer)
    buffer = 0.2

    corners = [
        (min_x + buffer, min_z + buffer),  # Bottom-left
        (max_x - buffer, min_z + buffer),  # Bottom-right
        (min_x + buffer, max_z - buffer),  # Top-left
        (max_x - buffer, max_z - buffer)   # Top-right
    ]

    for corner in corners:
        attempt = 0
        while attempt < 100:  # Avoid infinite loop by limiting the number of attempts
            width = random.uniform(min_width, max_width)
            height = random.uniform(min_height, max_height)

            if corner == (min_x + buffer, min_z + buffer):  # Bottom-left
                x1, z1 = corner
                x2, z2 = x1 + width, z1 + height
            elif corner == (max_x - buffer, min_z + buffer):  # Bottom-right
                x1, z1 = corner[0] - width, corner[1]
                x2, z2 = corner[0], corner[1] + height
            elif corner == (min_x + buffer, max_z - buffer):  # Top-left
                x1, z1 = corner[0], corner[1] - height
                x2, z2 = corner[0] + width, corner[1]
            elif corner == (max_x - buffer, max_z - buffer):  # Top-right
                x1, z1 = corner[0] - width, corner[1] - height
                x2, z2 = corner[0], corner[1]

            new_rectangle = box(x1, z1, x2, z2)
            inf_rec = new_rectangle.buffer(0.2)

            # Ensure the rectangle is within the room, doesn't intersect the door buffer,
            # and doesn't overlap with existing rectangles (including those passed as parameter)
            if (room_polygon.contains(inf_rec) and
                not inf_rec.intersects(door_buffer_polygon) and
                all(not inf_rec.intersects(existing_rectangle) for existing_rectangle in rectangles) and
                all(not inf_rec.intersects(existing_rectangle) for existing_rectangle in corner_conts)):
                rectangles.append(new_rectangle)
                break
            attempt += 1

    return rectangles


def create_non_overlapping_containers(room_points, num_rectangles, min_width,
                                      max_width, min_height, max_height,
                                      door_coords, door_buffer,
                                      corner_conts=[]):
    """
    Creates N number of non-over lapping containers
    """
    min_x = min(point['x'] for point in room_points)
    max_x = max(point['x'] for point in room_points)
    min_z = min(point['z'] for point in room_points)
    max_z = max(point['z'] for point in room_points)

    room_polygon = Polygon([(point['x'], point['z']) for point in room_points])
    rectangles = []

    # Create a buffer around the door
    door_line = LineString(door_coords)
    door_buffer_polygon = door_line.buffer(door_buffer)

    # Determine the boundary of the room to place rectangles closer to it
    boundary_width = min(min_width, min_height) * 0.5  # Consider a fraction of the smallest dimension as the boundary region

    for _ in range(num_rectangles):
        attempt = 0
        while attempt < 100:  # Avoid infinite loop by limiting the number of attempts
            width = random.uniform(min_width, max_width)
            height = random.uniform(min_height, max_height)

            # Attempt to position rectangle near the boundary
            if random.choice([True, False]):  # Choose whether to align with x or z boundary randomly
                x1 = random.choice([min_x + random.uniform(0, boundary_width), max_x - width - random.uniform(0, boundary_width)])
            else:
                x1 = random.uniform(min_x, max_x - width)

            if random.choice([True, False]):
                z1 = random.choice([min_z + random.uniform(0, boundary_width), max_z - height - random.uniform(0, boundary_width)])
            else:
                z1 = random.uniform(min_z, max_z - height)

            x2 = x1 + width
            z2 = z1 + height

            new_rectangle = box(x1, z1, x2, z2)
            inf_rec = new_rectangle.buffer(0.2)

            # Ensure the rectangle is within the room, doesn't intersect the door buffer, and doesn't overlap with existing rectangles
            if room_polygon.contains(inf_rec) and not inf_rec.intersects(door_buffer_polygon) and all(not inf_rec.intersects(existing_rectangle) for existing_rectangle in rectangles) and all(not inf_rec.intersects(existing_rectangle) for existing_rectangle in corner_conts):
                rectangles.append(new_rectangle)
                break
            attempt += 1

    return rectangles


def get_door(kitchen_polygon, serving_room_polygon, door_length=2.0):
    # Find the common wall
    common_wall = kitchen_polygon.intersection(serving_room_polygon.boundary)
    # Add a door in the middle of the common wall with length 2 and get the door coordinates
    if isinstance(common_wall, LineString):
        x1, z1, x2, z2 = *common_wall.coords[0], *common_wall.coords[1]
        door_center_x = (x1 + x2) / 2
        door_center_z = (z1 + z2) / 2
        door_half_length = door_length / 2
        if abs(x2 - x1) > abs(z2 - z1):  # Horizontal wall
            door_start_x = door_center_x - door_half_length
            door_end_x = door_center_x + door_half_length
            door_start_z = door_center_z
            door_end_z = door_center_z
        else:  # Vertical wall
            door_start_x = door_center_x
            door_end_x = door_center_x
            door_start_z = door_center_z - door_half_length
            door_end_z = door_center_z + door_half_length
        return [(door_start_x, door_start_z), (door_end_x, door_end_z)]
    return None


def generate_restaurant(seed, kitchen_containers_list,
                        serving_room_containers_list):
    """
    Takes a seed and two list of containers.
    One for kicthen and another for the serving room
    Returns the restaurant dictinonary
    """
    random.seed(seed)
    num_attempt = 100
    while True:
        datum = load_data()
        assets = load_assets()
        kitchen = datum[0]
        serving_room = datum[1]
        k_polygon = Polygon([(point['x'], point['z']) for point in kitchen])
        s_polygon = Polygon([(point['x'], point['z']) for point in serving_room])
        k_cen = k_polygon.centroid
        s_cen = s_polygon.centroid
        door_coords = get_door(k_polygon, s_polygon)
        min_width = 0.5
        max_width = 1.0
        min_height = 0.5
        max_height = 1.0
        door_buffer = 1.0  # Buffer around the door
        restaurant = {
                'rooms': {
                    'kitchen': {
                        'position': {
                            'x': k_cen.x,
                            'z': k_cen.y
                        },
                        'polygon': kitchen,
                        'name': 'kitchen',
                        "id": "kitchen"
                    },
                    'servingroom': {
                        'position': {
                            'x': s_cen.x,
                            'z': s_cen.y
                        },
                        'polygon': serving_room,
                        "name": "servingroom",
                        "id": "servingroom"
                    }
                },
                'doors': {
                    'door1': {
                        'position': [{'x': door_coords[0][0],
                                    'z': door_coords[0][1]},
                                    {'x': door_coords[1][0],
                                    'z': door_coords[1][1]}]
                    }
                },
                'agent': {
                    'name': 'robot'
                }
        }
        if len(kitchen_containers_list) == 0:
            kitchen_containers_list = ['dishwasher', 'fountain', 'coffeemachine', 'sandwichmaker', 'breadshelf', 'coffeeshelf', 'spreadshelf', 'cutleryshelf', 'dishshelf', 'mugshelf', 'cupshelf', 'agent']
        if len(serving_room_containers_list) == 0:
            serving_room_containers_list = ['servingtable1', 'servingtable2', 'servingtable3']

        # Create non-overlapping rectangles inside the kitchen
        kitchen_corners = create_corner_rectangles(kitchen, min_width, max_width,
                                                   min_height, max_height,
                                                   door_coords, door_buffer)
        min_num_cont_needed = len(kitchen_containers_list) + 1
        kitchen_random = create_non_overlapping_containers(kitchen,
                                                           min_num_cont_needed,
                                                           min_width, max_width,
                                                           min_height, max_height,
                                                           door_coords,
                                                           door_buffer,
                                                           kitchen_corners)

        service_corners = create_corner_rectangles(serving_room, min_width,
                                                   max_width, min_height,
                                                   max_height, door_coords,
                                                   door_buffer,
                                                   kitchen_corners + kitchen_random
                                                   )
        service_random = create_non_overlapping_containers(serving_room,
                                                           len(serving_room_containers_list),
                                                           min_width, max_width,
                                                           min_height, max_height,
                                                           door_coords,
                                                           door_buffer,
                                                           kitchen_corners + kitchen_random + service_corners
                                                           )
        if len(kitchen_containers_list) <= len(kitchen_corners) + len(kitchen_random):
            break
        num_attempt -= 1

        if num_attempt == 0:
            raise ValueError('Try increaing number of attempts.')

    containers = list()
    for item in kitchen_containers_list:
        if len(kitchen_corners) > 0:
            rectangle = kitchen_corners.pop()
        elif len(kitchen_random) > 0:
            rectangle = kitchen_random.pop()
        else:
            continue
        centroid = rectangle.centroid
        cords = list(rectangle.exterior.coords)
        temp = list()
        for cx, cz in cords:
            t_cords = {
                'x': cx,
                'z': cz
            }
            temp.append(t_cords)
        if item == 'agent':
            kc = restaurant['agent']
        else:
            kc = assets[item]
        kc['position'] = {
                'x': centroid.x,
                'z': centroid.y
        }
        kc['polygon'] = temp
        kc['loc'] = 'kitchen'
        if item == 'agent':
            continue
        containers.append(kc)

    for item in serving_room_containers_list:
        if len(service_corners) > 0:
            rectangle = service_corners.pop()
        elif len(service_random) > 0:
            rectangle = service_random.pop()
        else:
            continue
        centroid = rectangle.centroid
        cords = list(rectangle.exterior.coords)
        temp = list()
        for cx, cz in cords:
            t_cords = {
                'x': cx,
                'z': cz
            }
            temp.append(t_cords)
        kc = assets[item]
        kc['position'] = {
                    'x': centroid.x,
                    'z': centroid.y
        }
        kc['polygon'] = temp
        kc['loc'] = 'servingroom'
        x, y = rectangle.exterior.xy
        containers.append(kc)

    restaurant['objects'] = containers
    return restaurant
