import math
import numpy as np
from shapely.geometry import Polygon, Point
import random
import copy
import json

import gridmap
from taskplan_multi.environments.sampling import generate_restaurant, load_movables, load_assets
from taskplan_multi.utilities.restaurant_primitives import KITCHEN_CONTAINERS, SERVING_ROOM_CONTAINERS, COOK_BOT_RESTRICT, SERVER_BOT_RESTRICT, CLEANER_BOT_RESTRICT
from ai2thor.controller import Controller


INFLATE_UB = 0.15
INFLATE_LB = 0.1

PROCTHOR_DATA_PATH = './data/procthor-data/data.jsonl'
AI2THOR_PATH = "./data/ai2thor/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917"


def get_apartment(seed=3):
    with open(
        PROCTHOR_DATA_PATH,
        "r",
    ) as json_file:
        json_list = list(json_file)
    return json.loads(json_list[seed])


def load_restaurant(seed, agents):
    """
    Keep the name of the containers in ascending order
    that you want to place in the corners
    Also keep 'agent' in kitchen container list
    """
    kitchen_containers_list = list(KITCHEN_CONTAINERS)
    random.shuffle(kitchen_containers_list)
    if 'cook_bot' in agents:
        kitchen_containers_list.insert(0, 'cook_bot')
    serving_room_containers_list = list(SERVING_ROOM_CONTAINERS)
    if 'server_bot' in agents and 'cleaner_bot' in agents:
        if random.random() > 0.75:
            serving_room_containers_list.insert(0, 'server_bot')
            kitchen_containers_list.insert(0, 'cleaner_bot')
        elif random.random() > 0.5:
            serving_room_containers_list.insert(0, 'cleaner_bot')
            kitchen_containers_list.insert(0, 'server_bot')
        elif random.random() > 0.25:
            serving_room_containers_list.insert(0, 'server_bot')
            serving_room_containers_list.insert(0, 'cleaner_bot')
        else:
            kitchen_containers_list.insert(0, 'server_bot')
            kitchen_containers_list.insert(0, 'cleaner_bot')
    elif 'server_bot' in agents:
        if random.random() > 0.5:
            serving_room_containers_list.insert(0, 'server_bot')
        else:
            kitchen_containers_list.insert(0, 'server_bot')
    elif 'cleaner_bot' in agents:
        if random.random() > 0.5:
            serving_room_containers_list.insert(0, 'cleaner_bot')
        else:
            kitchen_containers_list.insert(0, 'cleaner_bot')

    datum = generate_restaurant(seed, kitchen_containers_list,
                                serving_room_containers_list)
    return datum


def world_to_grid(x, z, min_x, min_z, resolution):
    grid_x = int((x - min_x) / resolution)
    grid_z = int((z - min_z) / resolution)
    return grid_x, grid_z


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# Function to draw a line on the grid with a specific value
def draw_line(grid, x1, z1, x2, z2, value):
    num_steps = max(abs(x2 - x1), abs(z2 - z1))
    for step in range(num_steps + 1):
        inter_x = int(x1 + step * (x2 - x1) / num_steps)
        inter_z = int(z1 + step * (z2 - z1) / num_steps)
        grid[inter_x, inter_z] = value


def draw_polygon_on_grid(polygon, grid, value, min_x, min_z, resolution):
    exterior_coords = list(polygon.exterior.coords)
    for i in range(len(exterior_coords) - 1):
        x1, z1 = exterior_coords[i]
        x2, z2 = exterior_coords[i + 1]
        grid_x1, grid_z1 = world_to_grid(x1, z1, min_x, min_z, resolution)
        grid_x2, grid_z2 = world_to_grid(x2, z2, min_x, min_z, resolution)
        draw_line(grid, grid_x1, grid_z1, grid_x2, grid_z2, value)


# Function to update the occupancy grid with rectangles
def update_occupancy_grid_with_rectangles(occupancy_grid, rectangle, min_x, min_z, resolution, val):
    # Get the bounding box of the rectangle
    min_x_rect, min_z_rect, max_x_rect, max_z_rect = rectangle.bounds

    # Calculate grid indices for the bounding box
    grid_min_x, grid_min_z = world_to_grid(min_x_rect, min_z_rect, min_x, min_z, resolution)
    grid_max_x, grid_max_z = world_to_grid(max_x_rect, max_z_rect, min_x, min_z, resolution)

    # Iterate over all the grid cells within the bounding box
    for grid_x in range(grid_min_x, grid_max_x + 1):
        for grid_z in range(grid_min_z, grid_max_z + 1):
            world_x = min_x + grid_x * resolution
            world_z = min_z + grid_z * resolution
            point = Point(world_x, world_z)

            # Check if the point is inside the rectangle
            if rectangle.contains(point) or rectangle.touches(point):
                occupancy_grid[grid_x, grid_z] = val


# Function to inflate polygons by a specified distance
def inflate_polygon(polygon, distance):
    return polygon.buffer(distance)


# Function to get points around a single container that are not occupied
def get_unoccupied_points_around_container(occupancy_grid, min_x, min_z,
                                           resolution, container,
                                           inflation_distance, mother_poly):

    occupancy_grid = gridmap.utils.inflate_grid(occupancy_grid, INFLATE_LB*10)
    unoccupied_points = []
    grid_height, grid_width = occupancy_grid.shape

    inflated_container = inflate_polygon(container, INFLATE_LB)
    # Inflate the container polygon
    inflated_polygon = inflate_polygon(container, inflation_distance)

    # Get the bounds of the inflated polygon
    min_x_inf, min_z_inf, max_x_inf, max_z_inf = inflated_polygon.bounds

    # Calculate grid indices for the inflated polygon bounds
    grid_min_x_inf, grid_min_z_inf = world_to_grid(min_x_inf, min_z_inf,
                                                   min_x, min_z, resolution)
    grid_max_x_inf, grid_max_z_inf = world_to_grid(max_x_inf, max_z_inf,
                                                   min_x, min_z, resolution)

    # Iterate over the grid cells within the inflated polygon bounds
    for grid_x in range(max(0, grid_min_x_inf),
                        min(grid_height, grid_max_x_inf + 1)):
        for grid_z in range(max(0, grid_min_z_inf),
                            min(grid_width, grid_max_z_inf + 1)):
            world_x = min_x + grid_x * resolution
            world_z = min_z + grid_z * resolution
            point = Point(world_x, world_z)

            # Check if the point is within the inflated polygon, not within the original polygon, and not occupied
            if inflated_polygon.contains(point) and mother_poly.contains(
                point) and (
                    not inflated_container.contains(
                        point)):
                gx, gz = world_to_grid(world_x, world_z,
                                       min_x, min_z, resolution)
                if occupancy_grid[gx, gz] == 1:
                    continue
                unoccupied_points.append((world_x, world_z))

    p1 = (container.centroid.x, container.centroid.y)
    min_dis = float('inf')
    c_point = ()
    for p2 in unoccupied_points:
        dis = euclidean_distance(p1, p2)
        if (dis < min_dis):
            min_dis = dis
            c_point = p2
    return c_point


def get_nearest_free_point(point, free_points):
    _min = 1000000000
    tp = point
    fp_idx = 0
    for idx, rp in enumerate(free_points):
        dist = (rp[0]-tp['x'])**2 + (rp[1]-tp['z'])**2
        if dist < _min:
            _min = dist
            fp_idx = idx
    return free_points[fp_idx]



def change_assets(restaurant):
    assets = load_assets()
    movables = load_movables()
    MAX_OBJ_PER_CONT = 2
    map_assets = {
        'Countertop_L_10x8': {
            'assetId': 'Countertop_C_10x10',
            'description': 'place to make food',
            'plid': 'countertop',

        },
        'Fridge_29': {
            'assetId': 'Fridge_14',
            'description': 'fridge',
            'plid': 'fridge',

        },
        'Dining_Table_218_1': {
            'assetId': 'Desk_229_1',
            'description': 'table for serving',
            'plid': 'servingtable1',

        },
        'Shelving_Unit_303_1': {
            'assetId': 'Shelving_Unit_303_1',
            'description': 'shelf',
            'plid': 'shelf',

        },
        'Stool_4_1': {
            'assetId': 'Washing_Machine_1',
            'description': 'stove',
            'plid': 'stove',

        },
        'TV_Stand_206_3': {
            'assetId': 'Cart_1',
            'description': 'bussingcart',
            'plid': 'bussingcart',

        },
        'Dining_Table_205_1': {
            'assetId': 'Desk_229_1',
            'description': 'table for serving',
            'plid': 'servingtable2',

        },
        'Sofa_214_1': {
            'assetId': 'Shelving_Unit_206_1',
            'description': 'pantry',
            'plid': 'pantry',

        },
        'Side_Table_302_1_6': {
            'assetId': 'Shelving_Unit_324_1',
            'description': 'cabinet',
            'plid': 'cabinet',

        },
        'Armchair_207_4': {
            'assetId': 'Sink_10',
            'description': 'sink',
            'plid': 'sink',

        },
    }

    # new_conatiners = []
    for idx, container in enumerate(restaurant['objects']):
        if container['assetId'] in list(map_assets.keys()):
            temp = map_assets[container['assetId']]
            container.update({'assetId': temp['assetId']})
            container.update({'plid': temp['plid']})
            container.update({'description': temp['description']})
            # container.pop('kinematic', None)
            # container.pop('rotation', None)
            # container.pop('layer', None)
            # container.pop('material', None)
        
        # children = list()
        # max_pop = min(len(movables), MAX_OBJ_PER_CONT)
        # min_pop = min(len(movables), 1)
        # rand_item = random.randint(min_pop, max_pop)
        # if idx + 1 == len(restaurant['objects']):
        #     rand_item = len(movables)
        # for i in range(rand_item):
        #     t = movables.pop()
        #     if 'washable' in t and random.random() > 0.5:
        #         t['dirty'] = 1
        #     if 'food' in t and random.random() > 0.5:
        #         t['empty'] = 1
        #     children.append(t)
        # container.update({'children': children})
        # new_conatiners.append(container)
    # return new_conatiners



class RESTAURANT:
    def __init__(self, seed, agents=['cook_bot', 'server_bot', 'cleaner_bot'], active='cook_bot'):
        self.seed = seed
        self.agent_list = agents
        self.active_robot = active
        self.initial_object_state = list()
        self.active_all = False
        self.asked_help = False

        ### LOAD MAP
        self.grid_resolution = 0.05
        self.grid_offset = []
        self.grid_limits = []

        self.restaurant = get_apartment(self.seed)
        change_assets(self.restaurant)
        self.prev_restaurant = load_restaurant(self.seed, agents)
        self.rooms = self.restaurant['rooms']
        self.doors = self.restaurant['doors']
        self.containers = self.restaurant['objects']

        self.controller = Controller(scene=self.restaurant,
                                     local_executable_path=AI2THOR_PATH,
                                     gridSize=self.grid_resolution,
                                     width=480, height=480)
        self.occupancy_grid = self.get_occupancy_grid()
        # for room in self.prev_restaurant['rooms']:
        #     print(room)
        # for room in self.restaurant['rooms']:
        #     print(room)
        # print("=============")
        # for cont in self.prev_restaurant['objects']:
        #     print(cont)
        # for cont in self.restaurant['objects']:
        #     print(cont['assetId'])
        # raise NotImplementedError
        
        # self.init_containers = copy.deepcopy(self.containers)
        # self.grid, self.grid_min_x, self.grid_min_z, self.grid_max_x, \
        #     self.grid_max_z, self.grid_res = self.set_occupancy_grid()
        # inflation_distance = INFLATE_UB
        # relative_loc = {}
        
        
        # for agent in agents:
        #     agent_poly = Polygon([(point['x'], point['z'])
        #                             for point in self.restaurant[agent]['polygon']])
        #     if self.restaurant[agent]['loc'] == 'kitchen':
        #         mother_poly = Polygon([(point['x'], point['z'])
        #                                     for point in self.rooms['kitchen']['polygon']])
        #     else:
        #         mother_poly = Polygon([(point['x'], point['z'])
        #                                     for point in self.rooms['servingroom']['polygon']])
        #     point_cloud = get_unoccupied_points_around_container(
        #                                         self.grid,
        #                                         self.grid_min_x, self.grid_min_z,
        #                                         self.grid_res,
        #                                         agent_poly,
        #                                         inflation_distance,
        #                                         mother_poly
        #                                     )
        #     while len(point_cloud) == 0:
        #         inflation_distance += 0.05
        #         point_cloud = get_unoccupied_points_around_container(
        #                                     self.grid,
        #                                     self.grid_min_x, self.grid_min_z,
        #                                     self.grid_res,
        #                                     agent_poly,
        #                                     inflation_distance,
        #                                     mother_poly
        #                                 )
        #     agent_base = 'base_' + agent
        #     relative_loc[agent_base] = point_cloud
        #     self.restaurant[agent]['rob_at'] = agent_base
        # inflation_distance = INFLATE_UB
        # self.known_cost = {}
        # for container in self.containers:
        #     children = container.get('children')
        #     for child in children:
        #         self.initial_object_state.append(child)
        #     cont_ploy = Polygon([(point['x'], point['z'])
        #                          for point in container['polygon']])
        #     mother_poly = Polygon([(point['x'], point['z'])
        #                            for point in self.rooms['servingroom']['polygon']])
        #     if container['loc'] == 'kitchen':
        #         mother_poly = Polygon([(point['x'], point['z'])
        #                                for point in self.rooms['kitchen']['polygon']])
        #     point_cloud = get_unoccupied_points_around_container(
        #                                     self.grid,
        #                                     self.grid_min_x, self.grid_min_z,
        #                                     self.grid_res,
        #                                     cont_ploy,
        #                                     inflation_distance,
        #                                     mother_poly
        #                                 )
        #     while len(point_cloud) == 0:
        #         inflation_distance += 0.05
        #         point_cloud = get_unoccupied_points_around_container(
        #                                     self.grid,
        #                                     self.grid_min_x, self.grid_min_z,
        #                                     self.grid_res,
        #                                     cont_ploy,
        #                                     inflation_distance,
        #                                     mother_poly
        #                                 )
        #     inflation_distance = INFLATE_UB
        #     relative_loc[container['assetId']] = point_cloud
        # self.accessible_poses = {}
        # for item1 in relative_loc:
        #     point1 = relative_loc[item1]
        #     s_x, s_z = world_to_grid(point1[0], point1[1],
        #                              self.grid_min_x,
        #                              self.grid_min_z, self.grid_res)
        #     self.accessible_poses[item1] = (s_x, s_z)
        #     cost_grid = gridmap.planning.compute_cost_grid_from_position(
        #             self.grid,
        #             start=[
        #                 s_x,
        #                 s_z
        #             ],
        #             only_return_cost_grid=True)
        #     for item2 in relative_loc:
        #         # if item2 in self.known_cost and item1 in self.known_cost[item2]:
        #         #     continue
        #         point2 = relative_loc[item2]
        #         e_x, e_z = world_to_grid(point2[0], point2[1],
        #                                  self.grid_min_x,
        #                                  self.grid_min_z, self.grid_res)
        #         cost = cost_grid[e_x, e_z]
        #         if item1 not in self.known_cost:
        #             self.known_cost[item1] = {}
        #         self.known_cost[item1][item2] = cost
        # print(self.initial_object_poses)
        # print(self.known_cost)
        # for item in self.known_cost:
        #     print(item)
        #     print(self.known_cost[item])
        # raise NotImplementedError

    # def set_occupancy_grid(self):
    #     # Convert to Shapely Polygons
    #     kitchen = self.rooms['kitchen']['polygon']
    #     serving_room = self.rooms['servingroom']['polygon']
    #     kitchen_polygon = Polygon([(point['x'], point['z'])
    #                                for point in kitchen])
    #     serving_room_polygon = Polygon([(point['x'], point['z'])
    #                                     for point in serving_room])
    #     door = self.restaurant['doors']['door1']['position']

    #     # Define the resolution of the grid (e.g., each cell is 0.1 units)
    #     resolution = 0.1

    #     # Merge the two room points
    #     all_points = kitchen + serving_room

    #     # Extract x and z coordinates
    #     x_coords = [point['x'] for point in all_points]
    #     z_coords = [point['z'] for point in all_points]

    #     # Determine the bounds of the grid
    #     min_x, max_x = min(x_coords), max(x_coords)
    #     min_z, max_z = min(z_coords), max(z_coords)

    #     # Calculate grid dimensions
    #     grid_height = int((max_x - min_x) / resolution) + 1
    #     grid_width = int((max_z - min_z) / resolution) + 1

    #     # Initialize the occupancy grid (0 for free space, 1 for occupied)
    #     occupancy_grid = np.zeros((grid_height, grid_width), dtype=int)

    #     # Draw the kitchen and serving room polygons on the grid
    #     draw_polygon_on_grid(kitchen_polygon, occupancy_grid, 1,
    #                          min_x, min_z, resolution)
    #     draw_polygon_on_grid(serving_room_polygon, occupancy_grid, 1,
    #                          min_x, min_z, resolution)

    #     # Draw the door
    #     door_start_grid_x, door_start_grid_z = world_to_grid(
    #         door[0]['x'], door[0]['z'], min_x, min_z, resolution)
    #     door_end_grid_x, door_end_grid_z = world_to_grid(
    #         door[1]['x'], door[1]['z'], min_x, min_z, resolution)
    #     draw_line(occupancy_grid, door_start_grid_x, door_start_grid_z,
    #               door_end_grid_x, door_end_grid_z, 0)

    #     # Draw the containers (for now points)
    #     containers = self.restaurant['objects']
    #     for container in containers:
    #         rectangle = Polygon([(point['x'], point['z'])
    #                              for point in container['polygon']])
    #         update_occupancy_grid_with_rectangles(occupancy_grid, rectangle,
    #                                               min_x, min_z, resolution, 1)
        
    #     for agent in self.agent_list:
    #         agent_poly = Polygon([(point['x'], point['z'])
    #                                 for point in self.restaurant[agent]['polygon']])
    #         update_occupancy_grid_with_rectangles(occupancy_grid, agent_poly,
    #                                                 min_x, min_z, resolution, 1)

    #     return occupancy_grid, min_x, min_z, max_x, max_z, resolution

    def set_grid_offset(self, min_x, min_y):
        self.grid_offset = np.array([min_x, min_y])
    
    def scale_to_grid(self, point):
        x = round((point[0] - self.grid_offset[0]) / self.grid_resolution)
        y = round((point[1] - self.grid_offset[1]) / self.grid_resolution)
        return x, y

    def get_occupancy_grid(self):
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
        print(reachable_positions)
        RPs = reachable_positions

        xs = [rp["x"] for rp in reachable_positions]
        zs = [rp["z"] for rp in reachable_positions]

        # Calculate the mins and maxs
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        x_offset = min_x - self.grid_resolution if min_x < 0 else 0
        z_offset = min_z - self.grid_resolution if min_z < 0 else 0
        self.set_grid_offset(x_offset, z_offset)

        # Create list of free points
        points = list(zip(xs, zs))
        # print(points)
        grid_to_points_map = {self.scale_to_grid(point): RPs[idx]
                              for idx, point in enumerate(points)}
        height, width = self.scale_to_grid([max_x, max_z])
        occupancy_grid = np.ones((height+2, width+2), dtype=int)
        free_positions = grid_to_points_map.keys()
        for pos in free_positions:
            occupancy_grid[pos] = 0

        # store the mapping from grid coordinates to simulator positions
        self.g2p_map = grid_to_points_map

        # set the nearest freespace container positions
        for container in self.containers:
            position = container['position']
            if position is not None:
                # get nearest free space pose
                nearest_fp = get_nearest_free_point(position, points)
                # then scale the free space pose to grid
                scaled_position = self.scale_to_grid(np.array([nearest_fp[0], nearest_fp[1]]))  # noqa: E501
                # finally set the scaled grid pose as the container position
                container['position'] = scaled_position  # 2d only

                # next do the same if there is any children of this container
                if 'children' in container:
                    children = container['children']
                    for child in children:
                        child['position'] = container['position']

        for room in self.rooms:
            floor = [(rp["x"], rp["z"]) for rp in room["floorPolygon"]]
            room_poly = Polygon(floor)
            point = room_poly.centroid
            point = {'x': point.x, 'z': point.y}
            nearest_fp = get_nearest_free_point(point, points)
            scaled_position = self.scale_to_grid(np.array([nearest_fp[0], nearest_fp[1]]))  # noqa: E501
            room['position'] = scaled_position  # 2d only

        return occupancy_grid

    def get_current_object_state(self):
        current_state = list()
        for container in self.containers:
            children = copy.deepcopy(container.get('children'))
            if children is None:
                continue
            for child in children:
                current_state.append(child)
        return current_state

    def roll_back_to_init(self):
        self.containers = copy.deepcopy(self.init_containers)
    
    def randomize_objects_state(self, randomness=[0, 0.5, 0.5], bias=False):
        states = self.get_current_object_state()
        if bias:
            if self.active_robot == 'cook_bot':
                conts = COOK_BOT_RESTRICT
            elif self.active_robot == 'server_bot':
                conts = SERVER_BOT_RESTRICT
            else:
                conts = CLEANER_BOT_RESTRICT
            poses = [val for (key, val) in self.get_container_pos_list() if key not in conts]
        else:
            poses = [val for (key, val) in self.get_container_pos_list()]
        for state in states:
            if randomness[0] == 1:
                state['position'] = random.choice(poses)
            if 'washable' in state:
                if random.random() > randomness[1] or bias:
                    state['dirty'] = 0
                else:
                    state['dirty'] = 1
            if 'food' in state:
                if random.random() > randomness[2] or bias:
                    state['empty'] = 0
                else:
                    state['empty'] = 1
        return states

    def get_objects_by_container_name(self, name):
        for container in self.containers:
            if container.get('assetId') == name:
                temp = copy.deepcopy(container.get('children'))
                return temp

    def get_container_pos_list(self):
        return [(container['assetId'], container['position'])
                for container in self.containers]

    def get_container_pos(self, name):
        for container in self.containers:
            if container['assetId'] == name:
                return container['position']
        return None

    def get_container_name_by_pos(self, pos):
        for container in self.containers:
            if container['position'] == pos:
                return container['assetId']
        return None

    def get_object_props_by_name(self, name):
        state = self.get_current_object_state()
        for objct in state:
            if objct['assetId'] == name:
                return copy.deepcopy(objct)
        return None
    

    def get_final_state_from_plan(self, plan):
        objects = self.get_current_object_state()
        locations_dict = dict(self.get_container_pos_list())
        dirty_objs = set()
        cleaned_obsj = set()
        empty_foods = set()
        full_foods = set()
        conditions = []
        for p in plan:
            if "place" in p.name:
                obj = p.args[1]
                cnt = p.args[2]
                placed = (obj, cnt)
                conditions.append(placed)
            if "wash" in p.name:
                cleaned_obsj.add(p.args[1])
                if p.args[1] in dirty_objs:
                    dirty_objs.remove(p.args[1])
            if "cook" in p.name:
                empty_foods.add(p.args[1])
                dirty_objs.add(p.args[2])
                if p.args[1] in full_foods:
                    full_foods.remove(p.args[1])
                if p.args[2] in cleaned_obsj:
                    cleaned_obsj.remove(p.args[2])
            if "serve" in p.name:
                empty_foods.add(p.args[1])
                if p.args[1] in full_foods:
                    full_foods.remove(p.args[1])
                dirty_objs.add(p.args[2])
                if p.args[2] in cleaned_obsj:
                    cleaned_obsj.remove(p.args[2])
            if "mix" in p.name:
                empty_foods.add(p.args[1])
                if p.args[1] in full_foods:
                    full_foods.remove(p.args[1])
                dirty_objs.add(p.args[2])
                if p.args[2] in cleaned_obsj:
                    cleaned_obsj.remove(p.args[2])
            if "restock" in p.name:
                full_foods.add(p.args[1])
                if p.args[1] in empty_foods:
                    empty_foods.remove(p.args[1])
            # if "move" in p.name:
            #     rob_at = p.args[1]

        for obj_dict in objects:
            if obj_dict['assetId'] in dirty_objs:
                obj_dict['dirty'] = 1
            if obj_dict['assetId'] in cleaned_obsj:
                obj_dict['dirty'] = 0
            if obj_dict['assetId'] in full_foods:
                obj_dict['empty'] = 0
            if obj_dict['assetId'] in empty_foods:
                obj_dict['empty'] = 1
            for cond in conditions:  # Directly check if the condition's object key is in the dictionary
                if cond[0] == obj_dict['assetId'] and cond[1] in locations_dict:
                    obj_dict['position'] = locations_dict[cond[1]]
        return objects

    def update_container_props(self, object_state):
        for container in self.containers:
            children = []
            for child in object_state:
                if child.get('position') == container.get('position'):
                    children.append(child)
            container.update({'children': children})
    
    def place_object(self, obj, loc):
        state = self.get_current_object_state()
        new_objs = list()
        for objct in state:
            if objct['assetId'] == obj['assetId']:
                objct.update({'position': loc})
            new_objs.append(objct)
        return new_objs

    def place_washables(self, obj, loc, dirty=0):
        state = self.get_current_object_state()
        new_objs = list()
        for objct in state:
            if objct['assetId'] == obj['assetId']:
                objct.update({'position': loc})
                objct.update({'dirty': dirty})
            new_objs.append(objct)
        return new_objs
    
    def place_food_items(self, obj, loc, empty=0):
        state = self.get_current_object_state()
        new_objs = list()
        for objct in state:
            if objct['assetId'] == obj['assetId']:
                objct.update({'position': loc})
                objct.update({'empty': empty})
            new_objs.append(objct)
        return new_objs
    

    def get_top_down_image(self, orthographic=True):
        # Setup top down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = orthographic
        pose["farClippingPlane"] = 50
        if orthographic:
            pose["orthographicSize"] = 0.5 * max_bound
        else:
            del pose["orthographicSize"]

        # Add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_image = event.third_party_camera_frames[-1]
        top_down_image = top_down_image[::-1, ...]
        return top_down_image