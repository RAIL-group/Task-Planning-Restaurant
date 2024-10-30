import math
import numpy as np
from shapely.geometry import Polygon, Point
import random
import copy

import gridmap
from taskplan_multi.environments.sampling import generate_restaurant, load_movables

INFLATE_UB = 0.25
INFLATE_LB = 0.2


def load_restaurant(seed):
    """
    Keep the name of the containers in ascending order
    that you want to place in the corners
    Also keep 'agent' in kitchen container list
    """
    kitchen_containers_list = []
    temp = ['countertop','dishwasher', 'cabinet']
    random.shuffle(kitchen_containers_list)
    agents = ['agent_tall', 'agent_tiny']
    temp.extend(agents)
    kitchen_containers_list.extend(temp)
    serving_room_containers_list = ['servingtable1']
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



class RESTAURANT:
    def __init__(self, seed):
        self.seed = seed
        self.restaurant = load_restaurant(self.seed)
        self.rooms = self.restaurant['rooms']
        self.doors = self.restaurant['doors']
        self.agent_tall = self.restaurant['agent_tall']
        self.agent_tiny = self.restaurant['agent_tiny']
        self.containers = self.restaurant['objects']
        self.init_containers = copy.deepcopy(self.containers)
        self.grid, self.grid_min_x, self.grid_min_z, self.grid_max_x, \
            self.grid_max_z, self.grid_res = self.set_occupancy_grid()
        self.tall_agent_at = 'init_tall'
        self.tiny_agent_at = 'init_tiny'
        inflation_distance = INFLATE_UB
        relative_loc = {}
        self.initial_object_state = list()
        # print(self.restaurant)
        relative_loc['init_tall'] = (self.agent_tall['position']['x'], self.agent_tall['position']['z'])
        relative_loc['init_tiny'] = (self.agent_tiny['position']['x'], self.agent_tiny['position']['z'])
        self.known_cost = {}
        for container in self.containers:
            children = container.get('children')
            for child in children:
                self.initial_object_state.append(child)
            cont_ploy = Polygon([(point['x'], point['z'])
                                 for point in container['polygon']])
            mother_poly = Polygon([(point['x'], point['z'])
                                   for point in self.rooms['servingroom']['polygon']])
            if container['loc'] == 'kitchen':
                mother_poly = Polygon([(point['x'], point['z'])
                                       for point in self.rooms['kitchen']['polygon']])
            point_cloud = get_unoccupied_points_around_container(
                                            self.grid,
                                            self.grid_min_x, self.grid_min_z,
                                            self.grid_res,
                                            cont_ploy,
                                            inflation_distance,
                                            mother_poly
                                        )
            while len(point_cloud) == 0:
                inflation_distance += 0.05
                point_cloud = get_unoccupied_points_around_container(
                                            self.grid,
                                            self.grid_min_x, self.grid_min_z,
                                            self.grid_res,
                                            cont_ploy,
                                            inflation_distance,
                                            mother_poly
                                        )
            inflation_distance = INFLATE_UB
            relative_loc[container['assetId']] = point_cloud
        self.accessible_poses = {}
        for item1 in relative_loc:
            point1 = relative_loc[item1]
            s_x, s_z = world_to_grid(point1[0], point1[1],
                                     self.grid_min_x,
                                     self.grid_min_z, self.grid_res)
            self.accessible_poses[item1] = (s_x, s_z)
            cost_grid = gridmap.planning.compute_cost_grid_from_position(
                    self.grid,
                    start=[
                        s_x,
                        s_z
                    ],
                    only_return_cost_grid=True)
            for item2 in relative_loc:
                # if item2 in self.known_cost and item1 in self.known_cost[item2]:
                #     continue
                point2 = relative_loc[item2]
                e_x, e_z = world_to_grid(point2[0], point2[1],
                                         self.grid_min_x,
                                         self.grid_min_z, self.grid_res)
                cost = cost_grid[e_x, e_z]
                if item1 not in self.known_cost:
                    self.known_cost[item1] = {}
                self.known_cost[item1][item2] = cost
        # print(self.initial_object_poses)
        # print(self.known_cost)
        # for item in self.known_cost:
        #     print(item)
        #     print(self.known_cost[item])
        # raise NotImplementedError

    def set_occupancy_grid(self):
        # Convert to Shapely Polygons
        kitchen = self.rooms['kitchen']['polygon']
        serving_room = self.rooms['servingroom']['polygon']
        kitchen_polygon = Polygon([(point['x'], point['z'])
                                   for point in kitchen])
        serving_room_polygon = Polygon([(point['x'], point['z'])
                                        for point in serving_room])
        door = self.restaurant['doors']['door1']['position']

        # Define the resolution of the grid (e.g., each cell is 0.1 units)
        resolution = 0.1

        # Merge the two room points
        all_points = kitchen + serving_room

        # Extract x and z coordinates
        x_coords = [point['x'] for point in all_points]
        z_coords = [point['z'] for point in all_points]

        # Determine the bounds of the grid
        min_x, max_x = min(x_coords), max(x_coords)
        min_z, max_z = min(z_coords), max(z_coords)

        # Calculate grid dimensions
        grid_height = int((max_x - min_x) / resolution) + 1
        grid_width = int((max_z - min_z) / resolution) + 1

        # Initialize the occupancy grid (0 for free space, 1 for occupied)
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=int)

        # Draw the kitchen and serving room polygons on the grid
        draw_polygon_on_grid(kitchen_polygon, occupancy_grid, 1,
                             min_x, min_z, resolution)
        draw_polygon_on_grid(serving_room_polygon, occupancy_grid, 1,
                             min_x, min_z, resolution)

        # Draw the door
        door_start_grid_x, door_start_grid_z = world_to_grid(
            door[0]['x'], door[0]['z'], min_x, min_z, resolution)
        door_end_grid_x, door_end_grid_z = world_to_grid(
            door[1]['x'], door[1]['z'], min_x, min_z, resolution)
        draw_line(occupancy_grid, door_start_grid_x, door_start_grid_z,
                  door_end_grid_x, door_end_grid_z, 0)

        # Draw the containers (for now points)
        containers = self.restaurant['objects']
        for container in containers:
            rectangle = Polygon([(point['x'], point['z'])
                                 for point in container['polygon']])
            update_occupancy_grid_with_rectangles(occupancy_grid, rectangle,
                                                  min_x, min_z, resolution, 1)

        return occupancy_grid, min_x, min_z, max_x, max_z, resolution

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
        self.tall_agent_at = 'init_tall'
        self.tiny_agent_at = 'init_tiny'

    def get_objects_by_container_name(self, name):
        if name == 'initial_robot_pose':
            return []
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
        for objct in state[0]:
            if objct['assetId'] == name:
                return copy.deepcopy(objct)
        return None
    

    def get_final_state_from_plan(self, plan):
        objects = self.get_current_object_state()
        locations_dict = dict(self.get_container_pos_list())
        conditions = []
        for p in plan:
            if "place" in p.name:
                obj = p.args[1]
                cnt = p.args[2]
                placed = (obj, cnt)
                conditions.append(placed)
            # if "move" in p.name:
            #     rob_at = p.args[1]

        for obj_dict in objects:
            for cond in conditions:  # Directly check if the condition's object key is in the dictionary
                if cond[0] == obj_dict['assetId'] and cond[1] in locations_dict:
                    obj_dict['position'] = locations_dict[cond[1]]
        return objects
