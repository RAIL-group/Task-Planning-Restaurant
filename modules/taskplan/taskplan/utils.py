import io
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from sentence_transformers import SentenceTransformer

import gridmap
from taskplan.environments.restaurant import world_to_grid


def get_robot_pose(data):
    return data.accessible_poses['initial_robot_pose']


def get_graph(data):
    ''' This method creates graph data from the procthor-10k data'''
    # Create dummy apartment node
    node_count = 0
    nodes = {}
    assetId_idx_map = {}
    edges = []
    nodes[node_count] = {
        'id': 'apartment|0',
        'name': 'apartment',
        'pos': (0, 0),
        'type': [1, 0, 0, 0]
    }
    node_count += 1

    # Iterate over rooms but skip position coordinate scaling since not
    # required in distance calculations
    for room in data.rooms:
        _x, _y = world_to_grid(
            data.rooms[room]['position']['x'], data.rooms[room]['position']['z'],
            data.grid_min_x, data.grid_min_z, data.grid_res)
        nodes[node_count] = {
            'id': room+'|'+str(node_count),
            'name': data.rooms[room]['name'].lower(),
            'pos': (_x, _y),
            'type': [0, 1, 0, 0]
        }
        edges.append(tuple([0, node_count]))
        node_count += 1

    # add an edge between two rooms adjacent by a passable shared door
    room_edges = set([(1, 2)])
    edges.extend(room_edges)

    room_names = [nodes[n]['name'] for n in nodes]
    cnt_node_idx = []

    for container in data.containers:
        id = container['id']
        assetId = container['assetId']
        name = get_generic_name(container['id'])
        _x, _y = data.accessible_poses[assetId]
        src = room_names.index(container['loc'])
        assetId_idx_map[assetId] = node_count
        nodes[node_count] = {
            'id': id,
            'name': name,
            'pos': (_x, _y),
            'type': [0, 0, 1, 0]
        }
        edges.append(tuple([src, node_count]))
        cnt_node_idx.append(node_count)
        node_count += 1

    container_ids = [nodes[n]['id'] for n in nodes]
    obj_node_idx = []

    for container in data.containers:
        for connected_object in container['children']:
            id = connected_object['id']
            assetId = connected_object['assetId']
            name = get_generic_name(connected_object['id'])
            _x, _y = world_to_grid(
                container['position']['x'],
                container['position']['z'],
                data.grid_min_x, data.grid_min_z, data.grid_res)
            src = container_ids.index(container['id'])
            assetId_idx_map[assetId] = node_count
            nodes[node_count] = {
                    'id': id,
                    'name': name,
                    'pos': (_x, _y),
                    'type': [0, 0, 0, 1]
                }
            edges.append(tuple([src, node_count]))
            obj_node_idx.append(node_count)
            node_count += 1

    graph = {
        'nodes': nodes,  # dictionary {id, name, pos, type}
        'edge_index': edges,  # pairwise edge list
        'cnt_node_idx': cnt_node_idx,  # indices of contianers
        'obj_node_idx': obj_node_idx,  # indices of objects
        'idx_map': assetId_idx_map,  # mapping from assedId to graph index position
        'distances': data.known_cost  # mapped using assedId-assetId
    }

    # # Add edges to get a connected graph if not already connected
    # req_edges = get_edges_for_connected_graph(proc_data.occupancy_grid, graph)
    # graph['edge_index'] = graph['edge_index'] + req_edges

    return graph


def graph_formatting(graph):
    ''' This method formats the graph data from procthor-10k data
    to be used in PartialMap that maintains graph during object search
    '''
    node_coords = {}
    node_names = {}
    graph_nodes = []
    node_color_list = []

    for count, node_key in enumerate(graph['nodes']):
        node_coords[node_key] = graph['nodes'][node_key]['pos']
        node_names[node_key] = graph['nodes'][node_key]['name']
        node_feature = np.concatenate((
            get_sentence_embedding(graph['nodes'][node_key]['name']),
            graph['nodes'][node_key]['type']
        ))
        assert count == node_key
        graph_nodes.append(node_feature)
        node_color_list.append(get_object_color_from_type(
            graph['nodes'][node_key]['type']))

    graph['node_coords'] = node_coords
    graph['node_names'] = node_names
    graph['graph_nodes'] = graph_nodes  # node features
    src = []
    dst = []
    for edge in graph['edge_index']:
        src.append(edge[0])
        dst.append(edge[1])
    graph['graph_edge_index'] = [src, dst]

    graph['graph_image'] = get_graph_image(
        graph['edge_index'],
        node_names, node_color_list
    )

    return graph


def get_generic_name(name):
    return name.split('|')[0].lower()


def load_sentence_embedding(target_file_name):
    target_dir = '/data/sentence_transformers/cache/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Walk through all directories and files in target_dir
    for root, dirs, files in os.walk(target_dir):
        if target_file_name in files:
            file_path = os.path.join(root, target_file_name)
            if os.path.exists(file_path):
                return np.load(file_path)
    return None


def get_sentence_embedding(sentence):
    loaded_embedding = load_sentence_embedding(sentence + '.npy')
    if loaded_embedding is None:
        model_path = "/data/sentence_transformers/"
        model = SentenceTransformer(model_path)
        sentence_embedding = model.encode([sentence])[0]
        file_name = '/data/sentence_transformers/cache/' + sentence + '.npy'
        np.save(file_name, sentence_embedding)
        return sentence_embedding
    else:
        return loaded_embedding


def get_graph_image(edge_index, node_names, color_map):
    # Create a graph object
    G = nx.Graph()

    # Add nodes to the graph with labels
    for idx, _ in enumerate(node_names):
        G.add_node(idx)

    # Add edges to the graph
    G.add_edges_from(edge_index)

    # Draw the graph
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=150,
            labels=node_names, font_size=8, font_weight='regular', edge_color='black')

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    img = Image.open(buf)
    return img


def get_object_color_from_type(encoding):
    if encoding[0] == 1:
        return "red"
    if encoding[1] == 1:
        return "blue"
    if encoding[2] == 1:
        return "green"
    if encoding[3] == 1:
        return "orange"
    return "violet"


def get_container_pose(cnt_name, partial_map):
    '''This function takes in a container name and the
    partial map as input to return the container pose on the grid'''
    if cnt_name in partial_map.idx_map:
        return partial_map.container_poses[partial_map.idx_map[cnt_name]]
    raise ValueError('The container could not be located on the grid!')


def get_poses_from_plan(plan, partial_map):
    ''' This function takes input of a plan and the partial map
    and produces the robot_poses along known space
    '''
    robot_poses = []
    split_at = None
    count = -1
    for action in plan:
        if action.name == 'move':
            count += 1
            container_name = action.args[1]
            container_pose = get_container_pose(container_name, partial_map)
            robot_poses.append(container_pose)
        elif action.name == 'find':
            split_at = count
    if split_at is None:
        split_at = len(robot_poses) - 1
    if split_at < 0:
        split_at = 0

    return robot_poses, split_at


def get_object_to_find_from_plan(plan, partial_map):
    '''This function takes in a plan and the partial map as
    input to return the object index to find; limited to finding
    single object for now.'''
    for action in plan:
        if action.name == 'find':
            obj_name = action.args[0]
            if obj_name in partial_map.idx_map:
                return partial_map.idx_map[obj_name]
            raise ValueError('The object could not be found!')


def compute_path_cost(grid, path):
    ''' This function returns the total path and path cost
    given the occupancy grid and the trjectory as poses, the
    robot has visited througout the object search process,
    where poses are stored in grid cell coordinates.'''
    total_cost = 0
    total_path = None
    occ_grid = np.copy(grid)

    for point in path:
        occ_grid[int(point[0]), int(point[1])] = 0

    for idx, point in enumerate(path[:-1]):
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            occ_grid,
            start=point,
            use_soft_cost=True,
            only_return_cost_grid=False)
        next_point = path[idx + 1]

        cost = cost_grid[int(next_point[0]), int(next_point[1])]

        total_cost += cost
        did_plan, robot_path = get_path([next_point[0], next_point[1]],
                                        do_sparsify=False,
                                        do_flip=False)
        if total_path is None:
            total_path = robot_path
        else:
            total_path = np.concatenate((total_path, robot_path), axis=1)

    return total_cost, total_path


def get_pos_from_coord(coords, node_coords):
    coords_list = []
    for node in node_coords:
        coords_list.append(tuple(
            [node_coords[node][0],
             node_coords[node][1]]))
    if coords in coords_list:
        pos = coords_list.index(coords)
        return pos
    return None
