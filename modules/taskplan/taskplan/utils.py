import io
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from sentence_transformers import SentenceTransformer
import learning
import gridmap
from taskplan.environments.restaurant import world_to_grid
import torch
from torch_geometric.data import Data
import re
from collections import defaultdict


def get_robot_pose(data):
    return data.accessible_poses['initial_robot_pose']

# 'attribs:': [isLiquid, pickable, spreadable, spread, fillable, dirty, container, jar, slicable, filled]


def parse_symbolic_state(atom_lines):
    '''
    This function parses raw text lines of symbolic atoms into a structured dictionary.
    This state represents the "current" state of the environment, which can be
    different from the initial state described in the main data object.
    '''
    # Predicates that are not relevant for the graph structure are ignored.
    ignored = {"is-holding", "is-located", "hand-is-free", "restrict-move-to"}
    state = defaultdict(dict)

    for line in atom_lines:
        line = line.strip()
        if not line.startswith("Atom "):
            continue
        # Use regex to capture the predicate and its arguments
        match = re.match(r"Atom ([^(]+)\(([^)]*)\)", line)
        if not match:
            continue
        
        pred, args_str = match.groups()
        if pred in ignored:
            continue
        
        args = tuple(arg.strip() for arg in args_str.split(","))
        
        # Store the predicate as true for the given arguments
        state[pred][args] = True
            
    return dict(state)


def get_graph(data, state={}):
    '''
    This method creates graph data from the restaurant data.
    If a 'state' dictionary is provided, it updates the graph to reflect the
    current state of object locations and attributes.
    '''
    node_count = 0
    nodes = {}
    edges = []
    # This map is crucial for finding node indices from object/container IDs
    id_to_idx_map = {}
    assetId_idx_map = {}
    
    # 1. Create dummy restaurant node
    restaurant_id = 'restaurant'
    nodes[node_count] = {
        'id': restaurant_id,
        'name': 'restaurant',
        'desc': 'Restaurant',
        'pos': (0, 0),
        'type': [1, 0, 0, 0, 0],
        'attribs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    id_to_idx_map[restaurant_id] = node_count
    node_count += 1

    # 2. Create robot node and add its edge to the restaurant
    _x, _y = data.accessible_poses[data.rob_at]
    robot_id = 'robot'
    nodes[node_count] = {
        'id': robot_id,
        'name': 'robot',
        'desc': 'Robot',
        'pos': (_x, _y),
        'type': [0, 1, 0, 0, 0],
        'attribs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    id_to_idx_map[robot_id] = node_count
    node_count += 1
    
    rob_loc_room_name = 'kitchen'
    if 'servingtable' in data.rob_at:
        rob_loc_room_name = 'servingroom'

    # 3. Create room nodes
    for room in data.rooms:
        _x, _y = world_to_grid(
            data.rooms[room]['position']['x'], data.rooms[room]['position']['z'],
            data.grid_min_x, data.grid_min_z, data.grid_res)
        
        room_id = room + '|' + str(node_count)
        room_name = data.rooms[room]['name'].lower()
        nodes[node_count] = {
            'id': room_id,
            'name': room_name,
            'desc': room_name,
            'pos': (_x, _y),
            'type': [0, 0, 1, 0, 0],
            'attribs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        id_to_idx_map[room_id] = node_count
        id_to_idx_map[room_name] = node_count # Also map by simple name for convenience
        
        # Connect room to the main restaurant node
        edges.append(tuple([id_to_idx_map['restaurant'], node_count]))
        
        # Connect robot to the room it's in
        if room_name == rob_loc_room_name:
            edges.append(tuple([id_to_idx_map['robot'], node_count]))
            
        node_count += 1

    # Add an edge between two rooms adjacent by a passable shared door
    # This assumes kitchen is node 2 and servingroom is node 3
    edges.append(tuple([2, 3]))

    # 4. Create container nodes
    cnt_node_idx = []
    for container in data.containers:
        container_id = container['id']
        assetId = container['assetId']
        name = get_generic_name(container['id'])
        _x, _y = data.accessible_poses[assetId]
        
        # Determine container attributes from `data` as default
        is_fillable = 1 if 'fillable' in container and container['fillable'] else 0
        is_filled = 1 if 'filled' in container and container['filled'] else 0
        
        # If state is provided, override attributes
        if state:
            if state.get('is-filled', {}).get((container_id,)):
                is_filled = 1
            elif state.get('is-empty', {}).get((container_id,)):
                is_filled = 0
        
        attribs = [0, 0, 0, 0, is_fillable, 0, 0, 0, 0, is_filled]
        
        nodes[node_count] = {
            'id': container_id,
            'name': name,
            'desc': container['description'],
            'pos': (_x, _y),
            'type': [0, 0, 0, 1, 0],
            'attribs': attribs
        }
        # The container is located in a room
        src_room_idx = id_to_idx_map[container['loc']]
        edges.append(tuple([src_room_idx, node_count]))

        id_to_idx_map[container_id] = node_count
        assetId_idx_map[assetId] = node_count
        cnt_node_idx.append(node_count)
        node_count += 1

    # 5. Create object (children) nodes
    obj_node_idx = []
    for container in data.containers:
        for connected_object in container['children']:
            obj_id = connected_object['id']
            assetId = connected_object['assetId']
            name = get_generic_name(connected_object['id'])
            
            # --- Determine Object Location ---
            # Default location is the parent container from initial data
            src_id = container['id']
            # If state is provided, check for an `is-on` predicate to find new location
            if state:
                is_on_state = state.get('is-on', {})
                for (arg1, arg2), _ in is_on_state.items():
                    if arg1 == obj_id:
                        src_id = arg2 # Update source to the new parent's ID
                        break
            
            src_node_idx = id_to_idx_map[src_id]

            # --- Determine Object Attributes ---
            # **FIXED CODE BLOCK:** Initialize attributes and set them one by one.
            # This avoids the error-causing '.get()' method on `connected_object`.
            attribs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if 'isLiquid' in connected_object and connected_object['isLiquid'] == 1:
                attribs[0] = 1
            if 'pickable' in connected_object and connected_object['pickable'] == 1:
                attribs[1] = 1
            if 'spreadable' in connected_object and connected_object['spreadable'] == 1:
                attribs[2] = 1
            if 'spread' in connected_object and connected_object['spread'] == 1:
                attribs[3] = 1
            if 'fillable' in connected_object and connected_object['fillable'] == 1:
                attribs[4] = 1
            if 'dirty' in connected_object and connected_object['dirty'] == 1:
                attribs[5] = 1
            if 'container' in connected_object and connected_object['container'] == 1:
                attribs[6] = 1
            if 'jar' in connected_object and connected_object['jar'] == 1:
                attribs[7] = 1
            if 'slicable' in connected_object and connected_object['slicable'] == 1:
                attribs[8] = 1
            if 'filled' in connected_object and connected_object['filled'] == 1:
                attribs[9] = 1
            
            # If state is provided, override attributes with current values
            if state:
                # is-dirty / is-clean
                if state.get('is-dirty', {}).get((obj_id,)):
                    attribs[5] = 1
                elif state.get('is-clean', {}).get((obj_id,)):
                    attribs[5] = 0
                # is-filled / is-empty
                if state.get('is-filled', {}).get((obj_id,)):
                    attribs[9] = 1
                elif state.get('is-empty', {}).get((obj_id,)):
                    attribs[9] = 0
                # is-spread
                if state.get('is-spread', {}).get((obj_id,)):
                    attribs[3] = 1
            
            # Object position is its parent container's initial position
            _x, _y = world_to_grid(
                container['position']['x'], container['position']['z'],
                data.grid_min_x, data.grid_min_z, data.grid_res)
            
            nodes[node_count] = {
                'id': obj_id,
                'name': name,
                'desc': connected_object['description'],
                'pos': (_x, _y),
                'type': [0, 0, 0, 0, 1],
                'attribs': attribs
            }
            edges.append(tuple([src_node_idx, node_count]))
            
            id_to_idx_map[obj_id] = node_count
            assetId_idx_map[assetId] = node_count
            obj_node_idx.append(node_count)
            node_count += 1
            
    graph = {
        'nodes': nodes,
        'edge_index': edges,
        'cnt_node_idx': cnt_node_idx,
        'obj_node_idx': obj_node_idx,
        'idx_map': assetId_idx_map,
        'id_to_idx_map': id_to_idx_map, # Useful for debugging
        'distances': data.known_cost
    }

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
            graph['nodes'][node_key]['type'],
            graph['nodes'][node_key]['pos'],
            graph['nodes'][node_key]['attribs'],
        ))
        assert count == node_key
        graph_nodes.append(node_feature)
        node_color_list.append(get_object_color_from_type(graph['nodes'][node_key]))

    graph['node_coords'] = node_coords
    graph['node_names'] = node_names
    graph['graph_nodes'] = graph_nodes  # node features
    src = []
    dst = []
    new_feature = []
    for edge in graph['edge_index']:
        src.append(edge[0])
        dst.append(edge[1])
        e_cost = np.linalg.norm(np.array(
            [graph['nodes'][edge[0]]['pos']]) - np.array(
                                            [graph['nodes'][edge[1]]['pos']]))
        new_feature.append([e_cost])

    graph['graph_edge_feature'] = 1 - np.array(new_feature)/600
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


def get_object_color_from_type(node):
    encoding = node['type']
    if encoding[0] == 1:
        return "black"
    if encoding[1] == 1:
        return "gray"
    if encoding[2] == 1:
        return "blue"
    if encoding[3] == 1:
        return "green"
    if encoding[4] == 1:
        if node['attribs'][0] == 1:
            return "orange"
        if node['attribs'][5] == 1:
            return "red"
        if node['attribs'][9] == 1:
            return "yellow"
        return "violet"
    return "pink"


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
    count = 0
    for action in plan:
        if action.name == 'move':
            count += 1
            container_name = action.args[1]
            container_pose = get_container_pose(container_name, partial_map)
            robot_poses.append(container_pose)
        elif action.name == 'find':
            split_at = count - 1
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


def write_datum_to_file(args, datum, counter):
    data_filename = os.path.join(
        'pickles', f'dat_{args.current_seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(args.save_dir, data_filename), datum)
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.csv'
    with open(os.path.join(args.save_dir, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')


def preprocess_training_data(args=None):
    def make_graph(data):
        data = graph_formatting(data)
        data['node_feats'] = torch.tensor(
            np.array(data['graph_nodes']), dtype=torch.float)
        data['edge_index'] = torch.tensor(data['graph_edge_index'],
                                          dtype=torch.long)
        data['edge_features'] = torch.tensor(
            data['graph_edge_feature'], dtype=torch.float)
        src = data['edge_index'][0]
        dest = data['edge_index'][1]
        features = data['edge_features']

        # Create reversed edges
        rev_src = dest
        rev_dest = src

        # Efficiently concatenate using torch.cat for both indices and features
        data['edge_index'] = torch.cat((torch.stack([src, dest], dim=0),
                                        torch.stack([rev_src, rev_dest],
                                        dim=0)), dim=1)
        data['edge_features'] = torch.cat((features, features), dim=0)
        # print(data['edge_features'])
        # edge_features = torch.tensor(1-np.array(data['edge_features'])/600,
        #                              dtype=torch.float)
        # print(edge_features)
        # raise NotImplementedError
        data['label'] = torch.tensor(data['label'], dtype=torch.float)
        tg_GCN_format = Data(x=data['node_feats'],
                             edge_index=data['edge_index'],
                             edge_features=data['edge_features'],
                             y=data['label'])

        result = tg_GCN_format
        return result
    return make_graph


def preprocess_gcn_data(datum):
    data = graph_formatting(datum)
    data['edge_index'] = torch.tensor(
        data['graph_edge_index'], dtype=torch.long)
    data['edge_features'] = torch.tensor(
            data['graph_edge_feature'], dtype=torch.float)
    src = data['edge_index'][0]
    dest = data['edge_index'][1]
    features = data['edge_features']
    rev_src = dest
    rev_dest = src
    data['edge_data'] = torch.cat((
        torch.stack([src, dest], dim=0), torch.stack(
            [rev_src, rev_dest], dim=0)), dim=1)
    # data['edge_data'] = torch.tensor(data['graph_edge_index'], dtype=torch.long)
    data['edge_features'] = torch.cat((features, features), dim=0)
    data['latent_features'] = torch.tensor(np.array(
        data['graph_nodes']), dtype=torch.float)
    return data
