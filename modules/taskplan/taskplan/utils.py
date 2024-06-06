import io
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from sentence_transformers import SentenceTransformer

from taskplan.environments.restaurant import world_to_grid


def get_robot_pose(data):
    rob_x = data.agent['position']['x']
    rob_z = data.agent['position']['z']
    x, z = world_to_grid(
        rob_x, rob_z, data.grid_min_x, data.grid_min_z, data.grid_res)
    return (x, z)


def get_graph(data):
    ''' This method creates graph data from the procthor-10k data'''
    # Create dummy apartment node
    node_count = 0
    nodes = {}
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
        name = get_generic_name(container['id'])
        _x, _y = world_to_grid(
            container['position']['x'], container['position']['z'],
            data.grid_min_x, data.grid_min_z, data.grid_res)
        src = room_names.index(container['loc'])
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
            name = get_generic_name(connected_object['id'])
            _x, _y = world_to_grid(
                connected_object['position']['x'],
                connected_object['position']['z'],
                data.grid_min_x, data.grid_min_z, data.grid_res)
            src = container_ids.index(container['id'])
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
        'obj_node_idx': obj_node_idx  # indices of objects
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
