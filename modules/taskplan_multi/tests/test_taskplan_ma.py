import torch
import pandas as pd
import matplotlib.pyplot as plt
import taskplan_multi
import numpy as np
import os
import learning


def test_ma_inspect_data():
    root = "/data/cook-agent/"
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    true_costs = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = root+pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            true_costs.append(x['label'])
    plt.clf()
    plt.scatter(true_costs, true_costs, alpha=0.1)
    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(true_costs))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # Labeling the axes
    plt.xlabel('True Costs')
    plt.ylabel('True Costs')
    plt.title('Costs Scatter Plot with Line from Origin (On Training)')
    save_file = '/data/figs/data-viz-cost-cook.png'
    plt.savefig(save_file, dpi=600)

def test_ma_model_output():
    root = "/data/server-agent/"
    net_name = 'ap_server.pt'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
        network_file= root + 'logs/beta-v0/' + net_name,
        device=device
    )
    json_files = list()
    for path, _, files in os.walk(root):
        for name in files:
            if 'data_training_' in name and ".csv" in name:
                json_files.append(os.path.join(path, name))
    true_costs = list()
    exp_cost = list()
    for file in json_files:
        df = pd.read_csv(file, header=None)
        for idx, pickle_path in enumerate(df[0]):
            pickle_path = root + pickle_path
            x = learning.data.load_compressed_pickle(pickle_path)
            true_costs.append(x['label'])
            anticipated_cost = eval_net(x)
            exp_cost.append(anticipated_cost)
    plt.clf()
    plt.scatter(true_costs, exp_cost, alpha=0.1)
    # Draw a line from the origin to the farthest point
    max_value = max(max(true_costs), max(exp_cost))
    plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

    # Labeling the axes
    plt.xlabel('True Costs')
    plt.ylabel('Learned Costs')
    plt.title('Costs Scatter Plot with Line from Origin (On Training)')
    save_file = '/data/figs/' + net_name + '-compare-cost.png'
    plt.savefig(save_file, dpi=600)



# def test_ma_model_output_tiny():
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     net_name = 'ap_tiny.pt'
#     eval_net = taskplan_multi.models.gcn.AnticipateGCN.get_net_eval_fn(
#         network_file='/data/restaurant-multi-tiny/logs/beta-v0/' + net_name,
#         device=device
#     )
#     root = "/data/restaurant-multi-tiny/"
#     json_files = list()
#     for path, _, files in os.walk(root):
#         for name in files:
#             if 'data_training_' in name and ".csv" in name:
#                 json_files.append(os.path.join(path, name))
#     true_costs = list()
#     exp_cost = list()
#     for file in json_files:
#         df = pd.read_csv(file, header=None)
#         for idx, pickle_path in enumerate(df[0]):
#             pickle_path = "/data/restaurant-multi-tiny/"+pickle_path
#             x = learning.data.load_compressed_pickle(pickle_path)
#             true_costs.append(x['label'])
#             anticipated_cost = eval_net(x)
#             exp_cost.append(anticipated_cost)
#     plt.clf()
#     plt.scatter(true_costs, exp_cost, alpha=0.1)
#     # Draw a line from the origin to the farthest point
#     max_value = max(max(true_costs), max(exp_cost))
#     plt.plot([0, max_value], [0, max_value], 'grey')  # 'r' makes the line red

#     # Labeling the axes
#     plt.xlabel('True Costs')
#     plt.ylabel('Learned Costs')
#     plt.title('Costs Scatter Plot with Line from Origin (On Training)')
#     save_file = '/data/figs/' + net_name + 'tiny-compare-cost.png'
#     plt.savefig(save_file, dpi=600)