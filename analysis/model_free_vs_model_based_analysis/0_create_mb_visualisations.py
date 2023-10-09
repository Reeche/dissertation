import networkx as nx
import matplotlib.pyplot as plt
from mcl_toolbox.utils.experiment_utils import Experiment
import os

G = nx.DiGraph()

structure = {
    "layout": {
        "0": [0, 0],
        "1": [0, 1],
        "2": [0, 2],
        "3": [1, 2],
        "4": [-1, 2],
        "5": [1, 0],
        "6": [2, 0],
        "7": [2, -1],
        "8": [2, 1],
        "9": [-1, 0],
        "10": [-2, 0],
        "11": [-2, -1],
        "12": [-2, 1],
    },
    "initial": "0",
    "graph": {
        "0": {"up": [0, "1"], "right": [0, "5"], "left": [0, "9"]},
        "1": {"up": [0, "2"]},
        "2": {"right": [0, "3"], "left": [0, "4"]},
        "3": {},
        "4": {},
        "5": {"right": [0, "6"]},
        "6": {"up": [0, "7"], "down": [0, "8"]},
        "7": {},
        "8": {},
        "9": {"left": [0, "10"]},
        "10": {"up": [0, "11"], "down": [0, "12"]},
        "11": {},
        "12": {},
    },
}

exp_name = "mb_vs_mf_mf_v0"
E = Experiment(exp_name, data_path=f"../../data/human/{exp_name}")
participants = [1, 3]
number_of_trials = 32


for index, pid in enumerate(participants):
    os.makedirs(f"visualisation/{index}", exist_ok=True)
    for trial in range(2, number_of_trials):
        p = E.participants[pid]

        # get the clicks and envs seen by the participant
        deleted = p.clicks[trial].pop() #remove last action which is 0
        clicks = p.clicks[trial]
        values = p.envs[trial]

        # Add nodes with positions
        for node, pos in structure["layout"].items():
            G.add_node(node, pos=pos, value=values[int(node)])

        # Add edges with corresponding directions
        for node, directions in structure["graph"].items():
            for direction, target in directions.items():
                G.add_edge(node, target[1], direction=direction)

        # Draw the graph with selected values
        pos = nx.get_node_attributes(G, 'pos')
        node_values = nx.get_node_attributes(G, 'value')

        # Color nodes based on value
        font_colors = ['green' if v >= 0 else 'red' for v in node_values.values()]

        # Draw nodes with labels
        nx.draw_networkx_nodes(G, pos, node_size=1400, node_color='lightgrey')
        for node, color in zip(G.nodes(), font_colors):
            if int(node) in clicks:
                nx.draw_networkx_labels(G, pos, labels={node: node_values[node]}, font_size=16, font_weight='bold',
                                        font_color=color)

        # Draw edges
        nx.draw_networkx_edges(G, pos)

        plt.axis('off')
        # plt.show()
        plt.savefig(f"visualisation/{index}/{trial}.png")
        plt.close()
