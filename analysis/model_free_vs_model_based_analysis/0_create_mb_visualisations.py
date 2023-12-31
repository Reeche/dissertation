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

exp_name = "mf"
E = Experiment(exp_name, data_path=f"../../data/human/{exp_name}")
# get all participants
participants = list(E.participants.keys())

# MF participants who clicked: 121 participants
mf_clicked = [1, 2, 3, 4, 7, 10, 11, 12, 16, 19, 20, 21, 22, 23, 24, 25, 26, 29, 36, 42, 43, 45, 46, 47, 49, 51, 52, 55,
              57, 58, 61, 66, 71, 74, 76, 77, 81, 82, 84, 85, 87, 90, 91, 92, 93, 94, 96, 97, 98, 104, 106, 108, 109,
              111, 114, 116, 118, 121, 123, 126, 128, 130, 134, 139, 140, 142, 144, 145, 149, 150, 152, 153, 155, 157,
              158, 162, 163, 165, 166, 168, 170, 171, 172, 175, 178, 179, 181, 182, 183, 184, 185, 187, 192, 193, 194,
              200, 201, 202, 203, 204, 207, 210, 213, 218, 220, 221, 222, 223, 224, 229, 233, 237, 238, 240, 242, 244,
              245, 246, 252, 255, 258]

number_of_trials = 15  # need to offset by 2 #todo: actually only need the first 15 trials of the MF condition
good_pid = []
i = 1
for index, pid in enumerate(participants):
    p = E.participants[pid]
    # check if the len of every entry of the list is 0, if yes, skip this participant
    if all(len(entry) == 1 for entry in p.clicks):
        continue

    good_pid.append(pid)
    os.makedirs(f"visualisation/{i}", exist_ok=True)

    # filter for practice trials
    trials = list(zip(p.block, p.envs))
    training_trials = [item for item in trials if item[0] == "training"]

    for trial in range(0, number_of_trials):
        # get the clicks and envs seen by the participant
        deleted = p.clicks[trial].pop()  # remove last action which is 0
        clicks = p.clicks[trial]

        values = training_trials[trial][1]

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
        plt.savefig(f"visualisation/{i}/{trial}.png")
        plt.close()

    i += 1

print(good_pid)
print(len(good_pid))
