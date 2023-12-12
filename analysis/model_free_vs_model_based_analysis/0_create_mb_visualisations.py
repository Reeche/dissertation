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
participants = [3, 5, 7, 8, 9, 10, 12, 13, 15, 19, 23, 25, 28, 30, 32, 33, 34, 36, 37, 40, 41, 45, 46, 48, 52, 54, 56,
                58, 59, 62, 63, 66, 68, 69, 72, 74, 76, 78, 82, 84, 86, 89, 90, 91, 93, 94, 96, 98, 100, 102, 104, 106,
                108, 111, 113, 114, 115, 116, 121, 124, 125, 126, 127, 129, 130, 132, 133, 134, 137, 138, 139, 141,
                145, 146, 148, 149, 152, 156, 158, 159, 163, 167, 168, 172, 173, 175, 176, 179, 180, 182, 184, 186, 187,
                189, 190, 191]

#82 particiapnts who clicked anything
mf_clicked = [3, 5, 9, 10, 13, 15, 23, 25, 28, 30, 32, 33, 36, 37, 41, 45, 46, 52, 56, 58, 59, 62, 63, 66, 68,
                        69, 72, 74, 76, 78, 82, 84, 86, 89, 91, 93, 94, 96, 98, 100, 102, 104, 108, 111, 115, 116, 124,
                        125, 126, 127, 129, 130, 132, 134, 137, 138, 139, 141, 145, 146, 148, 149, 152, 156, 158, 159,
                        163, 167, 168, 172, 173, 175, 176, 179, 180, 182, 184, 186, 187, 189, 190, 191]

# participants = [141]
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
        # plt.savefig(f"visualisation/{i}/{trial}.png")
        plt.close()

    i += 1

print(good_pid)
print(len(good_pid))
