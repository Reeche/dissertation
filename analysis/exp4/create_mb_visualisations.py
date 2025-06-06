import os
import networkx as nx
import matplotlib.pyplot as plt
from mcl_toolbox.utils.experiment_utils import Experiment


def build_graph(layout: dict, graph_structure: dict, values: list) -> nx.DiGraph:
    """
    Constructs a directed graph based on a given layout, structure, and node values.

    Args:
        layout (dict): Mapping of node IDs to (x, y) positions.
        graph_structure (dict): Adjacency and directional mapping.
        values (list): Values associated with each node.

    Returns:
        nx.DiGraph: A fully constructed graph with positions and node values.
    """
    G = nx.DiGraph()

    # Add nodes with positions and values
    for node, pos in layout.items():
        G.add_node(node, pos=pos, value=values[int(node)])

    # Add directional edges
    for node, directions in graph_structure.items():
        for direction, target in directions.items():
            G.add_edge(node, target[1], direction=direction)

    return G


def draw_graph(G: nx.DiGraph, clicks: list, output_path: str):
    """
    Draws the graph and saves it as an image.

    Args:
        G (nx.DiGraph): The graph to be visualized.
        clicks (list): Node indices clicked by the participant.
        output_path (str): File path to save the graph image.
    """
    pos = nx.get_node_attributes(G, 'pos')
    node_values = nx.get_node_attributes(G, 'value')

    # Assign font color based on value
    font_colors = ['green' if v >= 0 else 'red' for v in node_values.values()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1400, node_color='lightgrey')

    # Draw clicked node labels with colors
    for node, color in zip(G.nodes(), font_colors):
        if int(node) in clicks:
            nx.draw_networkx_labels(
                G, pos,
                labels={node: node_values[node]},
                font_size=16, font_weight='bold', font_color=color
            )

    # Draw edges
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()


def process_participants(E: Experiment, structure: dict, clicked_ids: list, num_trials: int = 15):
    """
    Processes participants' data, builds and visualizes the environment per trial.

    Args:
        E (Experiment): The experiment object containing participants' data.
        structure (dict): The environment layout and structure.
        clicked_ids (list): List of participant IDs who clicked.
        num_trials (int): Number of trials to visualize per participant.
    """
    good_pids = []
    counter = 1
    participants = list(E.participants.keys())

    for pid in participants:
        p = E.participants[pid]

        # Skip if all click entries are length 1 (no real clicks)
        if all(len(entry) == 1 for entry in p.clicks):
            continue

        good_pids.append(pid)
        os.makedirs(f"visualisation/{counter}", exist_ok=True)

        # Filter for training trials
        training_trials = [(blk, env) for blk, env in zip(p.block, p.envs) if blk == "training"]

        for trial in range(num_trials):
            # Prepare trial data
            p.clicks[trial].pop()  # Remove last placeholder click
            clicks = p.clicks[trial]
            values = training_trials[trial][1]

            # Build and draw graph
            G = build_graph(structure["layout"], structure["graph"], values)
            output_path = f"visualisation/{counter}/{trial}.png"
            draw_graph(G, clicks, output_path)

        counter += 1

    print(good_pids)
    print(f"Number of valid participants: {len(good_pids)}")


if __name__ == "__main__":
    # Graph and experiment configuration
    structure = {
        "layout": {
            "0": [0, 0], "1": [0, 1], "2": [0, 2], "3": [1, 2], "4": [-1, 2],
            "5": [1, 0], "6": [2, 0], "7": [2, -1], "8": [2, 1],
            "9": [-1, 0], "10": [-2, 0], "11": [-2, -1], "12": [-2, 1],
        },
        "initial": "0",
        "graph": {
            "0": {"up": [0, "1"], "right": [0, "5"], "left": [0, "9"]},
            "1": {"up": [0, "2"]},
            "2": {"right": [0, "3"], "left": [0, "4"]},
            "3": {}, "4": {},
            "5": {"right": [0, "6"]},
            "6": {"up": [0, "7"], "down": [0, "8"]},
            "7": {}, "8": {},
            "9": {"left": [0, "10"]},
            "10": {"up": [0, "11"], "down": [0, "12"]},
            "11": {}, "12": {}
        },
    }

    exp_name = "mf"
    experiment = Experiment(exp_name, data_path=f"../../data/human/{exp_name}")

    # IDs of participants who clicked
    mf_clicked_ids = [
        1, 2, 3, 4, 7, 10, 11, 12, 16, 19, 20, 21, 22, 23, 24, 25, 26, 29, 36, 42, 43,
        45, 46, 47, 49, 51, 52, 55, 57, 58, 61, 66, 71, 74, 76, 77, 81, 82, 84, 85, 87,
        90, 91, 92, 93, 94, 96, 97, 98, 104, 106, 108, 109, 111, 114, 116, 118, 121,
        123, 126, 128, 130, 134, 139, 140, 142, 144, 145, 149, 150, 152, 153, 155,
        157, 158, 162, 163, 165, 166, 168, 170, 171, 172, 175, 178, 179, 181, 182,
        183, 184, 185, 187, 192, 193, 194, 200, 201, 202, 203, 204, 207, 210, 213,
        218, 220, 221, 222, 223, 224, 229, 233, 237, 238, 240, 242, 244, 245, 246,
        252, 255, 258
    ]

    process_participants(experiment, structure, mf_clicked_ids)
