import networkx as nx

# ========================================================================================
#
# These functions are used for various graph representations of the MDP
#
# ========================================================================================


def graph_from_adjacency_list(adjacency_list):
    adjacency_dict = {key: val for key, val in enumerate(adjacency_list)}
    return nx.from_dict_of_lists(adjacency_dict, create_using=nx.DiGraph)


def adjacency_list_from_graph(env_graph):
    adjacency_dict = nx.to_dict_of_lists(env_graph)
    adjacency_list = [adjacency_dict[key] for key in range(len(adjacency_dict))]
    return adjacency_list


def add_property_to_graph(env_graph, property_name, property_dict):
    """
    :param: env_graph, mdp_graph as networkx DiGraph
    :param: property_name, name of property
    :param: property_dict, dict with each node as a key
            and property values as values
    """
    if set(property_dict.keys()) != set(env_graph.nodes):
        raise ValueError("Dictionary keys and graph nodes must match exactly")

    for node, data in env_graph.nodes(data=True):
        data[property_name] = property_dict[node]

    return env_graph


def annotate_mdp_graph(env_graph, additional_graph_properties={}):
    """
    Adds node properties that might be inputs to a cost function
    :param: env_graph, mdp_graph as networkx DiGraph
    :param: additional_graph_properties, dictionary of node attributes
            in the form of {property_name: {1: property_val, ....}}
    :return: mdp graph with properties in kwargs
             and some other basic properties implemented
    """
    if "initial" in additional_graph_properties:
        initial_node = additional_graph_properties.pop("initial")
    else:
        initial_node = 0

    depth_dict = nx.shortest_path_length(env_graph, initial_node)
    for node, data in env_graph.nodes(data=True):
        data["depth"] = depth_dict[node]

    # add additional properties
    for property_name, property_dict in additional_graph_properties.items():
        env_graph = add_property_to_graph(env_graph, property_name, property_dict)

    # create cluster with side / direction or numbered if button not available
    env_graph.nodes[initial_node]["cluster"] = 0
    for cluster_idx, node in enumerate(env_graph.successors(initial_node)):
        if "resulting_key" in env_graph.nodes[node]:
            cluster = env_graph.nodes[node]["resulting_key"]
        else:
            cluster = cluster_idx + 1
        env_graph.nodes[node]["cluster"] = cluster

        for desc in nx.algorithms.dag.descendants(env_graph, node):
            env_graph.nodes[desc]["cluster"] = cluster

    return env_graph


def get_structure_properties(structure):
    """
    From dictionary from JSON file provided in experiment,
    reshapes and outputs property dictionary with:
    * resulting_key: dictionary key you need to press to get to node
    * initial: scalar, the initial node
    * layout: dictionary with the values being coordinates of each point
    """
    structure_properties = {}

    resulting_dict = {}
    for init_node, next_node_dict in structure["graph"].items():
        for key, next_node in next_node_dict.items():
            resulting_dict[int(next_node[1])] = key
    structure_properties["resulting_key"] = resulting_dict

    structure_properties["layout"] = {int(k): v for k, v in structure["layout"].items()}

    structure_properties["initial"] = int(structure["initial"])
    if structure_properties["initial"] not in structure_properties["resulting_key"]:
        structure_properties["resulting_key"][structure_properties["initial"]] = "init"

    return structure_properties
