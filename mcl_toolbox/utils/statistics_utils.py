def replace_none_with_empty_str(some_dict):
    return {k: (0 if v is None else v) for k, v in some_dict.items()}

# create empty dictionary so the clusters can be compared
def create_comparable_data(proportions, len):
    """
    This function creates clusters/decision system of equal length so that they can be compared.
    E.g. cluster_a = {1: 0.1, 3:0.5, 12:0.2}
    None values will be replaced by 0.

    Args:
        proportions: which porportions, e.g. cluster or strategy
        len: number of the cluster (13) or strategy (89)

    Returns: dict of certain length. For the example: {1: 0.1, 2: 0,..., 12:0.3, 13:0}

    """
    _keys = list(range(0, len))
    _dict = {key: None for key in _keys}

    for keys, values in _dict.items():
        value_new = proportions.get(keys, None)
        if value_new:
            _dict[keys] = value_new

    _dict = replace_none_with_empty_str(_dict)
    return _dict





