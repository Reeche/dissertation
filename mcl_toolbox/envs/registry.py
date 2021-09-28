class Env(object):
    def __init__(
        self, name, branching, reward_inputs, reward_dictionary, initial_node_value=0
    ):
        self.name = name
        self.branching = branching
        self.reward_inputs = reward_inputs
        self.reward_dictionary = reward_dictionary
        self.initial_node_value = initial_node_value

    def reward_function(self, *args):
        try:
            if len(args) == 1:
                # initial node
                if args[0] == 0:
                    return self.initial_node_value
                return self.reward_dictionary[args[0]]
            else:
                # reward doesn't depend on just depth, or depends on nothing (input: ())
                return self.reward_dictionary[args]
        except KeyError as exception:
            raise (exception)

    def __repr__(self):
        return_string = "\nname:" + self.name + "\n"
        return_string += (
            "branching: " + "-".join([str(entry) for entry in self.branching]) + "\n"
        )
        return_string += "inputs: ("
        return_string += ", ".join(self.reward_inputs) + ")\n"
        for key, val in self.reward_dictionary.items():
            return_string += "\t" + str(key) + ", " + str(val) + "\n"
        return return_string


class Registry(object):
    def __init__(self):
        self.envs = {}

    def register(self, **kwargs):
        if "name" not in kwargs:
            raise ValueError("No name provided for environment setting")
        elif "name" in self.envs:
            raise
            ValueError("This environment name is already present in the registry")
        else:
            self.envs[kwargs["name"]] = Env(**kwargs)

    def __repr__(self):
        complete_text = ""
        for val in self.envs.values():
            complete_text += val.__repr__()
            complete_text += "++++++++++++++++++++++++++++++++++"
        return complete_text

    def get_env(self, name):
        try:
            return self.envs[name]
        except Exception as exception:
            raise (exception)

    __call__ = get_env


# initializes registry itself
registry = Registry()


def register(**kwargs):
    registry.register(**kwargs)
