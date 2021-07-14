import itertools as it
import heapq
import numpy as np

# ---------- Functional utils ---------- #
from toolz.curried import curry, compose
from mcl_toolbox.utils.analysis_utils import get_data

max = curry(max)
min = curry(min)
call = lambda f: f()


@curry
def attr(name, obj):
    return getattr(obj, name)


@curry
def invoke(name, obj):
    return getattr(obj, name)()


lmap = curry(compose(list, map))
amap = curry(compose(np.array, lmap))


# ---------- Other ---------- #
def str_join(args, sep=" "):
    return sep.join(map(str, args))


def dict_product(d):
    """All possible combinations of values in lists in `d`"""
    for k, v in d.items():
        if not isinstance(v, list):
            d[k] = [v]

    for v in list(it.product(*d.values())):
        yield dict(zip(d.keys(), v))


def cum_returns(rewards):
    return np.flip(np.cumsum(np.flip(rewards, 0)), 0)


def clear_screen():
    print(chr(27) + "[2J")
    # clear_output() #todo: what is this function doing and where it is?


def softmax(x, temp=1):
    ex = np.exp((x - x.max()) / temp)
    return ex / ex.sum()


class Labeler(object):
    """Assigns unique integer labels."""

    def __init__(self, init=()):
        self._labels = {}
        self._xs = []
        for x in init:
            self.label(x)

    def label(self, x):
        if x not in self._labels:
            self._labels[x] = len(self._labels)
            self._xs.append(x)
        return self._labels[x]

    def unlabel(self, label):
        return self._xs[label]

    __call__ = label


class PriorityQueue(list):
    def __init__(self, key, max_first=True):
        self.key = key
        self.inv = -1 if max_first else 1

    def pop(self):
        return heapq.heappop(self)[1]

    def push(self, item):
        heapq.heappush(self, (self.inv * self.key(item), item))


def get_all_pid_for_env(exp_num):
    """get a list of all pid for a certain condition"""
    if exp_num == "c2.1_dec":
        exp_num = "c2.1"
    data = get_data(exp_num)
    return list(data["participants"]["pid"])
