from copy import deepcopy
from collections import deque


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    q = deque()
    for current in nested_list:
        if isinstance(current, list):
            q.extendleft(reversed(current))
        else:
            q.appendleft(current)
        while q:
            toFlat = q.popleft()
            if isinstance(toFlat, list):
                q.extendleft(reversed(toFlat))
            else:
                yield toFlat

#------------------------------------------------------------------------------

def get_key_for_grouping(item):
    """
    Returns the key for grouping params during update of SqE.

    Parameters
    ----------
    item    : tuple
        (key, value) as given by dict.items()
    """
    try:
        return item[0].split("_")[1] # This should be the name of the line
    except IndexError:
        return "model_params"